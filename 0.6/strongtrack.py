import cv2
import numpy as np
import dlib
import time
import render_functions as render
import xml_functions as rx
import decomp_functions as decomp
import os
import sys
import eyeblob_functions as eb
from pythonosc import udp_client
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QInputDialog, QWidget, QLineEdit, QComboBox, QSlider
from PyQt5.QtCore import QSize, QRect, Qt

def getEasyBox(frame):
    padding = int(frame.shape[1]/10.8)
    box = dlib.rectangle(padding,padding,frame.shape[1]-padding, frame.shape[0]-padding)
    return box

def findThreshold(frame, points, eye, config):
    found = False
    
    for i in range(100):
        if config == 'pupil':
            eye_left1 = eb.getPupil(frame, points, eye, i*10)
        else:
            eye_left1 = eb.getIris(frame, points, eye, i*10)
        
        if found == False:
            if eye_left1[3]==True:
                found = True
                low = i*10
                
        if found == True: 
            if eye_left1[3] == False:
                found = False
                
                high = i*10
                break

    auto = ((low+high)/2)
    return auto

def findLandmarksEyes(frame, predictor):
    global lpupilthreshold
    global rpupilthreshold
    global liristhreshold
    global riristhreshold

    box = getEasyBox(frame)  
    frame = render.drawBox(frame, box)
    
    shape = predictor(frame, box)
    points = []
    
    for i in range(70):
        coords = []

        coords.append((shape.part(i).x))
        coords.append((shape.part(i).y))
        points.append(coords)

    if frame_num == 1:
        try:
            lpupilthreshold = findThreshold(frame, points, 'left', 'pupil')
            rpupilthreshold = findThreshold(frame, points, 'right', 'pupil')
            liristhreshold = findThreshold(frame, points, 'left', 'iris')
            riristhreshold = findThreshold(frame, points, 'right', 'iris')
        except:
            print('Auto threshold detection failed. As of 0.5 try to ensure first frame of current footage has good eye placement')
    
    framel, pupil_l, amountl, foundl = eb.getPupil(frame, points, 'left', lpupilthreshold)
    framer, pupil_r, amountr, foundr = eb.getPupil(frame, points, 'right', rpupilthreshold)

    if foundl == True:
        points[68] = pupil_l
    else:
        if foundr == True:
            points[68] = eb.findEyePointsFromAmount(amountr, points, 'left')
        else:
            iris_l = eb.getIris(frame, points, 'left', liristhreshold)
            if iris_l[3] == True:
                points[68] = iris_l[1]


    if foundr == True:
        
        points[69] = pupil_r
    else:
        if foundl == True:
            
            points[69] = eb.findEyePointsFromAmount(amountl, points, 'right')
        else:
            iris_r = eb.getIris(frame, points, 'right', riristhreshold)
            
            if iris_r[3] == True:
                points[69] = iris_r[1]
        
    return points

def on_trackbar(val):
    global frame_num
    global points
    global frame_store
    global frame
    global win2
    
    if play == False:
        
        win2.cap.set(cv2.CAP_PROP_POS_FRAMES,val)
        ret, frame = win2.cap.read()
        frame = cv2.resize(frame, (int(frame.shape[1]*win2.resizeFac), int(frame.shape[0]*win2.resizeFac)))
        frame_store = frame
        frame_num=val
        
        if frame_num == win2.length:
            print('no frame')
        else:
            if model == True:

                points = findLandmarksEyes(frame, win2.predictor)
            
            updateFramePoints(points)
        
    pass


def pick_circle(event,x,y,flags,param):
    global points
    global lock
    
    if event == cv2.EVENT_RBUTTONDOWN:
        if play == False:
            
            entry = (x,y)
            points2 = np.array(points)
            for i in range(points2.shape[0]):
                sub = np.subtract(entry, points2[i][0:2])
                dist =(sum(abs(sub)))
                
                if dist <=10:
                    render.activatePortion(i, active_points)
                    active[0]=i
                    lock = (x,y)
                       
                    updateFramePoints(points)    
                    cv2.setMouseCallback('Video',place_circle)  
    
    if event == cv2.EVENT_LBUTTONDOWN:
       
        entry = (x,y)
        points2 = np.array(points)
        for i in range(points2.shape[0]):
            sub = np.subtract(entry, points2[i][0:2])
            dist =(sum(abs(sub)))
                
            if dist <=10:
                active[0] = i
                lock = (x,y)
                
                updateFramePoints(points)
                cv2.setMouseCallback('Video',place_circle)
                
    pass

# Function for placing points that have been picked up.
def place_circle(event,x,y,flags,param):
    global points
    global active_points
    global frame
    
    delta = (lock[0]-x,lock[1]-y)
    
    if (sum(active_points)) == 0.0:
        pointsmove = render.movePoints(points, delta, active[0])
    else:
        pointsmove = render.movePointsMultiple(points, active_points, delta)
    
    updateFramePoints(pointsmove)
    
    if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN:  
        active_points = np.zeros([68])
        active[0] = 999
        points = pointsmove
        updateFramePoints(points) 
        cv2.setMouseCallback('Video',pick_circle)
      
    pass

#Handles update of frame for mouse events
def updateFramePoints(points):
    global frame_store
    global frame
    
    frame = np.array(frame_store)
    frame = render.drawFace(frame, points, 'full')
    points = np.array(points)
    
    frame = render.drawControlPoints(points, frame, active)

def trainModel(xml_path):
    global win2
    
    check = rx.convertXMLPoints(xml_path)
    
    if check.shape[0] == 1:
        win2.options.oversampling_amount = 1
    else:
        win2.options.oversampling_amount = int((600/check.shape[0])+1)
        
    dlib.train_shape_predictor(xml_path, win2.predictor_name, win2.options)
    predictor = dlib.shape_predictor(win2.predictor_name)
    
    return predictor

#For streaming
ip = "127.0.0.1"
port = 5005
client = udp_client.SimpleUDPClient(ip, port)
morphs = np.zeros((50))
base = np.load('data/base.npy')
cindices = np.array([np.zeros((50)),base[19],base[30]+base[31],base[30],base[31],base[36]+base[37],base[41],base[40],base[16]+base[17]+base[18],base[15]+base[14],base[0]+base[1]])

#Eye blob thresholds (need to tidy)
lpupilthreshold = 30
rpupilthreshold = 30
liristhreshold = 140
riristhreshold = 140

video_path = 'empty'
xml_path = 'empty'
project_name = 'empty'

# GET HOTKEY GUIDE
hotkeys = cv2.imread('projects/images/hotkeys.jpg')

#VARIABLES
#For tracking which point/s is being manipulated
active = np.array([999])
active_points = np.zeros([68])

#To track initial mouse click location
lock = (0,0)
        
#Variables
frame_num = 0
frame = []
points = []
frame_store = np.array(frame)

#State checks
play = True
model = True

#Converts mouth coeffs to 51 morph targets

def coeffsToMorphs(coeffs):
    global win2
    morphs = np.zeros((50))
        
    for i in range(coeffs.shape[0]):
        poseindex = win2.setposes[i]

        #Skip the first because it represents neutral pose (not in morphs)
        if poseindex != 0: 
                
            cindex = cindices[poseindex]*coeffs[i]
            morphs = morphs + cindex
            
    points2 = np.array(points)
    
    #seal the lips if points indicate sealed lips
    dist = sum(abs(points2[66]-points2[62]))

    #Distance within which to gradually seal lips. 30 at width of 570
    width_points = 570/(points[16][0]-points[0][0]) 
    zone = 30/width_points
  
    if dist <= zone:
        dist = (zone-dist)/zone
      
        morphs[20] = morphs[19]*dist
        
    return morphs

#### MAIN LOOP ####
def playVideo():
    global win2
    global play
    global frame
    global model
    global frame_num
    global points
    global frame_store
     
    length = int(win2.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ret, frame = win2.cap.read()
 
    #SET UP DISPLAY
    cv2.namedWindow('Video')
    cv2.namedWindow('Hotkeys')
    cv2.imshow('Hotkeys', hotkeys)
    #Set Window size to avoid incorrectly drawn trackbar.
    cv2.resizeWindow('Video', int(frame.shape[1]*win2.resizeFac), int(frame.shape[0]*win2.resizeFac))
    cv2.createTrackbar('Frame', 'Video' , 0, win2.length, on_trackbar)
    cv2.setMouseCallback('Video',pick_circle)
    
    win2.cap.set(cv2.CAP_PROP_POS_FRAMES,0)
    
    while(1):
        
        # If video is playing get the frame and advance the trackbar.
        
        if play ==True:
            
            frame_num = frame_num+1
            cv2.setTrackbarPos('Frame', 'Video', frame_num)
            
            if frame_num == win2.length:
                win2.cap.set(cv2.CAP_PROP_POS_FRAMES,0)
                frame_num = 1
            
            ret, frame = win2.cap.read()
            frame = cv2.resize(frame, (int(frame.shape[1]*win2.resizeFac), int(frame.shape[0]*win2.resizeFac))) 
              
            #Store the frame for quick retrieval
            frame_store = np.array(frame)
            
            #If model was found, use associated predictor file to find landmarks
            if model ==True:
                points = findLandmarksEyes(frame, win2.predictor)
                
                frame = render.drawFace(frame, points, 'full') 
                
            
            #If model wasn't found set points to generic render for manual placement
            if model ==False:
                                
                points = win2.genericFace

                frame = render.drawFace(frame, points, 'full')
                
            if win2.stream == True:
              
                coeffs2, _, _, _ = decomp.findCoeffsAll(points,win2.keyposes)
                coeffs = coeffs2[0]
                
                morphs =coeffsToMorphs(coeffs)
                eye_lamount, _ = eb.findEyeVals(points[68], points, 'left')
                eye_ramount, _ = eb.findEyeVals(points[69], points, 'right')
                eye_amounts = (eye_lamount+eye_ramount)
                eye_amounts = ((eye_lamount[0]+eye_ramount[0])/2),((eye_lamount[1]+eye_ramount[1])/2)
                morphs = np.append(morphs, eye_amounts)
                client.send_message("/filter", morphs)
                
        #Display frame on screen    
        cv2.imshow('Video', frame)

        if win2.record == True:
            break
        
        #### HANDLES HOTKEY INPUTS ####
        
        k = cv2.waitKey(10)        
        
        if k==27: #ESC KEY - QUIT
            break
        else:
            if k==-1:
                pass
        
        if k==116: #T KEY = TRAIN MODEL
            print('training model')
            win2.predictor = trainModel(win2.xml_path)
            print('done')
            
        #Play and Pause
        if k==32: #SPACEBAR
            if play ==False:
                play=True
                
            else:
                play=False

                #Show movable points
                updateFramePoints(points)

        if k==110: # N KEY - Set lips to neutral
            if model == True:
                if play == False:
                    print('Lips set to neutral')
                    points = rx.setNeutral(win2.xml_path, points)
                    updateFramePoints(points)
                    
        if k==119: # W KEY - Weld lips
            
            if play == False:
                print('welding')
                points = render.weldLips(points)
                updateFramePoints(points)
                     
        
        if k==102: # F KEY - ADD TO XML
            if play ==False:
                
                box = getEasyBox(frame)
                filename = os.path.splitext(os.path.split(win2.video_path)[1])[0]
                filepath = 'images/{}_frame{}.jpg'.format(filename,frame_num)
                
                cv2.imwrite('projects/' + filepath, frame_store)
                rx.appendXML(points, box, filepath, win2.xml_path)
                print('added')

                #If this is the first entry, train the predictor using the first entry.
                if model ==False:
                    print('Training initial model')
                    dlib.train_shape_predictor(win2.xml_path, win2.predictor_name, win2.options)
                    win2.predictor = dlib.shape_predictor(win2.predictor_name)
                    print('done')
                    model = True
                    
            else:
                print('can only add when paused!')

    cv2.destroyAllWindows()

    # Export coeffs to text file
    if win2.record == True:
        print('exporting to text file')
        win2.cap.set(cv2.CAP_PROP_POS_FRAMES,0)
        frame_num = 0
        morphs_store =[]
        print('running')
        while(1):
            
            frame_num = frame_num +1
            ret, frame = win2.cap.read()
            if ret == True:
                points = findLandmarksEyes(frame, win2.predictor)
                coeffs2, _, _, _ = decomp.findCoeffsAll(points,win2.keyposes)
                coeffs = coeffs2[0]
                morphs =coeffsToMorphs(coeffs)
                morphs_store.append(morphs)
            else:
                break
                
        
        morphs_store = np.array(morphs_store)
        
        export_filename = 'projects/'+ os.path.splitext(os.path.split(win2.video_path)[1])[0]+ '_morphs.txt'
        np.savetxt(export_filename, morphs_store)
        print('export successful')
        
    win2.cap.release()
    
    
class MyWindow2(QMainWindow):
    
    def __init__(self):
        super(MyWindow2,self).__init__()
        self.stream = False
        self.record = False
        self.dropOptions = ["Neutral","JawOpen","Closed Smile","Smile L","Smile R","Frown", "Funnel", "Pucker", "Brows Up", "Brows Down", "Eyes Closed"]
        self.keydrops = np.zeros((11,70,2))
        self.initUI()

    def checkPredictor(self):
        global model
     
        #TRAINING
        self.predictor_name = 'projects/' + self.project_name + '_model.dat'
        self.options = dlib.shape_predictor_training_options()
        self.options.oversampling_amount = 300
        self.options.nu = 0.05
        self.options.tree_depth = 2
        
        #Check for model
        try:
            check = rx.convertXMLPoints(self.xml_path)
        except:
            rx.makeXML(self.xml_path)
            print('No XML file found for project name. Making new one')
            check = rx.convertXMLPoints(self.xml_path)

        if check.shape[0] == 0:
            print('no face values found in xml. Setting generic face')
            model = False
            self.genericFace = np.load('data/baseface.npy')    

            #Get width for generic face
            self.cap = cv2.VideoCapture(self.video_path)
            ret, frame = self.cap.read()
            width = (frame.shape[1]*self.resizeFac)
            
            #Comparing against original dimensions of 'generic face' footage
            widthfactor = (1080/width)           
            self.genericFace = np.divide(self.genericFace, [widthfactor,widthfactor]).astype(int)
            print('done')
            
        else:
            print('model with values found')
            
            self.predictor = dlib.shape_predictor(self.predictor_name)
            
    def setUpVideo(self):
        width, height = screen_resolution.width(), screen_resolution.height()
        self.cap = cv2.VideoCapture(self.video_path)
        self.length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ret, frame = self.cap.read()
        self.resizeFac = height/frame.shape[0]*0.8
    
    def beginSession(self):
        self.setUpVideo()
        self.checkPredictor()
        playVideo()               
        self.close()
        
    def addKeypose(self):
        
        name = self.comboBox.currentText()
        
        index = self.dropOptions.index(name)
        
        self.keydrops[index] = points
        export_filename = 'projects/'+ self.project_name + '_keyposes.npy'
        np.save(export_filename, self.keydrops)

    def streamOSC(self):
        
        import_filename = 'projects/'+ self.project_name + '_keyposes.npy'
        
        if os.path.exists(import_filename):
            keypose_data = np.load(import_filename)
            
            self.setposes = []
            for i in range(keypose_data.shape[0]):
                if sum(sum((keypose_data[i]))) != 0.0:
                    self.setposes.append(i)

            keyposes = []
            for entry in self.setposes:
                keyposes.append(keypose_data[entry])
            
            self.keyposes = np.array(keyposes)
            
            self.stream = True
            self.b5.setEnabled(False)
            self.comboBox.setEnabled(False)
            
        else:
            print("Keyposes file not found. Can't stream")
    
    def newModel(self):
               
        text, okPressed = QInputDialog.getText(self, "Get text","Enter a project name (no space):", QLineEdit.Normal, "")

        if okPressed and text != '':            
            xml_path_check = 'projects/' + text + '_source.xml'
            
            #Check if file already exists
            check = os.path.exists(xml_path_check)
            
            if check == False:
                self.project_name = text
                self.xml_path = xml_path_check
                self.b2.setEnabled(False)
                self.b3.setEnabled(False)
                self.b5.setEnabled(True)
                self.b6.setEnabled(True)
                self.comboBox.setEnabled(True)
                self.b7.setEnabled(True)
                self.beginSession()
               
            else:
                print('project file already exists. Pick another name')
                
            
    def loadModel(self):
       
        fileName, _ = QFileDialog.getOpenFileName(self,"Load model XML", "","XML Files (*.xml);")

        #Checks if it is associated with strongtrack    
        if fileName:
            check = rx.verifyXML(fileName)
            if check == True:
                self.xml_path = fileName
                
                self.project_name = os.path.splitext(os.path.split(fileName)[1])[0]
                self.project_name = self.project_name[:-7]
                self.b2.setEnabled(False)
                self.b3.setEnabled(False)
                self.b5.setEnabled(True)
                self.comboBox.setEnabled(True)
                self.b6.setEnabled(True)
                self.b7.setEnabled(True)
                self.beginSession()
            else:
                print('This XML file is not associated with strong track.')
            
    def openFileNameDialog(self):
                
        fileName, _ = QFileDialog.getOpenFileName(self,"Open Video", "","All Files (*);")
        if fileName:
            check = rx.verifyVideo(fileName)
            if check == True:                
                self.video_path = fileName
                self.b1.setEnabled(False)
                self.b3.setEnabled(True)
                self.b2.setEnabled(True)
            else:
                print('file extension not valid')

    def printVal(self, value):
        print(value)

    def exportTxt(self):
        
        import_filename = 'projects/'+ self.project_name + '_keyposes.npy'
        
        if os.path.exists(import_filename):
            keypose_data = np.load(import_filename)
            
            self.setposes = []
            for i in range(keypose_data.shape[0]):
                if sum(sum((keypose_data[i]))) != 0.0:
                    self.setposes.append(i)
            print(self.setposes)

            keyposes = []
            for entry in self.setposes:
                keyposes.append(keypose_data[entry])
            
            self.keyposes = np.array(keyposes)
            
            self.record = True
        else:
            print("Keyposes not found. Can't export")
    
    def initUI(self):
        self.setGeometry(200, 200, 200, 450)
        self.setWindowTitle("StrongTrack 0.6")
        
        self.b1 = QtWidgets.QPushButton(self)
        self.b1.setText("Open Video")
        self.b1.move(50,30)
        self.b1.clicked.connect(self.openFileNameDialog)
        
        self.b2 = QtWidgets.QPushButton(self)
        self.b2.setText("New Model")
        self.b2.move(50,80)
        self.b2.clicked.connect(self.newModel)
        self.b2.setEnabled(False)
        
        self.b3 = QtWidgets.QPushButton(self)
        self.b3.setText("Load Model")
        self.b3.move(50,130)
        self.b3.clicked.connect(self.loadModel)
        self.b3.setEnabled(False)
        '''
        self.sl = QSlider(Qt.Horizontal, self)
        self.sl.setGeometry(50, 380, 100, 30)
        self.sl.setMinimum(0)
        self.sl.setMaximum(100)
        self.sl.setValue(0)
        self.sl.valueChanged[int].connect(self.printVal)'''
        
        self.comboBox = QComboBox(self)
        self.comboBox.setGeometry(QRect(40, 280, 120, 31))
        self.comboBox.setObjectName(("comboBox"))
        for dropOption in self.dropOptions:
            self.comboBox.addItem(dropOption)
        self.comboBox.setEnabled(False)
         
        self.b5 = QtWidgets.QPushButton(self)
        self.b5.setText("Set Keypose")
        self.b5.move(50,230)
        self.b5.clicked.connect(self.addKeypose)
        self.b5.setEnabled(False)

        self.b6 = QtWidgets.QPushButton(self)
        self.b6.setText("Stream OSC")
        self.b6.move(50,330)
        self.b6.clicked.connect(self.streamOSC)
        self.b6.setEnabled(False)

        self.b7 = QtWidgets.QPushButton(self)
        self.b7.setText("Export to Txt")
        self.b7.move(50,380)
        self.b7.clicked.connect(self.exportTxt)
        self.b7.setEnabled(False)

app = QApplication(sys.argv)
screen_resolution = app.desktop().screenGeometry()
win2= MyWindow2()
win2.show()
app.exec_()
