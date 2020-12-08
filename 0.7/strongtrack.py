from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QSlider, QPushButton
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread


import render_functions as render
import xml_functions as rx
import decomp_functions as decomp

from pythonosc import udp_client
import numpy as np
import cv2
import dlib
import os

# Get the GUI
from ui import Ui_MainWindow

# Frame as original output. Used for calculating scales.
frame_raw = []

# Frame once upscaled and with added graphics
frame_scaled = []

# Frame after upscale but held for when graphics need refreshing
frame_store = np.array(frame_raw)

loc = (0,0)
play = False
showPoints = False

frame_num = 0
active = np.array([999])
active_points = np.zeros([68])
factor = 1
points = []
pointsmove = []

#To track initial mouse click location
lock = (0,0)

#For streaming
ip = "127.0.0.1"
port = 5005
client = udp_client.SimpleUDPClient(ip, port)
morphs = np.zeros((50))

def coeffsToMorphs(coeffs, browcoeffs, points):
    global window

    #Get mouth coeffs
    mouthcoeffs = coeffs[0]

    base = np.load('data/base.npy')

    # Matched against the key drops so that set poses can be used to convert to morphs. The first is a placeholders due to them being neutral
    cindices = np.array([np.zeros((50)), base[19], base[30] + base[31], base[30], base[31], base[36] + base[37], base[41], base[40], base[14], base[15], base[16], base[17],base[18]])

    #Set as zero to begin with
    morphs = np.zeros((50))

    for i in range(mouthcoeffs.shape[0]):
        poseindex = window.setposes[i]

        # Skip the first because it represents neutral pose or brows (not in morphs). This is probably unncessary due to them being zeros but....whatever
        if poseindex != 0 and poseindex != 8 and poseindex != 9:

            cindex = cindices[poseindex] * mouthcoeffs[i]

            morphs = morphs + cindex

    points2 = np.array(points)

    # seal the lips if points indicate sealed lips
    dist = sum(abs(points2[66] - points2[62]))
    
    # Distance within which to gradually seal lips. 30 at width of 570
    width_points = 570 / (points[16][0] - points[0][0])

    zone = 30 / width_points
    #print(width_points, zone, dist)
    if dist <= zone:
        dist = (zone - dist) / zone

        morphs[20] = morphs[19] * dist

    #Brows
    cindex = cindices[8]*browcoeffs[0][1]
    morphs = morphs + cindex

    cindex = cindices[9]*browcoeffs[1][1]
    morphs = morphs + cindex

    browcentre = (browcoeffs[0][0]+browcoeffs[1][0])/2
    cindex = cindices[10] * browcentre
    morphs = morphs + cindex

    cindex = cindices[11]*browcoeffs[0][0]
    morphs = morphs + cindex

    cindex = cindices[12] * browcoeffs[1][0]
    morphs = morphs + cindex

    return morphs

def getEasyBox(frame):
    padding = int(frame.shape[1]/10.8)
    box = dlib.rectangle(padding,padding,frame.shape[1]-padding, frame.shape[0]-padding)
    return box

def findLandmarks(frame, predictor):
    box = getEasyBox(frame)
    frame = render.drawBox(frame, box)
    
    shape = predictor(frame, box)
    
    points = []
    
    for i in range(70):
        coords = []
        coords.append((shape.part(i).x))
        coords.append((shape.part(i).y))
        points.append(coords)

    return points

def trainModel(xml_path):
    global window
    
    check = rx.convertXMLPoints(xml_path)

    if check.shape[0] == 1:
        window.options.oversampling_amount = 1
    else:
        window.options.oversampling_amount = int((600/check.shape[0])+1)
        
    dlib.train_shape_predictor(xml_path, window.predictor_name, window.options)
    predictor = dlib.shape_predictor(window.predictor_name)
    
    return predictor

#Handles update of frame for mouse and pause events
def updateFramePoints(points):
    
    #Get scaled frame without graphics
    frame = np.array(frame_store)
    #Add lines
    frame = render.drawFace(frame, points, 'full')
    # Add control points
    frame = render.drawControlPoints(points, frame, active)

    return frame

def reposNeutral(neutral_points, points):

    #Scale neutral points to
    width_points = points[16][0] - points[0][0]
    width_neutral = neutral_points[16][0] - neutral_points[0][0]
    width_fac = width_neutral / width_points

    neutral_points = np.divide(neutral_points, [width_fac, width_fac]).astype(int)

    nose_top = points[27]
    nose_top_neutral = neutral_points[27]
    nose_delta = (nose_top - nose_top_neutral)

    neutral_points = neutral_points + nose_delta

    return neutral_points

class VideoThread(QThread):
    
    pixmap_signal = pyqtSignal(np.ndarray)
    
    def run(self):
        
        global window
        global play
        global frame_num
        global factor
    
        global points
        
        global frame_raw
        global frame_scaled
        global frame_store

        while True:
           
            if play == True:
                ret, frame_raw = window.cap.read()
                
                if ret:
            
                    if showPoints == True:

                        targetres = (int(frame_raw.shape[1]*factor),int(frame_raw.shape[0]*factor))

                        frame_scaled = frame_raw
                        frame_scaled = cv2.resize(frame_scaled, targetres)

                        #Store for quick retrieval without graphics
                        frame_store = np.array(frame_scaled)

                        if window.model == True:
                            points = findLandmarks(frame_scaled, window.predictor)
                        else:
                            points = window.genericFace*factor
                            genericfactor = window.getGenericFactor()
                            points = window.genericFace * genericfactor

                        if window.stream == True:

                            mouth_coeffs, browscoeffs, _, _ = decomp.findCoeffsAll(points, window.keyposes, window.keydrops)

                            morphs = coeffsToMorphs(mouth_coeffs, browscoeffs, points)

                            client.send_message("/filter", morphs)

                        frame_scaled = render.drawFace(frame_scaled, points, 'full')

                        self.pixmap_signal.emit(frame_scaled)
                        
                    else:
                       
                        self.pixmap_signal.emit(frame_raw)

                    frame_num = frame_num+1
                    
            k = cv2.waitKey(10)

class MainWindow(QMainWindow, Ui_MainWindow):

    def __init__(self, *args, **kwargs):
        
        super(MainWindow, self).__init__(*args, **kwargs)

        self.displayingVideo = False

        self.options = dlib.shape_predictor_training_options()
        self.options.oversampling_amount = 300
        self.options.nu = 0.05
        self.options.tree_depth = 2

        self.genericFace = np.load('data/baseface.npy')
        self.model = False
        self.stream = False
        self.keydrops = np.zeros((10, 70, 2))

        self.setupUi(self)
        
        self.display_width = self.vidholder.geometry().width()
        self.display_height = self.vidholder.geometry().height()

        #Debug
        self.overlayNeutral = False

        #Slider
        self.horizontalSlider.valueChanged.connect(self.moveBar)
        
        #Menu Buttons
        self.actionQuick_Video.triggered.connect(self.setVideo)
        self.actionOpen_Video.triggered.connect(self.openVideo)
        self.actionPrevious_Model.triggered.connect(self.debugPoints)
        self.actionNew_Model.triggered.connect(self.newModel)
        self.actionLoad_Model.triggered.connect(self.loadModel)
        self.actionExit.triggered.connect(self.quit)
        self.actionExport.triggered.connect(self.export)
        self.actionStream_OSC.triggered.connect(self.streamOSC)
        
        #Buttons
        self.button_playPause.clicked.connect(self.pause)
        self.button_prevFrame.clicked.connect(self.prevFrame)
        self.button_nextFrame.clicked.connect(self.nextFrame)
        self.button_weld.clicked.connect(self.weld)
        self.button_neutral.clicked.connect(self.neutral)
        self.button_train.clicked.connect(self.trainModel)
        self.button_landmarks.clicked.connect(self.setLandmarks)
        self.button_setKeypose.clicked.connect(self.setKeypose)
        
        #Shortcuts
        self.shortcut = QShortcut(QKeySequence("space"), self)
        self.shortcut.activated.connect(self.pause)
        self.shortcut = QShortcut(QKeySequence("w"), self)
        self.shortcut.activated.connect(self.weld)
        self.shortcut = QShortcut(QKeySequence("n"), self)
        self.shortcut.activated.connect(self.neutral)
        self.shortcut = QShortcut(QKeySequence("t"), self)
        self.shortcut.activated.connect(self.trainModel)
        self.shortcut = QShortcut(QKeySequence("f"), self)
        self.shortcut.activated.connect(self.setLandmarks)
        self.shortcut = QShortcut(QKeySequence("a"), self)
        self.shortcut.activated.connect(self.prevFrame)
        self.shortcut = QShortcut(QKeySequence("d"), self)
        self.shortcut.activated.connect(self.nextFrame)
        self.shortcut = QShortcut(QKeySequence("p"), self)
        self.shortcut.activated.connect(self.debugNeutral)
        self.show()

    def debugNeutral(self):
        self.overlayNeutral = True
        self.neutral_points = rx.convertXMLPoints(self.xml_path)[0]

    def debugPoints(self):
        global showPoints
        global factor
        
        #showPoints = True
        factor = self.label.width()/frame_raw.shape[1]

        self.checkPredictor()
        
    def moveBar(self):
        
        global play
        global window
        global frame_num
        global points
        global frame_raw
        global frame_store

        if play == True:
            play = False
        
        val = self.horizontalSlider.value()
        
        frame_num = val
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES,val)
            
        ret, frame_raw = window.cap.read()
        
        if ret:
            targetres = (int(frame_raw.shape[1]*factor),int(frame_raw.shape[0]*factor))
            frame_scaled = frame_raw
            frame_scaled = cv2.resize(frame_scaled, targetres)
            frame_store = np.array(frame_scaled)

            if showPoints == True:
                if self.model == True:
                    points = findLandmarks(frame_scaled, window.predictor)
            
                frame_scaled = updateFramePoints(points)
            
            self.update_image_paused(frame_scaled)

            if self.stream == True:
                coeffs, browcoeffs, _, _ = decomp.findCoeffsAll(points, window.keyposes, self.keydrops)

                morphs = coeffsToMorphs(coeffs, browcoeffs, points)
                client.send_message("/filter", morphs)

    def prevFrame(self):
        global frame_num
        global frame_raw
        global frame_store
        global frame_scaled
        global points


        if play == False and frame_num != 0:

            frame_num = frame_num - 1

            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

            ret, frame_raw = window.cap.read()

            if ret:

                self.horizontalSlider.setValue(frame_num)

                targetres = (int(frame_raw.shape[1] * factor), int(frame_raw.shape[0] * factor))
                frame_scaled = frame_raw
                frame_scaled = cv2.resize(frame_scaled, targetres)
                frame_store = np.array(frame_scaled)

                if showPoints == True:
                    points = findLandmarks(frame_scaled, window.predictor)

                    frame_scaled = updateFramePoints(points)

                self.update_image_paused(frame_scaled)

    def nextFrame(self):
        global frame_num
        global frame_raw
        global frame_store
        global frame_scaled
        global points

        if play == False:

            frame_num = frame_num + 1

            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

            ret, frame_raw = window.cap.read()

            if ret:

                self.horizontalSlider.setValue(frame_num)

                targetres = (int(frame_raw.shape[1] * factor), int(frame_raw.shape[0] * factor))
                frame_scaled = frame_raw
                frame_scaled = cv2.resize(frame_scaled, targetres)
                frame_store = np.array(frame_scaled)

                if showPoints == True:
                    points = findLandmarks(frame_scaled, window.predictor)

                    frame_scaled = updateFramePoints(points)

                self.update_image_paused(frame_scaled)

    def weld(self):
        global points
        if showPoints == True:

            if play ==False:
                print('welding lips')
                points = render.weldLips(points)
                frame_scaled = updateFramePoints(points)
                self.update_image_paused(frame_scaled)

    def neutral(self):
        global points
        if showPoints == True:
            if play == False:
                print('setting lips to neutral')
                points = rx.setNeutral(window.xml_path, points)
                frame_scaled = updateFramePoints(points)
                self.update_image_paused(frame_scaled)

    def trainModel(self):
        if showPoints == True:

            print('training model')
            self.predictor = trainModel(self.xml_path)
            if self.model==False:
                self.model = True
            print('done')

    def setLandmarks(self):
        if showPoints ==True:
            if play ==False:

                box = getEasyBox(frame_scaled)
                filename = os.path.splitext(os.path.split(window.video_path)[1])[0]
                filepath = 'images/{}_frame{}.jpg'.format(filename,frame_num)
                cv2.imwrite('projects/' + filepath, frame_store)
                rx.appendXML(points, box, filepath, window.xml_path)
                print('Landmark added to model')
            
    def setKeypose(self):
        self.keydrops[self.comboBox.currentIndex()] = points
        print('Added keypose: ' + self.comboBox.currentText())
        export_filename = 'projects/' + self.project_name + '_keyposes.npy'
        np.save(export_filename, self.keydrops)

    def newModel(self):
        global showPoints
        global factor
        global play

        # Ensure video is paused if one is already loaded
        play = False

        text, okPressed = QInputDialog.getText(self, "Create New Model","Enter a project name (no space):", QLineEdit.Normal, "")

        if okPressed and text != '':
            
            xml_path_check = 'projects/' + text + '_source.xml'
            # Check if file already exists
            check = os.path.exists(xml_path_check)
            
            if check == False:
                
                self.project_name = text
                self.xml_path = xml_path_check
                self.checkPredictor()
                
                factor = self.getFactor()
                               
            else:
                print('project file already exists. Pick another name')
                
    def loadModel(self):
        global play

        # Ensure video is paused
        play = False

        fileName, _ = QFileDialog.getOpenFileName(self,"Load model XML", "","XML Files (*.xml);")

        # Checks if it is associated with strongtrack

        if fileName:
            check = rx.verifyXML(fileName)
            if check == True:
                self.xml_path = fileName

                self.project_name = os.path.splitext(os.path.split(fileName)[1])[0]
                self.project_name = self.project_name[:-7]
                print(self.project_name)
                self.checkPredictor()


            else:
                print('This XML file is not associated with strong track.')

    def checkPredictor(self):
        global showPoints
        global factor
        global play
        global frame_store
        global points

        self.predictor_name = 'projects/' + self.project_name + '_model.dat'

        try:
            check = rx.convertXMLPoints(self.xml_path)
            print('XML file loaded successfully')
        except:
            rx.makeXML(self.xml_path)
            print('No XML file found for project name. Making new one')
            check = rx.convertXMLPoints(self.xml_path)

        if check.shape[0] == 0:
            print('no face values found in xml. Setting generic face')
            self.model = False
            showPoints = True

        else:
            print('model with values found')

            self.predictor = dlib.shape_predictor(self.predictor_name)
            self.model = True
            showPoints = True
            #factor = self.label.width() / frame_raw.shape[1]
            try:
                import_filename = 'projects/' + self.project_name + '_keyposes.npy'
                self.keydrops = np.load(import_filename)
                print('Keyposes extraction poses found and loaded')
            except:
                print('No keyposes for extraction found')

        self.actionStream_OSC.setEnabled(True)
        self.actionExport.setEnabled(True)

        self.button_landmarks.setEnabled(True)
        self.button_weld.setEnabled(True)
        self.button_neutral.setEnabled(True)
        self.comboBox.setEnabled(True)
        self.button_setKeypose.setEnabled(True)
        self.button_train.setEnabled(True)

        if play == False:
            print('showing points first time')

            factor = self.getFactor()

            targetres = (int(frame_raw.shape[1] * factor), int(frame_raw.shape[0] * factor))

            frame_scaled = frame_raw
            frame_scaled = cv2.resize(frame_scaled, targetres)
            frame_store = np.array(frame_scaled)

            if self.model == True:
                points = findLandmarks(frame_scaled, window.predictor)

            else:
                #points = window.genericFace * factor
                genericfactor = self.getGenericFactor()
                
                points = window.genericFace *genericfactor

            frame_scaled = render.drawFace(frame_scaled, points, 'full')
            frame_scaled = render.drawControlPoints(points, frame_scaled, active)
            self.update_image_paused(frame_scaled)

    def setVideo(self):
        global play

        # create video thread
        self.thread = VideoThread()
        
        # connect to slot
        self.thread.pixmap_signal.connect(self.update_image)
        
        # start thread
        self.thread.start()
        
        self.horizontalSlider.setMaximum(int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))

        self.horizontalSlider.setValue(0)

        #Activate menu options
        self.actionNew_Model.setEnabled(True)
        self.actionLoad_Model.setEnabled(True)
        self.actionPrevious_Model.setEnabled(True)
        self.horizontalSlider.setEnabled(True)
        self.button_playPause.setEnabled(True)
        self.button_nextFrame.setEnabled(True)
        self.button_prevFrame.setEnabled(True)

        self.displayingVideo = True
        self.pause()

    def openVideo(self):
        global factor
        global play

        # Ensure video is paused if one is already loaded
        play = False
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "All Files (*);")

        if fileName:
            checktext = rx.verifyVideo(fileName)
            if checktext == True:
                self.video_path = fileName
                self.cap = cv2.VideoCapture(self.video_path)

                # If video thread hasn't been started, start it
                if self.displayingVideo == False:
                    self.setVideo()
                else:
                    self.horizontalSlider.setMaximum(int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))
                    self.horizontalSlider.setValue(0)

                    ret, frame_raw = self.cap.read()
                    factor = self.getFactor()

                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

                    self.pause()



            else:
                print('File extension not valid. StrongTrack currently supports files with mp4, avi and mov extensions')

    def mousePressEvent(self, QMouseEvent):
        global loc
        global frame_raw
        global factor        
        global frame_scaled
        global lock
        global points
        global pointsmove
        global active_points
        
        self.setMouseTracking(True)
        self.label.setMouseTracking(True)
        
        loc = (QMouseEvent.x(),QMouseEvent.y()-21)
         
        top = (self.label.y())
        left = (self.label.x())
        right = (self.label.width()+self.label.x())
        bottom = (self.label.height()+self.label.y())
            
        if left <= loc[0] <= right and top <= loc[1] <= bottom:

            # Left click moves single
            if QMouseEvent.button() == Qt.LeftButton:

                if active[0]==999:
                    #Get location relative to picture
                    locpic = (loc[0]-left, loc[1]-top)
                    
                    #Points were found after scaling
                    points2 = np.array(points)
                    
                    for i in range(points2.shape[0]):
                        
                        sub = np.subtract(locpic, points2[i][0:2])
                        dist =(sum(abs(sub)))

                        if dist <=10:
                         
                            active[0] = i
                            lock = locpic 

                            pointsmove=points
                            
                            #Get scaled frame without graphics
                            frame_scaled = np.array(frame_store)
                            frame_scaled = render.drawFace(frame_scaled, points, 'full')
                            frame_scaled = render.drawControlPoints(points, frame_scaled, active)

                            #frame_scaled = updateFramePoints(points)
                            self.update_image_paused(frame_scaled)
                else:
             
                    active[0] = 999
                    active_points = np.zeros([68])
                    points = pointsmove
                    frame_scaled = updateFramePoints(points)
                    self.update_image_paused(frame_scaled)
                    
            # Right click moves group        
            else:
                
                if active[0]==999:
                    #Get location relative to picture
                    locpic = (loc[0]-left, loc[1]-top)
                    
                    #Points were found after scaling
                    points2 = np.array(points)

                    for i in range(points2.shape[0]):
                        
                        sub = np.subtract(locpic, points2[i][0:2])
                        dist =(sum(abs(sub)))
                        
                        if dist <=10:
                         
                            active[0] = i
                            render.activatePortion(i, active_points)
                            lock = locpic

                            pointsmove=points
                            
                            #Get scaled frame without graphics
                            frame_scaled = np.array(frame_store)
                            frame_scaled = render.drawFace(frame_scaled, points, 'full')
                            frame_scaled = render.drawControlPoints(points, frame_scaled, active)

                            #frame_scaled = updateFramePoints(points)
                            self.update_image_paused(frame_scaled)

                else:
                     
                    active[0] = 999
                    active_points = np.zeros([68])
                    points = pointsmove
                    frame_scaled = updateFramePoints(points)
                    self.update_image_paused(frame_scaled)

    def mouseMoveEvent(self, event):
        global points
        global pointsmove
        global frame_scaled
        
        top = (self.label.y())
        left = (self.label.x())
        delta = (lock[0]-event.x()+left,lock[1]-event.y()+top+21)
        
        if active[0] != 999:
                    
            if (sum(active_points)) == 0.0:
                pointsmove = render.movePoints(points, delta, active[0])
            else:
                pointsmove = render.movePointsMultiple(points, active_points, delta)
                
            frame_scaled = updateFramePoints(pointsmove)
            self.update_image_paused(frame_scaled)

    def getGenericFactor(self):
        rawwidth = frame_raw.shape[1]
        rawheight = frame_raw.shape[0]

        scaledwidth = rawwidth  * factor
        scaledheight = rawheight * factor

        scaledratio = scaledwidth/ scaledheight
        genericratio = 1080/1200

        if scaledratio >= genericratio:
            # less portrait
            genericfactor = scaledheight / 1200

        else:
            # more landscape
            genericfactor = scaledwidth / 1080

        return genericfactor

    def getFactor(self):
        global frame_raw

        holderwidth = self.vidholder.geometry().width()
        holderheight = self.vidholder.geometry().height()
        rawwidth = frame_raw.shape[1]
        rawheight = frame_raw.shape[0]

        rawratio = rawwidth / rawheight
        holderratio = holderwidth / holderheight

        # >1 means landscape
        if rawratio >= holderratio:
            factor = holderwidth / rawwidth

        if rawratio < holderratio:
            factor = holderheight / rawheight
        return factor

    def resizeEvent(self, event):
        global factor
        global frame_raw
        global showPoints
        global points
        global frame_store
        global frame_scaled

        if self.displayingVideo == True:

            factor = self.getFactor()

            #When play is true, scaling is already handled by the frame update
            if play == False:

                targetres = (int(frame_raw.shape[1]*factor),int(frame_raw.shape[0]*factor))

                frame_scaled = frame_raw
                frame_scaled = cv2.resize(frame_scaled, targetres)

                #Store for quick retrieval without graphics
                frame_store = np.array(frame_scaled)

                if self.model == True:
                    points = findLandmarks(frame_scaled, window.predictor)

                    frame_scaled = render.drawFace(frame_scaled, points, 'full')
                    frame_scaled = render.drawControlPoints(points, frame_scaled, active)

                self.update_image_paused(frame_scaled)

    def export(self):
        global play

        play = False

        fileName, _ = QFileDialog.getSaveFileName(self, "Open Video", "", "Text File (*.txt);")

        if fileName:

            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_num = 0
            morphs_store = []

            self.prepKeyposes()

            while (1):
                frame_num = frame_num + 1
                ret, frame = window.cap.read()

                if ret == True:
                    print(frame_num)
                    points = findLandmarks(frame, self.predictor)
                    mouth_coeffs, brow_coeffs, _, _ = decomp.findCoeffsAll(points,self.keyposes, self.keydrops)
                    morphs = coeffsToMorphs(mouth_coeffs, brow_coeffs, points)
                    morphs_store.append(morphs)

                else:
                    break

            morphs_store = np.array(morphs_store)
            np.savetxt(fileName, morphs_store)

            print('export successful')
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def streamOSC(self):
        
        if self.model == True:

            self.prepKeyposes()
            
            self.stream = True

        print('stream')

    def prepKeyposes(self):

        self.setposes = []

        for i in range(self.keydrops.shape[0]):
            if sum(sum((self.keydrops[i]))) != 0.0:
                self.setposes.append(i)

        keyposes = []

        for entry in self.setposes:
            keyposes.append(self.keydrops[entry])

        self.keyposes = np.array(keyposes)

    def pause(self):
        
        global play
        global frame_scaled

        if play == True:
          
            play = False
            self.button_prevFrame.setEnabled(True)
            self.button_nextFrame.setEnabled(True)

            if self.model == True:
                self.button_landmarks.setEnabled(True)
                self.button_weld.setEnabled(True)
                self.button_neutral.setEnabled(True)
                self.comboBox.setEnabled(True)
                self.button_setKeypose.setEnabled(True)
                self.button_train.setEnabled(True)

            if showPoints == True:
                frame_scaled = updateFramePoints(points)
                self.update_image_paused(frame_scaled)
                        
        else:
            play = True
           
            self.button_prevFrame.setEnabled(False)
            self.button_nextFrame.setEnabled(False)

            if self.model == True:
                self.button_landmarks.setEnabled(False)
                self.button_weld.setEnabled(False)
                self.button_neutral.setEnabled(False)
                self.comboBox.setEnabled(False)
                self.button_setKeypose.setEnabled(False)
                self.button_train.setEnabled(False)

    def quit(self):
        self.close()

    def update_image(self, cv_frame):

        qt_frame = self.convert_cv_qt(cv_frame)

        if play==True:
            self.label.setPixmap(qt_frame)
        
        if play == True:
            self.horizontalSlider.blockSignals(True)
            self.horizontalSlider.setValue(frame_num)
            self.horizontalSlider.blockSignals(False)
        else:
            if showPoints==True:
                frame_scaled = updateFramePoints(points)
                self.update_image_paused(frame_scaled)

    def update_image_paused(self, cv_frame):

        qt_frame = self.convert_cv_qt(cv_frame)
        self.label.setPixmap(qt_frame)
        
        if play == True:
            self.horizontalSlider.blockSignals(True)
            self.horizontalSlider.setValue(frame_num)
            self.horizontalSlider.blockSignals(False)
            
    def convert_cv_qt(self, cv_frame):

        rgb_image = cv2.cvtColor(cv_frame, cv2.COLOR_BGR2RGB)
      
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)

        self.display_width = self.vidholder.geometry().width()

        self.display_height = self.vidholder.geometry().height()
        
        p = convert_to_Qt_format.scaled(self.display_width, self.display_height, Qt.KeepAspectRatio)

        return QPixmap.fromImage(p)
    
if __name__ == '__main__':
    app = QApplication([])
    app.setApplicationName("StrongTrack")

    window = MainWindow()
    app.exec_()
