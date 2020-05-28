import cv2
import numpy as np
import dlib
import time
import render_functions as render
import xml_functions as rx
import decomp_functions as decomp
import os
import sys
import argparse
import eyeblob_functions as eb
from pythonosc import udp_client

parser = argparse.ArgumentParser(description='StrongTrack 0.4')

parser.add_argument('--video', help='enter video path')
parser.add_argument('--project_name', help='enter name for project')

args = parser.parse_args()

#Eye blob thresholds
lpupilthreshold = 30
rpupilthreshold = 30
liristhreshold = 140
riristhreshold = 140

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


def findLandmarksEyes(frame):
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
            print('Auto threshold detection failed. As of 0.4 try to ensure first frame of current footage has good eye placement')
    
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
    if play == False: 
        cap.set(cv2.CAP_PROP_POS_FRAMES,val)
        ret, frame = cap.read()
        frame_store = frame
        frame_num=val
        if frame_num == length:
            print('no frame')
        else:
            if model == True:

                points = findLandmarksEyes(frame)
            
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
    
    frame = np.array(frame_store)
    frame = render.drawFace(frame, points, 'full')
    points = np.array(points)
    
    frame = render.drawControlPoints(points, frame, active)
        
    cv2.imshow("Video", frame)

    pass

def trainModel(xml_path, predictor):
    check = rx.convertXMLPoints(xml_path)
    if check.shape[0] == 1:
        options.oversampling_amount = 1
    else:
        options.oversampling_amount = int((600/check.shape[0])+1)
        
    dlib.train_shape_predictor(xml_path, predictor_name, options)
    predictor = dlib.shape_predictor(predictor_name)
    return predictor

#TRAINING
xml_path = 'projects/' + args.project_name+'_source.xml'
predictor_name = 'projects/' + args.project_name + '_model.dat'
options = dlib.shape_predictor_training_options()
options.oversampling_amount = 300
options.nu = 0.05
options.tree_depth = 2

#OSC
ip = "127.0.0.1"
port = 5005
client = udp_client.SimpleUDPClient(ip, port)

#VARIABLES
#For tracking which point/s is being manipulated
active = np.array([999])
active_points = np.zeros([68])
#For holding frame data
frame_store = []
#To track initial mouse click location
lock = (0,0)

try:
    streamPoses = np.load('projects/'+args.project_name+'_poses.npy')
    print('Extraction pose file loaded')
except:
    print('No extraction poses found')
    streamPoses = np.zeros((5,70,2))
        
# GET VIDEO

video_path = args.video
hotkeys = cv2.imread('projects/images/hotkeys.jpg')
cap = cv2.VideoCapture(video_path)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#Get width for generic face
ret, frame = cap.read()
width = (frame.shape[1])
#Comparing against original dimensions of 'generic face' footage
widthfactor = (1080/width)

#SET UP DISPLAY
cv2.namedWindow('Video')
cv2.namedWindow('Hotkeys')
cv2.imshow('Hotkeys', hotkeys)
#Set Window size to avoid incorrectly drawn trackbar.
cv2.resizeWindow('Video', frame.shape[1], frame.shape[0])
cv2.createTrackbar('Frame', 'Video' , 0, length, on_trackbar)
cv2.setMouseCallback('Video',pick_circle)

#Track frame number
frame_num = 0

#State checks
play = True
model = True
stream = False
export = False

#Check for model
try:
    check = rx.convertXMLPoints(xml_path)
except:
    rx.makeXML(xml_path)
    print('No XML file found for project name. Making new one')
    check = rx.convertXMLPoints(xml_path)

if check.shape[0] == 0:
    print('no face values found in xml. Setting generic face')
    model = False
    genericFace = np.load('robasbaseeyes.npy')    
    
    genericFace = np.divide(genericFace, [widthfactor,widthfactor]).astype(int)
else:
    print('model with values found')
    predictor = dlib.shape_predictor(predictor_name)


#### MAIN LOOP ####

cap.set(cv2.CAP_PROP_POS_FRAMES,0)
    
while(1):
    
    # If video is playing get the frame and advance the trackbar.
    
    if play ==True:        
        frame_num = frame_num+1
        cv2.setTrackbarPos('Frame', 'Video', frame_num)
        
        if frame_num == length:
            cap.set(cv2.CAP_PROP_POS_FRAMES,0)
            frame_num = 1
        
        ret, frame = cap.read()

        #Store the frame for quick retrieval
        frame_store = np.array(frame)

        #If model was found, use associated predictor file to find landmarks
        if model ==True:
            points = findLandmarksEyes(frame)

            frame = render.drawFace(frame, points, 'full')
            
            prev_points = points
            
        #If model wasn't found set points to generic render for manual placement
        if model ==False:
            
            points = genericFace

            frame = render.drawFace(frame, points, 'full')
            
        
        #Display the video on screen    
        cv2.imshow('Video', frame)

        #Stream the video via OSC
        if model== True:
            if stream == True:
                mouth, browcoeff, blinkcoeff, squintcoeff = decomp.findCoeffsAll(points, streamPoses)
                #print(mouth)
                client.send_message("/filter", (mouth[0][0],mouth[0][1],mouth[0][2],mouth[0][3],mouth[0][4]))
                                                
    #### HANDLES INPUTS - NEEDS TO BE CLEANED UP ####
    
    k = cv2.waitKey(10)
    
    if k==27: #ESC KEY - QUIT
        break
    else:
        if k==-1:
            pass
        else:
            if model == True:
                streamPoses = decomp.setKeyPose(k, points, streamPoses)
    
    if k==116: #T KEY = TRAIN MODEL
        print('training model')
        predictor = trainModel(xml_path, predictor)
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
                points = rx.setNeutral(xml_path, points)
                updateFramePoints(points)
                
    if k==119: # W KEY - Weld lips
        
        if play == False:
            print('welding')
            points = render.weldLips(points)
            updateFramePoints(points)
                 

    if k == 114: #R Key - EXPORT COEFFS 
        if model == True:
            if stream == False:
                
                if decomp.checkKeyPoses(streamPoses) == True:
                    export = True
                    break
                
    if k == 101: #E Key -    
        if model == True:
            if stream == False:
                check = decomp.checkKeyPoses(streamPoses)
                if check == True:
                    stream = True
        
    if k==102: # F KEY - ADD TO XML
        if play ==False:
            
            box = getEasyBox(frame)
            filename = os.path.splitext(os.path.split(video_path)[1])[0]
            filepath = 'images/{}_frame{}.jpg'.format(filename,frame_num)
            cv2.imwrite('projects/' + filepath, frame_store)
            rx.appendXML(points, box, filepath, xml_path)
            print('added')

            #If this is the first entry, train the predictor using the first entry.
            if model ==False:
                print('Training initial model')
                dlib.train_shape_predictor(xml_path, predictor_name, options)
                predictor = dlib.shape_predictor(predictor_name)
                print('done')
                model = True
                
        else:
            print('can only add when paused!')
    

cv2.destroyAllWindows()

# Export coeffs to text file
if export == True:
    print('exporting coeffs to text file')
    cap.set(cv2.CAP_PROP_POS_FRAMES,0)
    frame_num = 0
    coeffs_store =[]
    
    while(1):
        frame_num = frame_num +1
        ret, frame = cap.read()
        points = findLandmarksEyes(frame)
        mouth_coeffs, browcoeff, blinkcoeff, squintcoeff = decomp.findCoeffsAll(points, streamPoses)
        coeffs_store.append(mouth_coeffs)
        if frame_num == length:
            break

    print('export successful')
    
    coeffs_store = np.array(coeffs_store)
    coeffs_store = np.reshape(coeffs_store, (coeffs_store.shape[0],5))
    print(coeffs_store.shape)
    export_filename = 'projects/'+ os.path.splitext(args.video)[0]+ '_coeffs.txt'
    np.savetxt(export_filename, coeffs_store)

cap.release()

# Save the poses
streamPoses = np.array(streamPoses)
if streamPoses.shape[0]!=0:
    np.save('projects/' + args.project_name + '_poses.npy', streamPoses)
