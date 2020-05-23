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

parser = argparse.ArgumentParser(description='StrongTrack 0.4')

parser.add_argument('--video',default='robshapes2.mp4',help='enter video path')
parser.add_argument('--project_name',default='Untitled',help='enter name for project')

args = parser.parse_args()

def getEasyBox(frame):
    padding = int(frame.shape[1]/10.8)
    box = dlib.rectangle(padding,padding,frame.shape[1]-padding, frame.shape[0]-padding)
    return box

def findLandmarks(frame):
    
    box = getEasyBox(frame)  
    frame = render.drawBox(frame, box)
    shape = predictor(frame, box)
    points = []
    
    for i in range(68):
        coords = []

        coords.append((shape.part(i).x))
        coords.append((shape.part(i).y))
        points.append(coords)
        
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

                points = findLandmarks(frame)
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

#VARIABLES
active = np.array([999])
active_points = np.zeros([68])
frame_store = []
try:
    streamPoses = np.load('projects/'+args.project_name+'_poses.npy')
except:
    print('No extraction poses found')
    streamPoses = []
lock = (0,0)
        
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
    genericFace = np.load('robasbase.npy')    
    
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
            points = findLandmarks(frame)

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
                mouth_coeffs, browcoeff, blinkcoeff, squintcoeff = decomp.findCoeffsAll(points, streamPoses)
                #print(mouth_coeffs)

    #### HANDLES INPUTS ####
    
    k = cv2.waitKey(10)
    
    if k==27: #ESC KEY - QUIT
        break
    
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
                 

    if k == 101: #E Key - ADD TO POSES FOR DECOMPOSITION            
        if model == True:
            if play == False:
                streamPoses.append(points)
                print('Key Pose Added')
        
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
    
cap.release()
cv2.destroyAllWindows()

# Save the poses
streamPoses = np.array(streamPoses)
if streamPoses.shape[0]!=0:
    np.save('projects/' + args.project_name + '_poses.npy', streamPoses)
