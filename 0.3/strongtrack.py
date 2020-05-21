import cv2
import numpy as np
import dlib
import time
import face_functions as face
import rob_xml as rx
import decomp
import os
import movers

def getEasyBox(frame):
    padding = int(frame.shape[1]/10.8)
    box = dlib.rectangle(padding,padding,frame.shape[1]-padding, frame.shape[0]-padding)
    return box

def findLandmarks(frame):
    
    box = getEasyBox(frame)  
    frame = face.drawBox(frame, box)
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
    
    if play == False: 
        cap.set(cv2.CAP_PROP_POS_FRAMES,val)
        ret, frame = cap.read()
        frame_num=val

        if frame_num == length:
            print('no frame')
        else:
            cv2.imshow('Video', frame)

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
                    movers.activatePortion(i, active_points)
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
        pointsmove = movers.movePoints(points, delta, active[0])
    else:
        pointsmove = movers.movePointsMultiple(points, active_points, delta)
    
    updateFramePoints(pointsmove)
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if (sum(active_points)) == 0.0:
            active[0] = 999
            points = pointsmove
            updateFramePoints(points) 
            cv2.setMouseCallback('Video',pick_circle)
        
    if event == cv2.EVENT_RBUTTONDOWN:
        if (sum(active_points)) > 0:
            active[0] = 999
            active_points = np.zeros([68])
            points = pointsmove
            updateFramePoints(points)
            cv2.setMouseCallback('Video',pick_circle)

    pass

#Handles update of frame for mouse events
def updateFramePoints(points):
    global frame_store
    
    frame = np.array(frame_store)
    frame = face.drawFace(frame, points, 'full')
    points = np.array(points)
    
    for i in range(points.shape[0]):
        if i == active:
            cv2.circle(frame,(points[i][0],points[i][1]),5,(0,255,0),-1)
        else:
            cv2.circle(frame,(points[i][0],points[i][1]),5,(0,0,255),-1)
        
    cv2.imshow("Video", frame)

    pass

#TRAINING
xml_path = 'xmlbase.xml'
options = dlib.shape_predictor_training_options()
options.oversampling_amount = 300
options.nu = 0.05
options.tree_depth = 2
options.be_verbose = True

#VARIABLES
active = np.array([999])
active_points = np.zeros([68])
frame_store = []
#prev_points = []
streamPoses = []
lock = (0,0)
        
# GET VIDEO
video_path = 'C:/Users/Robert/Desktop/Scripts/robvid.mp4'
cap = cv2.VideoCapture(video_path)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#SET UP DISPLAY

# PLAYBACK
cv2.namedWindow('Video')
cv2.createTrackbar('Frame', 'Video' , 0, length, on_trackbar)
cv2.setMouseCallback('Video',pick_circle)

#Track frame number
frame_num = 0

#Initialise start
play = True
model = True
#track = True
#mouseDrag = False
stream = False

#Check for model
check = rx.convertXMLPoints(xml_path)
if check.shape[0] == 0:
    print('no face values found in xml. Setting generic face')
    model = False
    genericFace = np.load('genericface.npy')
else:
    print('model with values found')
    predictor = dlib.shape_predictor('predictor.dat')


#### MAIN LOOP ####
    
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

            frame = face.drawFace(frame, points, 'full')
            
            prev_points = points
            
        #If model wasn't found set points to generic face for manual placement
        if model ==False:
            
            points = genericFace[0]

            frame = face.drawFace(frame, points, 'full')
            
        
        #Display the video on screen    
        cv2.imshow('Video', frame)

        #Stream the video via OSC
        if model== True:
            if stream == True:
                mouth_coeffs, browcoeff, blinkcoeff, squintcoeff = decomp.findCoeffsAll(points, streamPoses)
                print(mouth_coeffs)

    #### HANDLES INPUTS ####
    
    k = cv2.waitKey(10)
    
    if k==27: #ESC KEY - QUIT
        break
    
    if k==116: #T KEY = TRAIN MODEL
        print('training model')
        dlib.train_shape_predictor(xml_path, "predictor.dat", options)
        predictor = dlib.shape_predictor('predictor.dat')
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
                neutral = rx.convertXMLPoints(xml_path)[0]
               
                for i in range(48,68):
                    points[i] = neutral[i]
                updateFramePoints(points)
                
    if k==119: # W KEY - Weld lips
        
        if play == False:
            print('welding')
            points = movers.weldLips(points)
            updateFramePoints(points)
                 

    if k == 101: #E Key - ADD TO POSES FOR DECOMPOSITION            
        if model == True:
            if play == False:
                streamPoses.append(points)
                print('stream Pose Added')
                
    if k == 100: #D - SWITCH BETWEEN MODES
        np.save('pointsfortest.npy', points)
        print('saved')
        
    if k==102: # F KEY - ADD TO XML
        if play ==False:
            
            box = getEasyBox(frame)
            filename = os.path.splitext(os.path.split(video_path)[1])[0]
            filepath = 'images/{}_frame{}.jpg'.format(filename,frame_num)
            cv2.imwrite(filepath, frame_store)
            rx.appendXML(points, box, filepath, xml_path)
            print('added')

            #If this is the first entry, train the predictor using the first entry.
            if model ==False:
                print('Training initial model')
                dlib.train_shape_predictor(xml_path, "predictor.dat", options)
                predictor = dlib.shape_predictor('predictor.dat')
                print('done')
                model = True
                
        else:
            print('can only add when paused!')
    
cap.release()
cv2.destroyAllWindows()

streamPoses = np.array(streamPoses)
np.save('streamposes.npy', streamPoses)
