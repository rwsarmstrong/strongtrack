import numpy as np
import cv2


def getMask(img, points, eye):
    points = np.array(points)
    
    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()
    # Change thresholds
    params.minThreshold = 10
    params.maxThreshold = 200
    # Filter by Area.
    params.filterByArea = True
    params.minArea = 1000
    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.1

    detector = cv2.SimpleBlobDetector_create(params)
    
    found = True
    
    if eye == 'right':
        eye_points = points[42:48]
    else:
        eye_points = points[36:42]

    eye_points = np.array([eye_points])
    #img = cv2.bitwise_not(img)
    #eye_points = np.subtract(eye_points, [0,-12])        
    black = np.zeros((1200, 1080),dtype=np.uint8)
    #eye_points = np.subtract(eye_points, [eye_left,eye_top])
    mask = cv2.fillPoly(black, eye_points, 255)
    #mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=2)
    img = cv2.bitwise_and(img,img,mask = mask)
    
    return img

def getPupil(img, points, eye, threshold):
    points = np.array(points)
    
    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()
    # Change thresholds
    params.minThreshold = 0
    params.maxThreshold = 200
    # Filter by Area.
    params.filterByArea = True
    params.minArea = 100
    params.maxArea = 300
    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0
    detector = cv2.SimpleBlobDetector_create(params)
    
    found = True
    
    if eye == 'right':
        eye_points = points[42:48]
    else:
        eye_points = points[36:42]

    eye_points = np.array([eye_points])
    
    #eye_points = np.subtract(eye_points, [0,5])

    padding = 50
    
    eye_top = eye_points[0][2][1]-padding
    eye_bottom = eye_points[0][5][1]+padding
    eye_left = eye_points[0][0][0]-padding
    eye_right = eye_points[0][3][0]+padding
    
    #img = cv2.bitwise_not(img)
    
    img = img[eye_top:eye_bottom,eye_left:eye_right]
    #Mask out the area
    black = np.zeros((eye_bottom-eye_top,eye_right-eye_left),dtype=np.uint8)
    eye_points = np.subtract(eye_points, [eye_left,eye_top])
    mask = cv2.fillPoly(black, eye_points, 255)
    mask = cv2.dilate(mask, None, iterations=2)
    img = cv2.bitwise_and(img,img,mask = mask)

    #Filter for blob detection
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
    img = cv2.erode(img, None, iterations=2)
            
    img = cv2.medianBlur(img, 5)

    #Threshold 0-40
    #threshold = 20
    ret,img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    keypoints = detector.detect(img)
    imkeypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    try:
        x = int(keypoints[0].pt[0])
        y = int(keypoints[0].pt[1])
    except:
        #print('No keypoint')
        x = int(500)
        y = int(500)
        found = False
    
    centre = (x+eye_left, y+eye_top)
    
    amount = findEyeVals(centre, points, eye)
        
    return imkeypoints, centre, amount[0], found

def getIris(img, points, eye, threshold):
    points = np.array(points)
    
    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()
    # Change thresholds
    params.minThreshold = 10
    params.maxThreshold = 200
    # Filter by Area.
    params.filterByArea = True
    params.minArea = 1000
    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.1

    detector = cv2.SimpleBlobDetector_create(params)
    
    found = True
    
    if eye == 'right':
        eye_points = points[42:48]
    else:
        eye_points = points[36:42]

    eye_points = np.array([eye_points])
    
    eye_points = np.subtract(eye_points, [0,-12])

    padding = 50
    
    eye_top = eye_points[0][2][1]-padding
    eye_bottom = eye_points[0][5][1]+padding
    eye_left = eye_points[0][0][0]-padding
    eye_right = eye_points[0][3][0]+padding
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.bitwise_not(img)
    
    img = img[eye_top:eye_bottom,eye_left:eye_right]
    
    black = np.zeros((eye_bottom-eye_top,eye_right-eye_left),dtype=np.uint8)
    eye_points = np.subtract(eye_points, [eye_left,eye_top])
    mask = cv2.fillPoly(black, eye_points, 255)
    img = cv2.bitwise_and(img,img,mask = mask)
    
    #threshold = 140
    ret, img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    img = cv2.erode(img, None, iterations=1)
    img = cv2.dilate(img, None, iterations=5)
    img = cv2.medianBlur(img, 1)
    
    img = cv2.bitwise_not(img)
    keypoints = detector.detect(img)
    imkeypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    try:
        x = int(keypoints[0].pt[0])
        y = int(keypoints[0].pt[1])
    except:
        #print('No keypoint')
        x = int(500)
        y = int(500)
        found = False
        
    centre = (x+eye_left, y+eye_top-8)
    
    amount = 0
    return imkeypoints, centre, amount, found

def findEyeVals(eye_point, points, config):
    
    if config == 'left':
        eye_left = points[36]
        eye_right = points[39]
        eye_top = np.array(points[37:39]).mean(0)
        eye_bottom = np.array(points[40:42]).mean(0)
    if config == 'right':
        eye_left = points[42]
        eye_right = points[45]
        eye_top = np.array(points[43:45]).mean(0)
        eye_bottom = np.array(points[46:48]).mean(0)
    
    distleft = np.subtract(eye_left, eye_point)
    distright = np.subtract(eye_right, eye_point)
    distleftright = abs(distright[0])+abs(distleft[0])
    
    disttop = np.subtract(eye_top, eye_point)
    distbottom = np.subtract(eye_bottom, eye_point)
    disttopbottom = (abs(disttop[1])+abs(distbottom[1]))
    
    amountleftright = abs(distleft[0])/distleftright
    amounttopbottom = abs(disttop[1])/disttopbottom     
    amounts = (amountleftright, amounttopbottom)

    offsetpoints = findEyePointsFromAmount(amounts, points, config)
    
    return amounts, offsetpoints

def findEyePointsFromAmount(amount, points, config):

    if config == 'left':
        eye_left = points[36]
        eye_right = points[39]
        eye_top = np.array(points[37:39]).mean(0)
        eye_bottom = np.array(points[40:42]).mean(0)
        
    if config == 'right':
        eye_left = points[42]
        eye_right = points[45]
        eye_top = np.array(points[43:45]).mean(0)
        eye_bottom = np.array(points[46:48]).mean(0)

    eyeline_leftright = np.subtract(eye_right, eye_left)
    eyeline_topbottom = np.subtract(eye_bottom, eye_top)

    offset = ((eyeline_leftright[0]*amount[0]),(eyeline_topbottom[1]*amount[1]))

    offset = [(eye_left[0]+offset[0]), (eye_top[1]+offset[1])]
    
    return offset

def findEyes(eyepoints, points):
    
    eye_l = np.array(eyepoints[36:42]).mean(0)
    eye_r = np.array(eyepoints[42:48]).mean(0)

    eye_lamount, eye_loffset = findEyeVals(eye_l, points, 'left')
    eye_ramount, eye_roffset = findEyeVals(eye_r, points, 'right')

    amount = np.add(eye_lamount,eye_ramount)/2    
    return amount
