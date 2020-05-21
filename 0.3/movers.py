import numpy as np

#Takes a point and makes its group active
def activatePortion(point, active_points):
    active_points[0:68]= 0
    
    mouth = [*range(48,68)]
    brow_L = [*range(17,22)]
    brow_R = [*range(22,27)]
    nose = [*range(27,36)]
    eye_L = [*range(36,42)]
    eye_R = [*range(42,48)]
    jaw = [*range(0,17)]
    
    if point in brow_L:
        active_points[17:22] = 1
    if point in brow_R:
        active_points[22:27] = 1
    if point in nose:
        active_points[27:36] = 1
    if point in eye_L:
        active_points[36:42] = 1
    if point in eye_R:
        active_points[42:48] = 1
    if point in mouth:
        active_points[48:68] = 1
    if point in jaw:
        active_points[0:17] = 1
        
    return active_points

def weldLips(points):
    
    points2 = np.array(points)
    centre1 = np.add(points2[61],points2[67])/2
    centre2 = np.add(points2[62],points2[66])/2
    centre3 = np.add(points2[63],points2[65])/2
    points[61] = points[67] = centre1.astype(int)
    points[62] = points[66] = centre2.astype(int)
    points[63] = points[65] = centre3.astype(int)
    return points

def movePoints(points, delta, i):
    
    if i == 0:
        pointsmove =moveJawLeft(points, delta)
        
    elif i == 8:
        pointsmove = moveChin(points, delta)
        
    elif i == 16:
        pointsmove = moveJawRight(points, delta)

    elif i == 51:
        pointsmove = moveMouthTop(points,delta)
        
    elif i == 57:
        pointsmove = moveMouthBottom(points,delta)
        
    elif i == 48:
        pointsmove = moveMouthLeft(points,delta)
        
    elif i == 54:
        pointsmove = moveMouthRight(points,delta)
        
    else:
        pointsmove = moveSingle(points, delta, i)

    return pointsmove

def movePointsMultiple(points, active_points, delta):

    points = np.array(points)
    points2 = np.array(points)

    for i in range(68):
        if active_points[i] == 1.0:
            points2[i] = (points[i][0]-delta[0],points[i][1]-delta[1])
    
    pointsmove = points2

    return pointsmove

#Top
def moveMouthTop(points, delta):
    points= np.array(points)
    points2 = np.array(points)
   
    bot = points[57]
    top = points[51]
    dist = abs(bot[1]-top[1])
    for i in range(48,68):
        degreeToMove = (bot[1]-points[i][1])/dist          
        points2[i] = (points[i][0]-(delta[0]*degreeToMove), points[i][1]-(delta[1]*degreeToMove))
    
    return points2

#Bottom
def moveMouthBottom(points, delta):
    points = np.array(points)
    points2 = np.array(points)
    bot = points[57]
    top = points[50]
    dist = abs(bot[1]-top[1])
        
    for i in range(48,68):
        degreeToMove = abs((top[1]-points[i][1])/dist)
        
        points2[i] = (points[i][0]-(delta[0]*degreeToMove), points[i][1]-(delta[1]*degreeToMove))
    
    return points2

#Bottom
def moveSingle(points, delta, i):
    points = np.array(points)
    points2 = np.array(points)
           
    points2[i] = (points[i][0]-delta[0], points[i][1]-delta[1])
    
    return points2

#LeftSide
def moveMouthLeft(points, delta):
    points2 = np.array(points)
    side = points[54]
            
    for i in range(48,68):
        degreeToMove = ((side[0]-points[i][0])/200)
        points2[i] = (points[i][0]-(delta[0]*degreeToMove), points[i][1]-(delta[1]*degreeToMove*degreeToMove*degreeToMove))
    
    return points2

#RightSide
def moveMouthRight(points, delta):
    points2 = np.array(points)
    side = points[49]
            
    for i in range(48,68):
        degreeToMove = ((side[0]-points[i][0])/200)
        points2[i] = (points[i][0]+(delta[0]*degreeToMove), points[i][1]+(delta[1]*degreeToMove*degreeToMove*degreeToMove))
    
    return points2

#Jaw
def moveChin(points, delta):
    points = np.array(points)
    points2 = np.array(points)
    chin = points[8]
    jaw_l = points[0]
    jaw_r = points[16]
    
    for i in range(0,9):
        dist_chin = sum(abs(points[i]-chin))
        dist_jaw = sum(abs(points[i]-jaw_l))
        degreeToMove = 1-(dist_chin/(dist_chin+dist_jaw))
        points2[i] = (points[i][0]-(delta[0]*degreeToMove), points[i][1]-(delta[1]*degreeToMove))
        
    for i in range(9,17):
        dist_chin = sum(abs(points[i]-chin))
        dist_jaw = sum(abs(points[i]-jaw_r))
        degreeToMove = 1-(dist_chin/(dist_chin+dist_jaw))
        
        points2[i] = (points[i][0]-(delta[0]*degreeToMove), points[i][1]-(delta[1]*degreeToMove))
        
    return points2

#Jaw
def moveJawLeft(points, delta):
    points= np.array(points)
    points2 = np.array(points)
    chin = points[8]
    jaw_l = points[0]
        
    for i in range(0,9):
        dist_chin = sum(abs(points[i]-chin))
        dist_jaw = sum(abs(points[i]-jaw_l))
        degreeToMove = 1-(dist_jaw/(dist_chin+dist_jaw))
        points2[i] = (points[i][0]-(delta[0]*degreeToMove), points[i][1]-(delta[1]*degreeToMove))
                
    return points2

#Jaw
def moveJawRight(points, delta):
    points = np.array(points)
    points2 = np.array(points)
    chin = points[8]
    jaw_r = points[16]
        
    for i in range(9,17):
        dist_chin = sum(abs(points[i]-chin))
        dist_jaw = sum(abs(points[i]-jaw_r))
        degreeToMove = 1-(dist_jaw/(dist_chin+dist_jaw))
        points2[i] = (points[i][0]-(delta[0]*degreeToMove), points[i][1]-(delta[1]*degreeToMove))
                
    return points2
