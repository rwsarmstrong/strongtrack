import cv2
import numpy as np
import math
from sklearn.decomposition import SparseCoder
# Function using SKLearn to find mouth coeffs
def findCoeffsMouth(points, keyposes):
    
    target = np.array(points).astype(float)
    target_mouth = target[48:68]
    target_mouth = np.reshape(target_mouth, (1,40))
    
    dict_2d = np.array(keyposes).astype(float)
    dict_2d_mouth=[]
    
    for i in range(dict_2d.shape[0]):
        dict_2d_mouth.append(dict_2d[i][48:68])

    dict_2d_mouth = np.array(dict_2d_mouth)
    dict_2d_mouth = np.reshape(dict_2d_mouth, (dict_2d.shape[0], 40))
 
    coder = SparseCoder(dictionary=dict_2d_mouth, transform_n_nonzero_coefs=None,
                    transform_alpha=10, transform_algorithm='lasso_lars')

    coeffs = coder.transform(target_mouth)

    return coeffs

def findCoeffsAll(frame, points, keyposes):
    
    #Find mouth coeffs
    mouth = np.array(points[48:68])
    mouth_centre = mouth.mean(0)

    shiftedPosesMouth = shiftKeyPoses(mouth_centre, keyposes, 'mouth')
    mouth_coeffs = findCoeffsMouth(points,shiftedPosesMouth)
    
    #Find brows coeffs
    nosetop = points[27]
    shiftedPosesBrows = shiftKeyPoses(nosetop, keyposes, 'brows')
    
    browsposes = np.array(shiftedPosesBrows[0][17:27])
    browsposescentre = browsposes.mean(0)
    brows = np.array(points[17:27])
    browscentre = brows.mean(0)
    browcoeff = (browsposescentre[1]-browscentre[1])/67
    
    #Find blink coeffs
    eye_top = np.array(points[37:39])
    eye_bottom = np.array(points[40:42])
    eye_mid = np.array([points[36],points[39]])
    eye_mid = eye_mid.mean(0)
    eye_top = eye_top.mean(0)
    eye_bottom = eye_bottom.mean(0)
    blinkcoeff = ((eye_top[1]-eye_mid[1])+28)/48

    #Find squint coeffs (-9.5 = 1 | -17 = 0)
    squintcoeff = ((eye_mid[1]-eye_bottom[1])+17)/7.5
    #print(squintcoeff)
    #cv2.circle(frame,(int(eye_bottom[0]),int(eye_bottom[1])),5,(0,0,255),-1)
    return frame, mouth_coeffs, browcoeff, blinkcoeff, squintcoeff*0.5

def shiftKeyPoses(centroid, keyposes, config):
   
    keyposes = np.array(keyposes)
    new_poses = []

    for i in range(keyposes.shape[0]):
        if config == 'brows':
            keypose = np.array(keyposes[i][36:48])
        if config == 'mouth':
            keypose = np.array(keyposes[i][48:68])
            
        centroid_keypose = keypose.mean(0)
        delta = centroid_keypose-centroid
        
        new_pose = keyposes[i]-delta.astype(int)
        new_poses.append(new_pose)
        
    return np.array(new_poses)

def getFaceRT(points, model_points, points_store, frame):
        size = frame.shape

        focal_length = size[1]
        center = (size[1]/2, size[0]/2)
        camera_matrix = np.array(
                             [[focal_length, 0, center[0]],
                             [0, focal_length, center[1]],
                             [0, 0, 1]], dtype = "double"
                             )
        dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
        
        points_store = np.concatenate([points, points_store[0:-1]])
        points_store_mean = points_store.mean(0)
                
        pointseyes = points_store_mean[17:48]
           
        pointseyes = np.reshape(pointseyes, (1,31,2))
            
        modeleyes = model_points[17:48]
            
        success, rotation_vector, translation_vector = cv2.solvePnP(modeleyes, pointseyes, camera_matrix, dist_coeffs)
        #success, rotation_vector, translation_vector = cv2.solvePnP(model_points, points_store_mean, camera_matrix, dist_coeffs)

        
        point2D, jacobian = cv2.projectPoints(np.array([model_points]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        point2D = point2D.reshape((68,2))
            # print(point2D[0])
            
        for i in range(point2D.shape[0]):
            point = point2D[i]
            if i < 8:
                cv2.circle(frame, (int(point[0]), int(point[1])), 3, (0,255,0), -1)
            else:
                cv2.circle(frame, (int(point[0]), int(point[1])), 3, (0,0,255), -1)

        rotmatrix = cv2.Rodrigues(rotation_vector)
        euler = rotationMatrixToEulerAngles(rotmatrix[0])

        return euler, translation_vector, points_store


# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).

def rotationMatrixToEulerAngles(R) :

    assert(isRotationMatrix(R))
    
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    
    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])
  
def drawBox(frame, box):
    first = [box.left(),box.top()]
    second =[box.right(),box.top()]
    third = [box.left(),box.bottom()]
    fourth = [box.right(),box.bottom()]
    frame = cv2.line(frame, tuple(first), tuple(second),(0,0,255), 1)
    frame = cv2.line(frame, tuple(second), tuple(fourth),(0,0,255), 1)
    frame = cv2.line(frame, tuple(fourth), tuple(third),(0,0,255), 1)
    frame = cv2.line(frame, tuple(third), tuple(first),(0,0,255), 1)
    return frame

def noseOrientation(points):
    leftright=(points[33][0]-points[30][0])
    #print(leftright)
    #eyecentr =(points
    updown=(points[30][1]-points[39][1])
    sideside=(points[36][1]-points[45][1])

    return leftright

#Removes the unwanted points around the jaw. 
def slicearray(points):
    chin_orig = points[7:10]
    rest_orig = points[17:68]

    points = np.concatenate((chin_orig, rest_orig))
    
    return points

def drawFace(img, points, config):
    
    chin = points[7:9]
    brow_left = points[17:21]
    brow_right = points[22:26]
    nose_length = points[27:30]
    nose_under = points[31:35]
    eye_left = points[36:42]
    eye_right = points[42:48]
    mouth_in = points[60:68]
    mouth_out = points[48:60]

    
    
    if config=='debug':
        img = drawLines(img, points, points[len(points)-1])

    if config=='mouth':
        img = drawLines(img, mouth_in, points[60])
        img = drawLines(img, mouth_out, points[48])
            
    if config=='full':
        img = drawLines(img, nose_length, points[30])
        img = drawLines(img, nose_under,  points[35])
        img = drawLines(img, eye_left, points[36])
        img = drawLines(img, eye_right, points[42])
        img = drawLines(img, mouth_in, points[60])
        img = drawLines(img, mouth_out, points[48])
        img = drawLines(img, brow_right,  points[26])
        img = drawLines(img, brow_left, points[21])
        img = drawLines(img, chin, points[9])
        
    if config=='nose_eyes':
        img = drawLines(img, nose_length, points[30])
        img = drawLines(img, nose_under,  points[35])
        img = drawLines(img, eye_left, points[36])
        img = drawLines(img, eye_right, points[42])

    if config=='nose_eyes_chin':
        img = drawLines(img, nose_length, points[30])
        img = drawLines(img, nose_under, points[35])
        img = drawLines(img, eye_left, points[36])
        img = drawLines(img, eye_right, points[42])
        img = drawLines(img, chin, points[9])

    if config=='eye_rot':
        img = drawLinesGreen(img, eye_left, points[36])
        img = drawLinesGreen(img, eye_right, points[42])

    return img

def drawLines(img, points, loop):

    
    for i in range(len(points)-1):
        first = points[i]
        second = points[i+1]
            
        img = cv2.line(img, tuple(first), tuple(second),(0,0,255), 1)

    img = cv2.line(img, tuple(second), tuple(loop),(0,0,255), 1)

    return img

def drawLinesGreen(img, points, loop):
 
    for i in range(len(points)-1):
        first = points[i]
        second = points[i+1]
            
        img = cv2.line(img, tuple(first), tuple(second),(0,255,0), 1)

    img = cv2.line(img, tuple(second), tuple(loop),(0,255,0), 1)

    return img


