import numpy as np
from sklearn.decomposition import SparseCoder

def setKeyPose(k, points, streamPoses):
    #print(streamPoses.shape)
    if k==49:
        print('neutral pose set')
        streamPoses[0] = points
    if k==50:
        print('jaw open set')
        streamPoses[1] = points
    if k==51:
        print('smile set')
        streamPoses[2] = points
    if k==52:
        print('funnel set')
        streamPoses[3] = points
    if k==53:
        print('pucker set')
        streamPoses[4] = points
    if k==54:
        print('flat set')
        streamPoses[5] = points
        
    return streamPoses

def checkKeyPoses(streamPoses):
    check = True
    for i in range(streamPoses.shape[0]):
        #print(sum(streamPoses[i][0]))
        if sum(streamPoses[i][0]) == 0.0:
            check = False
    return check

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

def findCoeffsAll(points, keyposes):
    
    #Find mouth coeffs
    mouth = np.array(points[48:68])
    mouth_centre = mouth.mean(0)

    width_points = points[16][0]-points[0][0]
   
    shiftedPosesMouth = shiftKeyPoses(width_points, mouth_centre, keyposes, 'mouth')
    mouth_coeffs = findCoeffsMouth(points,shiftedPosesMouth)
    
    #Find brows coeffs
    nosetop = points[27]
    shiftedPosesBrows = shiftKeyPoses(width_points, nosetop, keyposes, 'brows')
    
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

    return mouth_coeffs, browcoeff, blinkcoeff, squintcoeff*0.5

def shiftKeyPoses(new_width, centroid, keyposes, config):
   
    #Scale keypose based on head width to accomodate for translation or different video size.
    width_keypose = (keyposes[0][16][0]-keyposes[0][0][0])
    width_fac = width_keypose/new_width
    
    keyposes = np.divide(keyposes, [width_fac,width_fac]).astype(int)
    
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
