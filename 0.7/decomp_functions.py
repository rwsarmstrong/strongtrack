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

# Function using SKLearn to find mouth coeffs for a section of a pose
def findCoeffsSub(points, keyposes, config):
    
    target = np.array(points).astype(float)

    if config == 'mouth':

        target_sub = target[48:68]
        target_sub = np.reshape(target_sub, (1,40))

    if config == 'brows':
        target_sub = target[17:27]
        target_sub = np.reshape(target_sub, (1, 20))

    dict_2d = np.array(keyposes).astype(float)
    dict_2d_sub=[]

    if config == 'mouth':

        for i in range(dict_2d.shape[0]):
            dict_2d_sub.append(dict_2d[i][48:68])

        dict_2d_sub = np.array(dict_2d_sub)
        dict_2d_sub = np.reshape(dict_2d_sub, (dict_2d.shape[0], 40))

    if config == 'brows':
        for i in range(dict_2d.shape[0]):
            dict_2d_sub.append(dict_2d[i][17:27])

        dict_2d_sub = np.array(dict_2d_sub)
        dict_2d_sub = np.reshape(dict_2d_sub, (dict_2d.shape[0], 20))
 
    coder = SparseCoder(dictionary=dict_2d_sub, transform_n_nonzero_coefs=None,
                    transform_alpha=10, transform_algorithm='lasso_lars')

    coeffs = coder.transform(target_sub)

    return coeffs


def findCoeffsBrows(points, keyposes):

    target = np.array(points).astype(float)

    target_left = target[17:22]
    target_left = np.reshape(target_left, (1, 10))
    target_right = target[22:27]
    target_right = np.reshape(target_right, (1, 10))

    dict_2d = np.array(keyposes).astype(float)
    dict_2d_left = []
    dict_2d_right = []

    for i in range(dict_2d.shape[0]):
        dict_2d_left.append(dict_2d[i][17:22])

    for i in range(dict_2d.shape[0]):
        dict_2d_right.append(dict_2d[i][22:27])

    dict_2d_left = np.array(dict_2d_left)
    dict_2d_left = np.reshape(dict_2d_left, (dict_2d.shape[0], 10))
    dict_2d_right = np.array(dict_2d_right)
    dict_2d_right = np.reshape(dict_2d_right, (dict_2d.shape[0], 10))

    coder = SparseCoder(dictionary=dict_2d_left, transform_n_nonzero_coefs=None,
                        transform_alpha=0.5, transform_algorithm='lasso_lars')

    coeffsleft = coder.transform(target_left)

    coder = SparseCoder(dictionary=dict_2d_right, transform_n_nonzero_coefs=None,
                        transform_alpha=0.5, transform_algorithm='lasso_lars')

    coeffsright = coder.transform(target_right)

    return coeffsleft, coeffsright

def browCoeff(raw):
    if raw[0] < raw[3]:
        upcoeff = raw[1] / raw[3]
        downcoeff = 0
    else:
        downcoeff = raw[1] / raw[4]
        upcoeff = 0

    return upcoeff, downcoeff

# Filter for eyebrow down
def filter(val, lower, upper):
    factor = 1 / lower

    if lower > val:
        new_val = 0.0
    if lower <= val < upper:
        new_val = (val - lower) * factor
    if val >= upper:
        new_val = 1.0

    return new_val

# Brow coeffs for brows going up.
def simpleBrowUp(points, keyposes, config, config2):
    if config == 'left':
        first = 17
        last = 22

    if config == 'right':
        first = 22
        last = 27

    if config2 == "up":
        target = 1
    else:
        target = 2

    deltashifted = keyposes[target][first:last] - keyposes[0][first:last]
    deltashifted = (sum(sum(abs(deltashifted))))
    deltapoints = (points[first:last]) - (keyposes[target][first:last])
    deltapoints = (sum(sum(abs(deltapoints))))

    if deltapoints < (deltashifted):
        val = 1 - (deltapoints / deltashifted)

    else:
        val = 0.0

    return val

# Brow coeffs with detection of verticality for helping with brow down
def complexBrow(points, keyposes, config, config2):
    if config == 'left':
        first = 17
        last = 22

    if config == 'right':
        first = 22
        last = 27

    if config2 == "up":
        target = 1
    else:
        target = 2

    deltashifted = keyposes[target][first:last] - keyposes[0][first:last]
    deltashifted = (sum(sum(abs(deltashifted))))
    deltapoints = (points[first:last]) - (keyposes[target][first:last])
    deltapoints = (sum(sum(abs(deltapoints))))

    ydelt = keyposes[2][first:last] - points[first:last]

    ydelt = sum(ydelt.T[1])

    if deltapoints < (deltashifted):
        val = 1 - (deltapoints / deltashifted)

    else:
        val = 0.0

    if ydelt <= 0:
        val = 1.0

    return val

def findCoeffsBrowsSimple(points, keyposes):

    val_l_up = simpleBrowUp(points, keyposes, 'left', 'up')
    val_r_up = simpleBrowUp(points, keyposes, 'right', 'up')
    val_l_down = complexBrow(points, keyposes, 'left', 'down')
    val_r_down = complexBrow(points, keyposes, 'right', 'down')

    val_l_down = filter(val_l_down, 0.4, 0.8)
    val_r_down = filter(val_r_down, 0.4, 0.8)

    coeffsleft = (val_l_up, val_l_down)
    coeffsright = (val_r_up, val_r_down)

    return coeffsleft, coeffsright

def seperateKeyPoses(keydrops):

    keyposes_mouth = []

    for i in range(8):
        if keydrops[i][0][0] != 0.0:
            keyposes_mouth.append(keydrops[i])

    keyposes_brows = []
    keyposes_brows.append(keydrops[0])

    for i in range(8,10):
        if keydrops[i][0][0] != 0.0:
            keyposes_brows.append(keydrops[i])

    return keyposes_mouth, keyposes_brows

def findCoeffsAll(points, keyposes, keydrops):

    #Find mouth coeffs
    mouth = np.array(points[48:68])
    mouth_centre = mouth.mean(0)

    width_points = points[16][0]-points[0][0]

    #Temporary means for removing eye keyposes from mouth by removing last two (brows up and down)
    keyposes_mouth = keyposes

    if keydrops[8][0][0] != 0.0:
        'Brow Up set. Removing from keyposes for mouth'
        nobrowsindex = len(keyposes_mouth) - 1

        keyposes_mouth= keyposes_mouth[0:nobrowsindex]

    if keydrops[9][0][0] != 0.0:
        'Brow Down set. Removing from keyposes for mouth'
        nobrowsindex = len(keyposes_mouth) - 1

        keyposes_mouth= keyposes_mouth[0:nobrowsindex]

    keyposes_mouth, keyposes_brows = seperateKeyPoses(keydrops)

    shiftedPosesMouth = shiftKeyPoses(width_points, mouth_centre, keyposes_mouth, 'mouth')

    mouth_coeffs = findCoeffsSub(points,shiftedPosesMouth, 'mouth')
    
    #Find brows coeffs
    eyes = np.array(points[36:48])
    eyes_centre = eyes.mean(0)

    # Temporary means for removing mouth keyposes from eyes
    #keyposes_brows = np.array([keydrops[0],keydrops[8],keydrops[9]])

    shiftedPosesBrows = shiftKeyPoses(width_points, eyes_centre, keyposes_brows, 'brows')

    brow_coeffs_left, brow_coeffs_right = findCoeffsBrowsSimple(points, shiftedPosesBrows)

    brow_coeffs = np.array([brow_coeffs_left, brow_coeffs_right])

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

    return mouth_coeffs, brow_coeffs, blinkcoeff, squintcoeff*0.5

def shiftKeyPoses(new_width, centroid, keyposes, config):
   
    #Scale keypose based on head width to accomodate for translation or different video size.
    width_keypose = (keyposes[0][16][0]-keyposes[0][0][0])
    width_fac = width_keypose/new_width
    
    keyposes = np.divide(keyposes, [width_fac,width_fac]).astype(int)
    
    new_poses = []
    
    for i in range(keyposes.shape[0]):
        #For brows we take average of eyes points
        if config == 'brows':
            keypose = np.array(keyposes[i][36:48])
        #Fo mouth we take average of mouth points
        if config == 'mouth':
            keypose = np.array(keyposes[i][48:68])
            
        centroid_keypose = keypose.mean(0)
        delta = centroid_keypose-centroid
        
        new_pose = keyposes[i]-delta.astype(int)
        new_poses.append(new_pose)
        
    return np.array(new_poses)
