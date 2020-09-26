import bpy 
import numpy as np

# OPEN THIS FILE IN BLENDER'S TEXT EDITOR

#ENTER THE FILE PATH FOR TXT FILE EXPORTED FROM STRONGTRACK HERE. THEN HIT THE PLAY BUTTON ABOVE THIS TEXT TO RUN THE SCRIPT
data = np.loadtxt('C:/Users/Robert/Desktop/anim_export.txt')

#Replace 'Head_mesh' with the name of your object.
ob = bpy.data.objects['FaceMesh']

#These are the names of the shapekeys that correspond (in this specific order) to the the morph targets exported out of strongtrack.
names = ['eyeBlinkLeft', 'eyeBlinkRight', 'eyeSquintLeft', 'eyeSquintRight', 'eyeLookDownLeft', 'eyeLookDownRight', 'eyeLookInLeft', 
'eyeLookInRight', 'eyeWideLeft', 'eyeWideRight', 'eyeLookOutLeft', 'eyeLookOutRight', 'eyeLookUpLeft', 'eyeLookUpRight', 'browDownLeft', 'browDownRight', 
'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight', 'jawOpen', 'mouthClose', 'jawLeft', 'jawRight', 'jawFwd',
'mouthUpperUpLeft', 'mouthUpperUpRight', 'mouthLowerDownLeft', 'mouthLowerDownRight', 'mouthRollUpper', 
'mouthRollLower', 'mouthSmileLeft', 'mouthSmileRight', 'mouthDimpleLeft','mouthDimpleRight', 'mouthStretchLeft', 
'mouthStretchRight', 'mouthFrownLeft', 'mouthFrownRight', 'mouthPressLeft', 'mouthPressRight', 'mouthPucker', 'mouthFunnel', 'mouthLeft','mouthRight',
'mouthShrugLower','mouthShrugUpper', 'noseSneerLeft', 'noseSneerRight', 'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight']

# This script runs through the export file and looks for shapekeys with matching names. If it finds them it adds a keyframe.
for i in range(data.shape[0]):
    for j in range(len(names)):
        name = names[j]
        try:
            shape = ob.data.shape_keys.key_blocks[name]
            shape.value=(data[i][j])
            shape.keyframe_insert("value", frame=i)
        except:
            pass
