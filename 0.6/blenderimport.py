import bpy 
import numpy as np

#ENTER THE FILE PATH FOR MORPH TXT FILE HERE
data = np.loadtxt('C:/Users/Robert/Desktop/robshapes_morphs.txt')

#Replace 'Head_mesh' with the name of your object.
ob = bpy.data.objects['Head_Mesh']

#These are the names of the shapekeys that correspond (in this specific order) to the the morph targets exported out of strongtrack.
names = ['EyeBlink_L', 'EyeBlink_R', 'EyeSquint_L', 'EyeSquint_R', 'EyeDown_L', 'EyeDown_R', 'EyeIn_L', 
'EyeIn_R', 'EyeOpen_L', 'EyeOpen_R', 'EyeOut_L', 'EyeOut_R', 'EyeUp_L', 'EyeUp_R', 'BrowsD_L', 'BrowsD_R', 
'BrowsU_C', 'BrowsU_L', 'BrowsU_R', 'JawOpen', 'LipsTogether', 'JawLeft', 'JawRight', 'JawFwd',
'LipsUpperUp_L', 'LipsUpperUp_R', 'LipsLowerDown_L', 'LipsLowerDown_R', 'LipsUpperClose', 
'LipsLowerClose', 'MouthSmile_L', 'MouthSmile_R', 'MouthDimple_L','MouthDimple_R', 'LipsStretch_L', 
'LipsStretch_R', 'MouthFrown_L', 'MouthFrown_R', 'MouthPress_L', 'MouthPress_R', 'LipsPucker', 'LipsFunnel', 'MouthLeft','MouthRight',
'ChinLowerRaise','ChinUpperRaise', 'Sneer_L', 'Sneer_R', 'Puff', 'CheekSquint_L', 'CheekSquint_R']

# This script runs through the export file and looks for shapekeys with matching names. If it finds them it adds a keyframe.
for i in range(data.shape[0]):
    for j in range(len(names)):
        name = names[j]
        try:
            shape = ob.data.shape_keys.key_blocks[name]
            if data[i][j] != 0.0:
                shape.value=(data[i][j])
                shape.keyframe_insert("value", frame=i)
        except:
            pass
