# StrongTrack v0.8

Download the windows 10, (64 bit only) version [here](https://drive.google.com/file/d/1RSHuZtHB_VTBN37-PuapUriQn49aI6jJ/view?usp=sharing) (google drive. 114 MB zip).

Example projects for UE4 and Blender may be found at the bottom of the page.

Hi there! Here you can find code for StrongTrack, a tool for landmark annotation and finding coefficents for facial animation. If installing python/libraries (see below) is intimidating and you're running a windows 10 (with a 64 bit installation...which it probably is) I recommend trying the exectuable linked above. This executable consists of the code contained within this reposititory passed through pyinstaller for packaging. I'm working on a full install process for windows/mac/linux to further lower the barrier to entry but this may take a while.

**This tool is still at an early stage of development by a non-professional. Although I have put it through its paces as thoroughly as I can, you use this software at your own risk (see license for more)**

![Screenshot](https://github.com/rwsarmstrong/strongtrack/blob/0.8/0.8/images/screenshot.jpg)

# 0.8 Release Notes
* Pretrained model that allows for the ability to 'guess' landmarks for the whole face or mouth seperately, rather than manual placement.
* Initial webcam support for recording and streaming animation as well as record and reopen video.
* Removal of non function eye tracking to avoid confusion (eye tracking to be added in 0.9 or 1.0 depending on development of pretrained model).
* Improved UI feedback for certain actions such as training or alerts when creating duplicate files.
* Better automatic enabling and disabling of buttons/options that would lead to crashes if pressed at times prior to intended time.
* Improved handling of video with different frame rates
* Option to open 'previous model' instead of having to manually selected each time when using a previously prepared model
* Minor bug fixes and UI clean up.
* New default training assets with texturing

# Overview
This a python based tool for finding coefficients for facial animation from RGB video. Coefficients can be exported as a numpy save file (for importing into Blender for example) but can also stream into Unreal Engine (or elsewhere) via OSC.

This solution is made up of two core components; Facial landmark tracking and a decomposition to produce coefficients. For landmark tracking this respository includes a method to train and refine a model based on your own footage. Once you are satisfied with the landmark tracking you have each subject pull a number of distinct poses (neutral, smile, jaw open etc) and store these as key poses that are then used as the basis for the decomposition into coefficients. Landmarks from someone half smiling with their mouth open would be decomposited into a result with a 0.5 smile, 0.5 jaw open for example. These coefficients are what is then exported out to a save file or streamed (or both).

# Requirements
* Python3
* Unreal Engine 4.25 or above (if using UE4)

As well as these python libraries
* OpenCV (for media and rendering)
* Dlib (facial landmark tracking and training)
* Sklearn (linear regression/decomposition for coefficients)
* Numpy
* PythonOSC 
* XML elementree 
* PyQT5 (for GUI)

# Installation and running
Assuming you have python and the libraries listed above installed correctly run python strongtrack.py
If you have using the executable nagivate to the 'executable' and run 'strongtrack.exe'.

You'll be presented with a GUI interface where you can pick a video to analyse as well as the option to create a new model or load up a previously created model. Only once a model has been created or loaded will the video and annotation tool appear.

The project name you enter/use is used to set aside different training data and keyposes for multiple faces. XML files, model files and keyposes are created in the 'projects' directory.

# Workflow pt 1 - Landmark placement.
Video the subject pulling a series of keyposes. Neutral, jaw fully open, closed smile, lips funnel, lip pucker, brow up, brow down, eye closed. These keyposes are useful for quickly training a landmark model. As of ver 0.5 this tool is stil very much built for mostly stationary faces so if possible a head mounted camera is strongly recommended, but footage with a mostly stationary subject will still work.

Upon opening this video with StrongTrack you'll be presented with the video alongside a generic unmatched set of facial landmarks. Scrolling the video and entering landmarks is only possible when the video is paused. Pause the video with the SPACE KEY at the neutral pose and place the landmarks at the corresponding place upon the face. It's important you start with a neutral frame because this will enable you to later use the N key whenever you want to return the lips to a neutral pose, which is a great time saver.

Left mouse click to move individual points and right click to move face groups (jaw, eye, nose etc). The W key welds the lips together as a time saver. Points in white move points around them with a drop off in influence. 

Once you're happy with the placement add this to the training set with the F KEY. For this initial entry the model will then train the predictor automatically.

Head to the next frame (jaw open preferably because it is the most different) and repeat. Carry on in this manner, hitting the T KEY whenever you want to train the model. As the model becomes more accurate, less and less manual placement should be necessary.

Use the W key to weld lip centre together or N key to return mouth points to neutral. These are useful timesavers.

Hitting ESC will quit the viewer.

# Workflow pt 2 - Morph targets/shapekey export.
Once you are happy with the accuracy of landmark placement you can assign keyposes for the decomposition algorithm to use to produce animation data. You should use the same footage of 'extreme' poses (neutral, jaw open, smile, funnel etc) as used in the landmark training for similar reasoning; the more different poses are, the easier it is to produce data from them. 

Using the dropdown list in the control panel, assign different frames as representing different key poses, making sure to hit 'set keypose' each time. Whenever you hit 'set keypose' a datafile called 'PROJECTNAME_keyposes.npy' will be created/modified.

Once this file has been created you can proceed with the remaining two buttons on the control panel; Export to Txt and Stream OSC. It's best to commit as many different poses to the file as possible, though not all are necessary. The more provided the more accurate animation can be generated.

# Workflow pt3 - Usage
To export animation ensure you're happy with the landmark tracking model and the keypose set. Open the video file you want to animate with and select the landmark model. The corresponding keypose set with be selected automatically based off the filenames. As part of this session you can continue to update and train the landmark training with the source footage to account for particular face shapes. Indeed, this will almost certainly be required if not using headmounted cameras.

When ready to export animation, either to file or OSC, it will take the form of 51 different values that combine to describe many different possible facial expressions. The names of the values can be found listed in the 'blenderimport' python script. Your model does not need to have all 51 shapes (all footage I've currently shared uses about 5-6), but the names do need to match. 

These values are written to the text file as plain text as an array in the shape of (number of frames, 51). With OSC the values are streamed once a frame to 127.0.0.1 as a float array of length 51.

# Workflow pt 4 - Refinement
Coming hopefully in 0.9. A process that draws on the animation analysis to make better targeted and full use of all available blend shapes/morph targets on a model.

# Example project files
An Unreal Engine project (4.25 and above) can be found [here](https://drive.google.com/file/d/1jOlB9IA068MmkdfMyCxCW0TFL3oD1AFk/view?usp=sharing) (google drive. 4.6 MB zip). A Blender example project (created with 2.83) can be found [here](https://drive.google.com/file/d/1esG5yJNPG0h7Tzv66Qd5h-R35Je0IWnT/view?usp=sharing) (google drive. 5.3 MB zip)
