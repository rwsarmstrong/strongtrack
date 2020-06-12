# StrongTrack v0.6
Hi there! Here you can find code for StrongTrack, a tool for landmark annotation and finding coefficents for facial animation. If installing python/libraries (see below) is intimidating and you're running windows I recommend trying the exectuable  which can be found [here](https://drive.google.com/file/d/1q-SJoISpylqYbaZILuCMIJWKkvI9_S6z/view?usp=sharing) (google drive. 90.6 MB zip). This executable consists of the code contained within this reposititory passed through pyinstaller for packaging. I'm now working on a full install process for windows/mac/linux to further lower the barrier to entry.

**This tool is still at an early stage of development by a non-professional. Although I have put it through its paces as thoroughly as I can, you use this software at your own risk (see license for more). v0.6 represents an early release of just the landmark training component although limited export of coefficients is possible with the instructions below. The resulting training files may continue to be usable in later versions but I cannot guarantee this. Apologies**

# 0.6 Release Notes
* Added additional GUI for assigning keyposes and for streaming/recording morphs (limited to mouth shapes only still).
* Removed initial gesture at eye/eyebrow tracking due to quality being too low for public use.

# Overview
This a python based tool for finding coefficients for facial animation from RGB video. Coefficients can be exported as a numpy save file (for importing into Blender for example) but can also stream into Unreal Engine (or elsewhere) via OSC.

![Screenshot](/0.6/projects/images/screenshot.jpg)

This solution is made up of two core components; Facial landmark tracking and a decomposition to produce coefficients. For landmark tracking this respository includes a method to train and refine a model based on your own footage. Once you are satisfied with the landmark tracking you have each subject pull a number of distinct poses (neutral, smile, jaw open etc) and store these as key poses that are then used as the basis for the decomposition into coefficients. Landmarks from someone half smiling with their mouth open would be decomposited into a result with a 0.5 smile, 0.5 jaw open for example. These coefficients are what is then exported out to a save file or streamed (or both).

# Requirements
* Python3
* Unreal Engine 4.25 or above (if using UE4)

As well as these python libraries
* OpenCV (for media and GUI)
* Dlib (facial landmark tracking and training)
* Sklearn (linear regression/decomposition for coefficients)
* Numpy
* PythonOSC 
* XML elementree 
* PyQT5 (for additional GUI)

# Installation and running
Assuming you have python and the libraries listed above installed correctly run python strongtrack.py
If you have using the executable nagivate to the 'executable' and run 'strongtrack.exe'.

Now in 0.6 you'll be presented with a GUI interface where you can pick a video to analyse as well as the option to create a new model or load up a previously created model. Only once a model has been created or loaded will the video and annotation tool appear.

The project name you enter/use is used to set aside different training data and keyposes for multiple faces. XML files, model files and keyposes are created in the 'projects' directory.

# Example - Landmark placement.
Video the subject pulling a series of keyposes. Neutral, jaw fully open, closed smile, lips funnel, lip pucker, brow up, brow down, eye closed. These keyposes are useful for quickly training a landmark model. As of ver 0.5 this tool is stil very much built for mostly stationary faces so if possible a head mounted camera is strongly recommended, but footage with a mostly stationary subject will still work.

Upon opening this video with StrongTrack you'll be presented with the video alongside a generic unmatched set of facial landmarks. Scrolling the video and entering landmarks is only possible when the video is paused. Pause the video with the SPACE KEY at the neutral pose and place the landmarks at the corresponding place upon the face. It's important you start with a neutral frame because this will enable you to later use the N key whenever you want to return the lips to a neutral pose, which is a great time saver.

Left mouse click to move individual points and right click to move face groups (jaw, eye, nose etc). The W key welds the lips together as a time saver. Points in white move points around them with a drop off in influence. 

Once you're happy with the placement add this to the training set with the F KEY. For this initial entry the model will then train the predictor automatically.

Head to the next frame (jaw open preferably because it is the most different) and repeat. Carry on in this manner, hitting the T KEY whenever you want to train the model. As the model becomes more accurate, less and less manual placement should be necessary.

Use the W key to weld lip centre together or N key to return mouth points to neutral. These are useful timesavers.

Hitting ESC will quit the viewer.

# Example - Morph targets/shapekey export.
Once you are happy with the accuracy of landmark placement you can assign keyposes for the decomposition algorithm to use to produce animation data. You should use the same footage of 'extreme' poses (neutral, jaw open, smile, funnel etc) as used in the landmark training for similar reasoning; the more different poses are, the easier it is to produce data from them. 

Using the dropdown list in the control panel, assign different frames as representing different key poses, making sure to hit 'set keypose' each time. Whenever you hit 'set keypose' a datafile called 'PROJECTNAME_keyposes.npy' will be created/modified.

Once this file has been created you can proceed with the remaining two buttons on the control panel; Export to Txt and Stream OSC. It's best to commit as many different poses to the file as possible, though not all are necessary (as of 0.6 the only keyposes provided are mouth shapes). The more provided the more accurate animation can be generated.

When exporting animation, either to file or OSC, it will take the form of 51 different values that combine to describe many different possible facial expressions.
