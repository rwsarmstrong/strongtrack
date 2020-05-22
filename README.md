# StrongTrack v0.3
Hi there! Here you can find code for StrongTrack, a tool for landmark annotation and finding coefficents for facial animation. If installing python/libraries (see below) is intimidating and you're running windows **I recommend waiting for an exectuable that I aim to push to this repository soon.**

**This tool is still at an early stage of development by a non-professional. v0.3 represents an early release of just the landmark training component. The resulting training files may continue to be usable in later versions but I cannot guarentee this. Apologies **

# Overview
This a python based tool for finding coefficients for facial animation from RGB video. Coefficients can be exported as a numpy save file (for importing into Blender for example) but can also stream into Unreal Engine (or elsewhere) via OSC.

![Screenshot](/0.3/projects/images/screenshot.jpg)

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

# Installation
Assuming you have python and the libraries listed above installed correctly run python strongtrack.py --video PATHTOYOURVIDEO.MP4 --project_name PROJECTNAME  and you should be presented with the interface as shown above.

# Example Usage
Video the subject pulling a series of keyposes. Neutral, jaw fully open, closed smile, lips funnel, lip pucker, brow up, brow down, eye closed. These keyposes are useful for quickly training a landmark model. As of ver 0.3 this tool is stil very much built for mostly stationary faces so if possible a head mounted camera is strongly recommended, but footage with a mostly stationary subject will still work.

Upon opening this video with StrongTrack you'll be presented with the video alongside a generic unmatched set of facial landmarks. Scrolling the video and entering landmarks is only possible when the video is paused. Pause the video with the SPACE KEY at the neutral pose and place the landmarks at the corresponding place upon the face. It's important you start with a neutral frame because this will enable you to later use the N key whenever you want to return the lips to a neutral pose, which is a great time saver.

Left mouse click to move individual points and right click to move face groups (jaw, eye, nose etc). The W key welds the lips together as a time saver. Points in white move points around them with a drop off in influence. 

Once you're happy with the placement add this to the training set with the F KEY. For this initial entry the model will then train the predictor automatically.

Head to the next frame (jaw open preferably because it is the most differnet) and repeat. Carry on in this manner, hitting the T KEY whenever you want to train the model. As the model becomes more accurate, less and less manual placement should be necessary. 

Hitting ESC will quit.

# Eye tracking
Still in development. Am trying to decide between blob detection in OpenCV or extending dlib's traing set.

# Exporting coefficients (not fully functional/documented)
Once you are happy with landmark tracking, return to the footage of the key poses (neutral, jaw open, smile etc) pause and hit E to add that to the set of keyposes for coefficient extraction. This is a potentially confusingly similar process to providing the landmark training with data but the difference here is that this is the data that will be used in the second step of the process; finding coefficients. However, I'm still working on the full functionality of the documentation for this, so this step is not recommended.

# Live webcam footage
This is something that I'm able to produce myself - once you have a model trained it's basically the exact same thing - but I'm struggling to incorporate it in a way that will be usable for general use.

