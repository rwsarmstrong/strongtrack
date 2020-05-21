# StrongTracker
A python tool with facial landmark annotation and coefficient finder.

# Overview
This is a python based solution for finding coefficients for facial animation from RGB video. It was written for the purpose of exporting coefficients that can be later exported as a numpy save file (for importing into Blender for example) but can also stream into Unreal Engine (or elsewhere) via OSC.

This solution is made up of two core components; Facial landmark tracking and a decomposition to produce coefficients. For landmark tracking this respository includes a method to train and refine a model based on your own footage. Once you are satisfied with the landmark tracking you have each subject pull a number of distinct poses (neutral, smile, jaw open etc) and store these as key poses that are then used as the basis for the decomposition into coefficients. Landmarks from someone half smiling with their mouth open would be decomposited into a result with a 0.5 smile, 0.5 jaw open for example. These coefficients are what is then exported out to a save file or streamed (or both).

# Dependencies
* Python
