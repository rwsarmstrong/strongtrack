#Import python libraries
from pythonosc import udp_client
import numpy as np
import cv2
import dlib
import os
import time
from datetime import datetime
import render_functions as rf
from sklearn.decomposition import SparseCoder

def getKeyPoints(points_store, neutral, tol):
    
    found_frames = []
    found_frames.append(neutral)
    neutral = points_store[neutral]
    
    first_diff, first_frame = getFirstDiff(points_store, neutral)
    #print(first_diff, first_frame)
    dict_2d_all = np.array([neutral, points_store[first_frame]]).astype(float)
    found_frames.append(first_frame)

    next_diff, next_frame = getDiffFrame(points_store, dict_2d_all)
    
    new = np.reshape(points_store[next_frame], (1,68,2))
    dict_2d_all = np.concatenate((dict_2d_all, new), axis=0)
    #print(next_diff, next_frame)
    
    while(next_diff > tol):
        next_diff, next_frame = getDiffFrame(points_store, dict_2d_all)
        new = np.reshape(points_store[next_frame], (1,68,2))
        dict_2d_all = np.concatenate((dict_2d_all, new), axis=0)
        print(next_diff, next_frame)
        found_frames.append(next_frame)
        
    return dict_2d_all, found_frames

def getFirstDiff(points_store, neutral):

    delta_store = []
    
    for i in range(len(points_store)):
        points = points_store[i]
        centroid= getCentre(points_store[i])
        centroid_neutral = getCentre(neutral)
        deltaneutral = (centroid[0]-centroid_neutral[0], centroid[1]-centroid_neutral[1])
        points_neutral = neutral+deltaneutral
        delta_overall = sum(sum(abs(points_neutral[48:68]-points[48:68])))
        delta_store.append(delta_overall)

    diff = max(delta_store)
    diff_frame = delta_store.index(max(delta_store))

    return diff, diff_frame

def getDiffFrame(points_store, dict_2d_all):

    delta_store = []
    
    for i in range(len(points_store)):
        
        centroid= getCentre(points_store[i])
        
        shiftedKeyposes = shiftKeyPoses(centroid, dict_2d_all)
        
        dict_2d = np.reshape(shiftedKeyposes, (dict_2d_all.shape[0],40))
        
        coder = SparseCoder(dictionary=dict_2d, transform_n_nonzero_coefs=None,
                        transform_alpha=10, transform_algorithm='lasso_lars')
        
        points = points_store[i][48:68].astype(float)
        
        points = np.reshape(points, (1,40))
     
        coeffs = coder.transform(points)
        
        points_debug = np.tensordot(coeffs, dict_2d_all, axes=1)    
        centroid_debug = getCentre(points_debug[0])

        deltadebug = (centroid[0]-centroid_debug[0], centroid[1]-centroid_debug[1])
                
        points_debug = points_debug[0] + deltadebug
        
        delta_overall = sum(sum(abs(points_store[i][48:68]-points_debug[48:68])))
        delta_store.append(delta_overall)
               
        diff = max(delta_store)
        diff_frame = delta_store.index(max(delta_store))
        
    return diff, diff_frame

def getCentre(points):
    mouth = points[48:68]
    centroid = mouth.mean(0)
    return (int(centroid[0]), int(centroid[1]))

def shiftKeyPoses(centroid, keyposes):
    
    new_poses = []
    
    for i in range(keyposes.shape[0]):
        
        keypose = np.array(keyposes[i][48:68])
            
        centroid_keypose = keypose.mean(0)
        delta = centroid_keypose-centroid
        new_pose = keyposes[i][48:68]-delta.astype(int)
        new_poses.append(new_pose)
        
    return np.array(new_poses)
