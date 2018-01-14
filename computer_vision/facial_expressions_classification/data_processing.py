# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 18:24:36 2017

@author: Karolis
"""
import os
import cv2
import glob as gb
import numpy as np
from shutil import copyfile
from face_detector.detector import DlibDetector


class ImageProcessor(DlibDetector):
    RIGHT_EYE_R, LEFT_EYE_L = 36, 45
    
    def __init__(self, facial_landmarks_predictor_path):
        super().__init__(facial_landmarks_predictor_path)
    
    def __rotate_image(self, img, rotate_around, angle):
        '''Rotates img around the rotate_around x, y 
        coordinates by the given angle'''
        rot_mat = cv2.getRotationMatrix2D(rotate_around, angle, scale=1.0)
        rotated = cv2.warpAffine(img, rot_mat, img.shape[:2])
        return rotated    
        

    def __get_centre_of_face(self, face, landmarks):
        '''Finds the centre of the face, 
        i.e. calculates the mean of x and y coordinates'''
        mean_x = []
        mean_y = []  
        
        for feature in self.FACIAL_LANDMARKS:
            for mark in range(self.FACIAL_LANDMARKS[feature][0],
                              self.FACIAL_LANDMARKS[feature][1]):
                mean_x.append(landmarks.part(mark).x)
                mean_y.append(landmarks.part(mark).y)    
                    
        mean_x = int(np.mean(mean_x))
        mean_y = int(np.mean(mean_y))
        
        return mean_x, mean_y
    
    
    def align_single(self, img_path):
        '''Attempts to align the face so that the eye level is horizontal. 
        There can only be one face per image for this to work.'''
        all_landmarks, faces, img = self.detect_from_path(img_path)
        
        if len(all_landmarks) == 1:
            landmarks = all_landmarks[0]
            angle = 0                            
            if landmarks.part(self.RIGHT_EYE_R).y != \
                landmarks.part(self.LEFT_EYE_L).y:
    
                slope = ((landmarks.part(self.RIGHT_EYE_R).y - 
                         landmarks.part(self.LEFT_EYE_L).y) /
                        (landmarks.part(self.RIGHT_EYE_R).x - 
                         landmarks.part(self.LEFT_EYE_L).x))
        
                angle = np.arctan(slope) * 180
                angle /= np.pi
            
            x, y = self.__get_centre_of_face(img, landmarks)
            img = self.__rotate_image(img, (x, y), angle) 
            
        return img

    def get_landmarks_vector(self, img_path, align=True, normalise=True):
        '''Returns a vector of concatenated x and y landmarks coordinates'''
        # TODO: figure out how to avoid detecting landmarks twice 
        # TODO: find out how to align-extract in one step
        
        if align:
            img = self.align_single(img_path)
        all_landmarks, faces, img = self.detect_from_raw(img) 
        
        if len(all_landmarks) == 1:
            X, Y = [], []
            
            for feature in self.FACIAL_LANDMARKS:
                for mark in range(self.FACIAL_LANDMARKS[feature][0],
                                  self.FACIAL_LANDMARKS[feature][1]):
                   X.append(all_landmarks[0].part(mark).x) 
                   Y.append(all_landmarks[0].part(mark).y)
                   
            if normalise:
                X = (X - np.mean(X)) / np.std(X)
                Y = (Y - np.mean(Y)) / np.std(Y)
            
            return [val for vals in zip(X, Y) for val in vals]
        
        
def get_all_data(sorted_data_root, labels):
        
    processor = ImageProcessor('face_detector/shape_predictor_68_face_landmarks.dat')
    X, y, img_paths  = [], [], []
    
    for label in labels:       
        for img_path in gb.glob(os.path.join(sorted_data_root, label, '*')):
            if 'jpg' in img_path or 'png' in img_path:
                img_paths.append(img_path)
                y.append(label)
                X.append(processor.get_landmarks_vector(img_path))
                
    return X, y, img_paths


def sort_CK_data(images_root, labels_root, labels_dict, save_to_path):
    # Create folders for the sorted facial emotions/expressions
    if not os.path.exists(save_to_path):
        os.mkdir(save_to_path)
    for label in labels_dict:
        path = os.path.join(save_to_path, labels_dict[label])
        if not os.path.exists(path):
            os.mkdir(path)
    
    # Sort the database
    for folder in gb.glob(os.path.join(labels_root, '*')):
        
        participant_num = str(folder[-4:])
        
        for session in gb.glob(os.path.join(folder, '*')):
            for label_file in gb.glob(os.path.join(session, '*')):
                
                current_session = label_file[39:-21]
                with open(label_file, 'r') as in_file:
                    int_label = int(float(in_file.readline()))
                             
                # First image displays the neutral facial expression
                neutral_img_path = gb.glob(os.path.join(images_root, 
                                                        participant_num, 
                                                        current_session, 
                                                        '*'))[0]

                
                if int_label in labels_dict:  
                    img_list = gb.glob(os.path.join(images_root, 
                                                    participant_num, 
                                                    current_session, 
                                                    '*'))  
                    # Last image displays the emotion/facial expression
                    if len(img_list):                                
                        emotion_img_path = img_list[-1]
                            
                print(os.path.join(save_to_path, 
                                      labels_dict[int_label],
                                      os.path.basename(emotion_img_path)))
                            

                # Copy the last image showing the emotion
                copyfile(emotion_img_path,
                         os.path.join(save_to_path, 
                                      labels_dict[int_label],
                                      os.path.basename(emotion_img_path)))
                    
                # Copy the first image showing the neutral emotion
                copyfile(neutral_img_path,
                         os.path.join(save_to_path, 
                                      labels_dict[0],
                                      os.path.basename(neutral_img_path)))
                
