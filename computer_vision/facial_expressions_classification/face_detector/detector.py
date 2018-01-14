# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 15:53:54 2017

@author: Karolis
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 18:44:09 2017

@author: Karolis
"""

import os
import dlib
import cv2
import argparse
from copy import deepcopy
        
        
class DlibDetector():
    # contains the numbers of the first and last landmarks of each facial feature
    FACIAL_LANDMARKS = {'mouth'         : (48, 68),
                        'right_eyebrow' : (17, 22),
                        'left_eyebrow'  : (22, 27),
                        'right_eye'     : (36, 42),
                        'left_eye'      : (42, 48),
                        'nose'          : (27, 36),
                        'jaw'           : (0, 17)}
    
    def __init__(self, facial_landmarks_predictor_path='shape_predictor_68_face_landmarks.dat'):
        
        self.predictor = dlib.shape_predictor(facial_landmarks_predictor_path)
        self.detector = dlib.get_frontal_face_detector()
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
      
        
    def detect_from_raw(self,
                        img=None,
                        gray_scale=False,
                        clahe=False):
        '''
        Input
        -------
        img
            raw image
        gray_scale
            converts image into gray_scale
        clahe
            applies Contrast Limited Adaptive Histogram Equalization
            
        Returns
        -------
        all_landmarks
            list of objects containing the coordinates of the landmarks 
            for each facial feature
        faces
            object containing the coordinates of the detected faces rectangles
        img
            image object from img_path
        '''
        img = deepcopy(img)
        
        if gray_scale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
        if clahe: 
            # CLAHE (Contrast Limited Adaptive Histogram Equalization) 
            img = self.clahe.apply(img)
            
        all_landmarks = []
        faces = self.detector(img, 1)

        if faces:
            for k, d in enumerate(faces):
                all_landmarks.append(self.predictor(img, d))   
                
        return all_landmarks, faces, img

        
    def detect_from_path(self,
                         img_path=None,
                         img=None,
                         gray_scale=False,
                         clahe=False):
        '''
        Input
        -------
        img_path
            path to an image
        gray_scale
            converts image into gray_scale
        clahe
            applies Contrast Limited Adaptive Histogram Equalization
            
        Returns
        -------
        all_landmarks
            list of objects containing the coordinates of the landmarks 
            for each facial feature
        faces
            object containing the coordinates of the detected faces rectangles
        img
            image object from img_path
        '''
        return self.detect_from_raw(cv2.imread(img_path), gray_scale, clahe)
    
    
    def mark_landmarks(self,
                       img, 
                       faces, 
                       all_landmarks,
                       circle_faces,
                       mark_landmarks,
                       mark_landmarks_numbers,
                       circle_color=(0, 0, 255),
                       circle_thickness=2,
                       mark_color=(0, 0, 255),
                       mark_thickness=2,
                       font_scale=0.5,
                       font_thickness=1):
        
        img_copy = deepcopy(img)
        
        if circle_faces:
            for face in faces:
                point = face.center()
                radius = face.width() / 2
                
                cv2.circle(img_copy,
                           (point.x, point.y),
                           int(radius * 1.50),
                           circle_color,
                           circle_thickness)

        if mark_landmarks:
            for landmarks in all_landmarks:
                for feature in self.FACIAL_LANDMARKS:
                    for mark in range(self.FACIAL_LANDMARKS[feature][0],
                                      self.FACIAL_LANDMARKS[feature][1]):
                                
                                x, y = landmarks.part(mark).x, landmarks.part(mark).y
                                cv2.circle(img_copy, 
                                           (x, y), 
                                           1, 
                                           mark_color, 
                                           mark_thickness)
                                
                                if mark_landmarks_numbers:
                                    cv2.putText(img_copy, 
                                                str(mark),
                                                (x, y), 
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                font_scale,
                                                mark_color,
                                                font_thickness)
        return img_copy
        
    
    def save(self, 
             img, 
             faces,
             all_landmarks,
             save_as,
             circle_faces=False,
             mark_landmarks=False,
             mark_landmarks_numbers=False,
             resize=None):
        '''
        Saves the detected faces from an original image.
        
        Input
        -------
        img
            raw image
        faces
            detected faces in the img
        save_as
            basename to save the face imageas as, e.g.:
                if save_as = 'some_dir/some_image.jpg' (original image path) 
                then the first detected face image will be saved as 
                'some_image_face_1.jpg'            
        resize
            (new num_rows, num_cols)
        '''        
        img_copy = self.mark_landmarks(img, 
                                       faces, 
                                       all_landmarks, 
                                       circle_faces=circle_faces,
                                       mark_landmarks=mark_landmarks,
                                       mark_landmarks_numbers=mark_landmarks_numbers)
        
        circled = self.mark_landmarks(img, 
                                      faces, 
                                      all_landmarks, 
                                      circle_faces=True,
                                      mark_landmarks=mark_landmarks,
                                      mark_landmarks_numbers=mark_landmarks_numbers)
        
        save_to_path, extension = os.path.splitext(args.img_path)
        save_to_path += '_detected' + extension
        cv2.imwrite(save_to_path, circled)
        
        for face_num, face in enumerate(faces):
            point = face.center()
            radius = (face.width() / 2) * 1.5
            radius = int(radius)
            x, y = point.x - radius, point.y - radius
            w = h = radius * 2           

            face = img_copy[y:y+h, x:x+w]
            
            if resize:
                face = cv2.resize(face, resize)
                
            save_to_path = os.path.splitext(save_as)[0]
            extension = os.path.splitext(save_as)[1]
            save_to_path += '_face_' + str(face_num + 1) + extension
            
            cv2.imwrite(save_to_path, face)
            
            
    def display(self, 
                title,
                img, 
                faces,
                all_landmarks,
                circle_faces=True,
                mark_landmarks=False,
                mark_landmarks_numbers=False):
        '''
        Displays detected landmarks on an image
        
        Input
        -------
        img
            image object
        faces
            contains the coordinates of the detected face rectangles
        all_landmarks
            list of objects containing the coordinates of the landmarks 
            for each facial feature          
        '''
        
        marked = self.mark_landmarks(img, 
                                     faces, 
                                     all_landmarks,
                                     circle_faces,
                                     mark_landmarks,
                                     mark_landmarks_numbers)

        cv2.imshow(title, marked)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    
    ap.add_argument("img_path", 
                    help="path to an image")
    
    ap.add_argument("--facial_landmarks_predictor_path", 
                    help="path to shape_predictor_68_face_landmarks.dat",
                    default='shape_predictor_68_face_landmarks.dat')
    
    ap.add_argument("--gray_scale",
                    help="converts image into gray_scale",
                    action="store_true")
    ap.add_argument("--clahe",
                    help="applies Contrast Limited Adaptive Histogram Equalization",
                    action="store_true")
    
    ap.add_argument("--mark_landmarks",
                    help="displays the numbers on the landmarks",
                    action="store_true")   
    ap.add_argument("--mark_landmarks_numbers",
                    help="displays the numbers on the landmarks",
                    action="store_true")
    
    ap.add_argument("--save",
                    help=
                    '''
                    Saves detected 'faces' from an 'img' in the 'img_path'. 
                    All detected 'faces' are saved as img_path + '_face_' + 
                    str(face_number) + img_extension.
                    ''',
                    action="store_true")
    
    ap.add_argument("--resize",
                    help="Resize detected 'faces' before saving them",
                    type=int,
                    nargs=2)
                    
    args = ap.parse_args()
        
    detector = DlibDetector(args.facial_landmarks_predictor_path)
    
    all_landmarks, faces, img = detector.detect_from_path(args.img_path,
                                                          args.gray_scale,
                                                          args.clahe)
    
    detector.display(args.img_path,
                     img, 
                     faces,
                     all_landmarks, 
                     mark_landmarks=args.mark_landmarks,
                     mark_landmarks_numbers=args.mark_landmarks_numbers)    
    
    if args.save:            
        detector.save(img, 
                      faces,
                      all_landmarks,
                      args.img_path, 
                      mark_landmarks=args.mark_landmarks,
                      mark_landmarks_numbers=args.mark_landmarks_numbers,
                      resize=tuple(args.resize))

    
    