# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 14:41:17 2017

@author: Karolis

Cuts images in two equal parts along the specified axis
"""
import os
import cv2
from argparse import ArgumentParser 


def crop_image(image, axis, image_path):
    # y - rows, x - columns
    rows, cols = image.shape[:2]
    
    # NOTE: img[y: y + h, x: x + w]
    if axis:
        part1 = image[:rows, :cols//2]
        part2 = image[:rows, cols//2:]
    else:
        part1 = image[:rows//2, :cols]
        part2 = image[rows//2:, :cols]
       
    filename, ext = os.path.splitext(image_path)
    cv2.imwrite(''.join([filename, '_part1', ext]), part1)
    cv2.imwrite(''.join([filename, '_part2', ext]), part2)
        

def crop_image_paths(image_path, axis):
    image = cv2.imread(image_path)
    crop_image(image, axis, image_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--image",
                        help="path to Image",
                        required="True",
                        type=str)  
    parser.add_argument("--axis",
                        help="axis to cut along",
                        required=True,
                        type=int,
                        choices=[0, 1])
    args = parser.parse_args()
    
    crop_image_paths(args.image, args.axis)
