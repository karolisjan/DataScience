# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 13:59:54 2017

@author: Karolis

Compares two images using Structural Similarity Index Measure (SSIM)

References:
    http://www.cns.nyu.edu/pub/eero/wang03-reprint.pdf
    http://www.pyimagesearch.com/2017/06/19/image-difference-with-opencv-and-python/
"""
import cv2
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from skimage.measure import compare_ssim as SSIM


def draw_contours(image, contours, color):
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)


def show_diff(image_a, image_b, image_ssim, ssim, similarity):
    image_ssim = (image_ssim * 255).astype("uint8")
    
    # Apply thresholding technique to find the contours
    threshold = cv2.threshold(image_ssim, 
                              0, 
                              255, 
                              cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(threshold.copy(), 
                                cv2.RETR_EXTERNAL, 
                                cv2.CHAIN_APPROX_SIMPLE)[1]
    
    # Draw the contours on the original images where differences were detected
    draw_contours(image_a, contours, (0, 255, 0))
    draw_contours(image_b, contours, (0, 255, 0))
  
    fig, axes = plt.subplots(ncols=2)
    plt.suptitle('SSIM = %.3f, Similarity = %.2f%%' % (ssim, similarity))    
    axes[0].imshow(cv2.cvtColor(image_a, cv2.COLOR_BGR2RGB))
    axes[0].axis('off')
    axes[1].imshow(cv2.cvtColor(image_b, cv2.COLOR_BGR2RGB))
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()
    
    cv2.imshow("SSIM", image_ssim)
    cv2.imshow("Threshold", threshold)
    cv2.waitKey()


def compare_images_raw(image_a, image_b, show=False):
    bw_image_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
    bw_image_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)
    
    # Ensure both images have the sime size, by downsampling the larger one
    # to the smaller one's dimensions    
    if bw_image_a.shape[0] < bw_image_b.shape[0] or bw_image_a.shape[1] < bw_image_b.shape[1]:
        bw_image_b = cv2.resize(bw_image_b, bw_image_a.shape[::-1])
    elif bw_image_a.shape[0] > bw_image_b.shape[0] or bw_image_a.shape[1] > bw_image_b.shape[1]:
        bw_image_a = cv2.resize(bw_image_a, bw_image_b.shape[::-1])
            
    # Calculate ssim value and image     
    ssim, image_ssim = SSIM(bw_image_a, bw_image_b, full=True)
    
    # Convert ssim value to a percentage
    similarity = 100 * (ssim + 1) / 2.0  
    
    if show:
        show_diff(image_a, image_b, image_ssim, ssim, similarity)
        
    return similarity


def compare_images_paths(image_a_path, image_b_path, show=False):
    image_a = cv2.imread(image_a_path)
    image_b = cv2.imread(image_b_path)
    
    return compare_images_raw(image_a, image_b, show)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-a", "--image_a",
                        help="path to Image A",
                        required="True",
                        type=str)  
    parser.add_argument("-b", "--image_b",
                        help="path to Image B",
                        required="True",
                        type=str)
    parser.add_argument("--show",
                        help="display both images and the differences between them",
                        action="store_true")
    args = parser.parse_args()
    
    print(compare_images_paths(args.image_a, args.image_b, args.show))
    