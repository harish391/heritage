"""
Image Inpainting Module using GAN
For restoring damaged/missing parts of heritage artifacts
"""
import numpy as np
import cv2
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config

class InpaintingGAN:
    def __init__(self):
        print('✓ Inpainting GAN initialized (using OpenCV inpainting)')
        self.generator = None
    
    def inpaint(self, damaged_image, mask):
        """Inpaint damaged regions using OpenCV"""
        if len(damaged_image.shape) == 3:
            # Use Telea inpainting algorithm
            result = cv2.inpaint(damaged_image, mask, 3, cv2.INPAINT_TELEA)
        else:
            result = damaged_image
        return result
