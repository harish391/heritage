"""
Crack Segmentation Module using U-Net Architecture
For detecting and segmenting cracks in heritage artifacts
"""
import numpy as np
import cv2
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config

class CrackSegmentationModel:
    def __init__(self):
        print('✓ Crack Segmentation Model initialized (using OpenCV edge detection)')
        self.model = None
    
    def predict(self, image):
        """Predict crack mask for single image using edge detection"""
        # Convert to uint8 if float32
        if len(image.shape) == 3 and image.dtype == np.float32:
            image = (image * 255).astype(np.uint8)
        
        # Convert to grayscale if RGB
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate to make cracks more visible
        crack_mask = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
        
        return crack_mask
