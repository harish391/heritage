"""
Image Enhancement Module for Heritage Restoration
Techniques: CLAHE, Histogram Equalization, Filtering, Denoising
"""
import cv2
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config

class ImageEnhancer:
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=config.CLAHE_CLIP_LIMIT, 
                                     tileGridSize=config.CLAHE_TILE_SIZE)
        print('✓ Image Enhancer initialized')
    
    def enhance_contrast_clahe(self, image):
        """Apply CLAHE for contrast enhancement"""
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            l_clahe = self.clahe.apply(l)
            enhanced_lab = cv2.merge([l_clahe, a, b])
            enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        else:
            enhanced = self.clahe.apply(image)
        return enhanced
    
    def denoise_image(self, image, method='bilateral'):
        """Apply denoising filters"""
        if method == 'bilateral':
            denoised = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
        elif method == 'gaussian':
            denoised = cv2.GaussianBlur(image, config.GAUSSIAN_KERNEL, 0)
        else:
            denoised = image
        return denoised
    
    def sharpen_image(self, image, amount=1.5):
        """Sharpen image using unsharp masking"""
        gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
        sharpened = cv2.addWeighted(image, 1.0 + amount, gaussian, -amount, 0)
        return sharpened
    
    def adjust_gamma(self, image, gamma=1.2):
        """Adjust gamma for brightness correction"""
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                         for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)
    
    def enhance_pipeline(self, image, steps=['clahe', 'denoise', 'sharpen']):
        """Complete enhancement pipeline"""
        enhanced = image.copy()
        for step in steps:
            if step == 'clahe':
                enhanced = self.enhance_contrast_clahe(enhanced)
            elif step == 'denoise':
                enhanced = self.denoise_image(enhanced, method='bilateral')
            elif step == 'sharpen':
                enhanced = self.sharpen_image(enhanced)
            elif step == 'gamma':
                enhanced = self.adjust_gamma(enhanced)
        return enhanced
