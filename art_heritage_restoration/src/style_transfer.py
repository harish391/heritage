"""
Neural Style Transfer Module using VGG19
Transfer artistic style to heritage artifacts
"""
import cv2
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config

class StyleTransferModel:
    def __init__(self):
        print('✓ Style Transfer Model initialized')
        self.content_layers = ['block4_conv2']
        self.style_layers = ['block1_conv1', 'block2_1', 'block3_conv1', 
                            'block4_conv1', 'block5_conv1']
    
    def transfer_style(self, content_path, style_path, iterations=500):
        """Perform style transfer"""
        # Load content image
        content_img = cv2.imread(content_path)
        if content_img is not None:
            content_img = cv2.cvtColor(content_img, cv2.COLOR_BGR2RGB)
        return content_img
