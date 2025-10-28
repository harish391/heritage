"""
Main Pipeline for Art & Heritage Restoration
With GUI file picker and visual output windows
"""
import cv2
import numpy as np
import argparse
import sys
import os
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.image_enhancement import ImageEnhancer
from src.crack_segmentation import CrackSegmentationModel
from src.texture_classification import TextureClassifier
from src.style_transfer import StyleTransferModel
from src.inpainting import InpaintingGAN
from src.utils import load_image, save_image, create_mask

class HeritageRestorationPipeline:
    def __init__(self):
        print('='*60)
        print('INITIALIZING HERITAGE RESTORATION PIPELINE')
        print('='*60)
        self.enhancer = ImageEnhancer()
        self.crack_detector = CrackSegmentationModel()
        self.texture_classifier = TextureClassifier()
        self.style_transfer = StyleTransferModel()
        self.inpainting_gan = InpaintingGAN()
        print('='*60)
        print('PIPELINE READY!')
        print('='*60)
    
    def show_results(self, original, enhanced, crack_mask, final):
        """Display results in separate window"""
        plt.figure(figsize=(16, 10))
        
        # Original Image
        plt.subplot(2, 3, 1)
        plt.imshow(original)
        plt.title('1. Original Image', fontsize=14, fontweight='bold')
        plt.axis('off')
        
        # Enhanced Image
        plt.subplot(2, 3, 2)
        plt.imshow(enhanced)
        plt.title('2. Enhanced (CLAHE + Denoise + Sharpen)', fontsize=14, fontweight='bold')
        plt.axis('off')
        
        # Crack Detection Mask
        plt.subplot(2, 3, 3)
        plt.imshow(crack_mask, cmap='hot')
        plt.title('3. Crack Detection Mask', fontsize=14, fontweight='bold')
        plt.axis('off')
        
        # Crack Overlay
        plt.subplot(2, 3, 4)
        overlay = enhanced.copy()
        if len(crack_mask.shape) == 2:
            overlay[crack_mask > 0] = [255, 0, 0]  # Red overlay for cracks
        plt.imshow(overlay)
        plt.title('4. Cracks Highlighted (Red)', fontsize=14, fontweight='bold')
        plt.axis('off')
        
        # Final Restored
        plt.subplot(2, 3, 5)
        plt.imshow(final)
        plt.title('5. Final Restored Image', fontsize=14, fontweight='bold')
        plt.axis('off')
        
        # Comparison
        plt.subplot(2, 3, 6)
        h, w = original.shape[:2]
        target_h = 400
        target_w = int(w * (target_h / h))
        comparison = np.hstack([
            cv2.resize(original, (target_w, target_h)), 
            cv2.resize(final, (target_w, target_h))
        ])
        plt.imshow(comparison)
        plt.title('6. Before → After', fontsize=14, fontweight='bold')
        plt.axis('off')
        
        plt.tight_layout()
        plt.suptitle('🏛️ Art & Heritage Restoration Results', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.show()
    
    def run_complete_pipeline(self, input_path, style_path=None, output_dir='data/output', show_visual=True):
        print(f'\n📂 Loading image: {input_path}')
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        original = load_image(input_path)
        print(f'✓ Image loaded: {original.shape}')
        
        print('\n🎨 Step 1: Enhancing image...')
        enhanced = self.enhancer.enhance_pipeline(original)
        save_image(enhanced, f'{output_dir}/1_enhanced.jpg')
        print('  ✓ Enhancement complete')
        
        print('🔍 Step 2: Detecting cracks...')
        crack_mask = self.crack_detector.predict(enhanced)
        save_image(crack_mask, f'{output_dir}/2_crack_mask.jpg')
        print('  ✓ Crack detection complete')
        
        print('🧱 Step 3: Classifying texture...')
        texture = self.texture_classifier.predict(enhanced)
        print(f'  ✓ Texture: {texture["class"]} ({texture["confidence"]:.0%} confidence)')
        
        if style_path and os.path.exists(style_path):
            print('🎭 Step 4: Applying style transfer...')
            temp_path = 'data/temp_content.jpg'
            save_image(enhanced, temp_path)
            stylized = self.style_transfer.transfer_style(temp_path, style_path)
            if stylized is not None:
                save_image(stylized, f'{output_dir}/4_stylized.jpg')
                working = stylized
                print('  ✓ Style transfer complete')
            else:
                working = enhanced
        else:
            working = enhanced
        
        if np.sum(crack_mask) > 0:
            print('🖌️ Step 5: Inpainting damaged regions...')
            restored = self.inpainting_gan.inpaint(working, crack_mask)
            save_image(restored, f'{output_dir}/5_final_restored.jpg')
            print('  ✓ Inpainting complete')
        else:
            restored = working
            save_image(working, f'{output_dir}/5_final_restored.jpg')
            print('  ℹ️ No cracks detected, skipping inpainting')
        
        print(f'\n✅ PROCESSING COMPLETE!')
        print(f'📁 All outputs saved to: {output_dir}')
        
        # Show visual results
        if show_visual:
            print('\n🖼️ Displaying results in new window...')
            self.show_results(original, enhanced, crack_mask, restored)
        
        return original, enhanced, crack_mask, restored

def select_image_file():
    """Open file dialog to select image"""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    root.attributes('-topmost', True)  # Bring dialog to front
    
    file_path = filedialog.askopenfilename(
        title='Select Heritage Artifact Image',
        filetypes=[
            ('Image Files', '*.jpg *.jpeg *.png *.bmp *.tiff'),
            ('JPEG', '*.jpg *.jpeg'),
            ('PNG', '*.png'),
            ('All Files', '*.*')
        ]
    )
    
    root.destroy()
    return file_path

def main():
    parser = argparse.ArgumentParser(description='Heritage Restoration Pipeline')
    parser.add_argument('--input', type=str, help='Input image path (optional - will open file picker if not provided)')
    parser.add_argument('--style', type=str, help='Style image (optional)')
    parser.add_argument('--output', type=str, default='data/output', help='Output directory')
    parser.add_argument('--step', type=str, default='all', help='Step to run')
    parser.add_argument('--no-display', action='store_true', help='Disable visual display')
    args = parser.parse_args()
    
    # Get input file
    if args.input and os.path.exists(args.input):
        input_path = args.input
        print(f'\n✓ Using specified image: {input_path}')
    else:
        print('\n📂 Opening file picker... Please select your heritage artifact image.')
        input_path = select_image_file()
        
        if not input_path:
            print('❌ No file selected. Exiting.')
            return
        print(f'✓ Selected: {input_path}')
    
    if not os.path.exists(input_path):
        print(f'❌ ERROR: File not found: {input_path}')
        return
    
    # Initialize pipeline
    pipeline = HeritageRestorationPipeline()
    
    # Run pipeline
    if args.step == 'all':
        pipeline.run_complete_pipeline(
            input_path, 
            args.style, 
            args.output,
            show_visual=not args.no_display
        )
    elif args.step == 'enhance':
        image = load_image(input_path)
        result = pipeline.enhancer.enhance_pipeline(image)
        os.makedirs(args.output, exist_ok=True)
        save_image(result, f'{args.output}/enhanced.jpg')
        
        # Show comparison
        if not args.no_display:
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(image)
            plt.title('Original', fontsize=14, fontweight='bold')
            plt.axis('off')
            plt.subplot(1, 2, 2)
            plt.imshow(result)
            plt.title('Enhanced', fontsize=14, fontweight='bold')
            plt.axis('off')
            plt.tight_layout()
            plt.show()
    
    print('\n✅ All done! Press any key to exit...')
    input()

if __name__ == '__main__':
    main()
