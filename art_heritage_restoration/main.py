"""
Art & Heritage Restoration - Modern GUI Application
Professional interface with real-time preview
"""
import cv2
import numpy as np
import sys
import os
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.image_enhancement import ImageEnhancer
from src.crack_segmentation import CrackSegmentationModel
from src.texture_classification import TextureClassifier
from src.style_transfer import StyleTransferModel
from src.inpainting import InpaintingGAN
from src.utils import load_image, save_image, create_mask

class ModernHeritageRestorationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🏛️ Art & Heritage Restoration System")
        self.root.geometry("1400x900")
        self.root.configure(bg='#1e1e1e')
        
        # Variables
        self.input_image = None
        self.original_image = None
        self.enhanced_image = None
        self.crack_mask = None
        self.final_image = None
        self.input_path = None
        
        # Initialize pipeline
        self.init_pipeline()
        
        # Create UI
        self.create_ui()
    
    def init_pipeline(self):
        """Initialize restoration pipeline"""
        print("Initializing pipeline...")
        self.enhancer = ImageEnhancer()
        self.crack_detector = CrackSegmentationModel()
        self.texture_classifier = TextureClassifier()
        self.style_transfer = StyleTransferModel()
        self.inpainting_gan = InpaintingGAN()
        print("Pipeline ready!")
    
    def create_ui(self):
        """Create modern UI"""
        # Style configuration
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TButton', font=('Segoe UI', 10), padding=10)
        style.configure('TLabel', background='#1e1e1e', foreground='white', font=('Segoe UI', 10))
        style.configure('Title.TLabel', font=('Segoe UI', 16, 'bold'))
        
        # Title Bar
        title_frame = tk.Frame(self.root, bg='#2d2d2d', height=80)
        title_frame.pack(fill='x', pady=(0, 10))
        
        title = tk.Label(title_frame, text="🏛️ Art & Heritage Restoration System", 
                        font=('Segoe UI', 24, 'bold'), bg='#2d2d2d', fg='#4CAF50')
        title.pack(pady=20)
        
        subtitle = tk.Label(title_frame, text="AI-Powered Cultural Heritage Preservation", 
                           font=('Segoe UI', 12), bg='#2d2d2d', fg='#b0b0b0')
        subtitle.pack()
        
        # Main Container
        main_container = tk.Frame(self.root, bg='#1e1e1e')
        main_container.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Left Panel - Controls
        left_panel = tk.Frame(main_container, bg='#2d2d2d', width=350)
        left_panel.pack(side='left', fill='y', padx=(0, 10))
        left_panel.pack_propagate(False)
        
        self.create_control_panel(left_panel)
        
        # Right Panel - Image Display
        right_panel = tk.Frame(main_container, bg='#1e1e1e')
        right_panel.pack(side='right', fill='both', expand=True)
        
        self.create_display_panel(right_panel)
        
        # Status Bar
        self.status_bar = tk.Label(self.root, text="Ready", 
                                   font=('Segoe UI', 9), bg='#2d2d2d', 
                                   fg='#4CAF50', anchor='w', padx=20)
        self.status_bar.pack(side='bottom', fill='x')
    
    def create_control_panel(self, parent):
        """Create control buttons panel"""
        # Load Image Section
        load_frame = tk.LabelFrame(parent, text=" 📂 Load Image ", 
                                   font=('Segoe UI', 12, 'bold'),
                                   bg='#2d2d2d', fg='white', padx=20, pady=20)
        load_frame.pack(fill='x', padx=10, pady=10)
        
        self.load_btn = tk.Button(load_frame, text="📁 Select Heritage Image", 
                                  command=self.load_image_file,
                                  bg='#4CAF50', fg='white', font=('Segoe UI', 11, 'bold'),
                                  relief='flat', cursor='hand2', padx=20, pady=15)
        self.load_btn.pack(fill='x', pady=5)
        
        self.file_label = tk.Label(load_frame, text="No file selected", 
                                   bg='#2d2d2d', fg='#888', font=('Segoe UI', 9))
        self.file_label.pack(pady=5)
        
        # Processing Section
        process_frame = tk.LabelFrame(parent, text=" ⚙️ Processing Steps ", 
                                      font=('Segoe UI', 12, 'bold'),
                                      bg='#2d2d2d', fg='white', padx=20, pady=20)
        process_frame.pack(fill='x', padx=10, pady=10)
        
        # Individual step buttons
        steps = [
            ("🎨 Enhance Image", self.run_enhancement, '#2196F3'),
            ("🔍 Detect Cracks", self.run_crack_detection, '#FF9800'),
            ("🧱 Classify Texture", self.run_texture_classification, '#9C27B0'),
            ("🖌️ Inpaint Damage", self.run_inpainting, '#F44336'),
        ]
        
        for text, command, color in steps:
            btn = tk.Button(process_frame, text=text, command=command,
                           bg=color, fg='white', font=('Segoe UI', 10),
                           relief='flat', cursor='hand2', padx=15, pady=12)
            btn.pack(fill='x', pady=5)
        
        # Run All Button
        self.run_all_btn = tk.Button(process_frame, text="⚡ RUN COMPLETE PIPELINE", 
                                     command=self.run_all_steps,
                                     bg='#4CAF50', fg='white', 
                                     font=('Segoe UI', 12, 'bold'),
                                     relief='flat', cursor='hand2', padx=20, pady=18)
        self.run_all_btn.pack(fill='x', pady=(15, 5))
        
        # Progress Bar
        self.progress = ttk.Progressbar(process_frame, mode='indeterminate')
        self.progress.pack(fill='x', pady=10)
        
        # Save Section
        save_frame = tk.LabelFrame(parent, text=" 💾 Save Results ", 
                                   font=('Segoe UI', 12, 'bold'),
                                   bg='#2d2d2d', fg='white', padx=20, pady=20)
        save_frame.pack(fill='x', padx=10, pady=10)
        
        self.save_btn = tk.Button(save_frame, text="💾 Save All Results", 
                                 command=self.save_results,
                                 bg='#607D8B', fg='white', font=('Segoe UI', 11),
                                 relief='flat', cursor='hand2', padx=20, pady=15)
        self.save_btn.pack(fill='x', pady=5)
        
        self.export_btn = tk.Button(save_frame, text="📊 View Detailed Report", 
                                    command=self.show_detailed_report,
                                    bg='#00BCD4', fg='white', font=('Segoe UI', 11),
                                    relief='flat', cursor='hand2', padx=20, pady=15)
        self.export_btn.pack(fill='x', pady=5)
    
    def create_display_panel(self, parent):
        """Create image display panel"""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill='both', expand=True)
        
        # Tab 1: Original & Enhanced
        tab1 = tk.Frame(self.notebook, bg='#1e1e1e')
        self.notebook.add(tab1, text='  Original & Enhanced  ')
        
        # Create comparison frame
        comp_frame = tk.Frame(tab1, bg='#1e1e1e')
        comp_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Original image
        orig_frame = tk.Frame(comp_frame, bg='#2d2d2d')
        orig_frame.pack(side='left', fill='both', expand=True, padx=5)
        
        tk.Label(orig_frame, text="Original Image", font=('Segoe UI', 14, 'bold'),
                bg='#2d2d2d', fg='white').pack(pady=10)
        
        self.original_canvas = tk.Label(orig_frame, bg='#1e1e1e')
        self.original_canvas.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Enhanced image
        enh_frame = tk.Frame(comp_frame, bg='#2d2d2d')
        enh_frame.pack(side='right', fill='both', expand=True, padx=5)
        
        tk.Label(enh_frame, text="Enhanced Image", font=('Segoe UI', 14, 'bold'),
                bg='#2d2d2d', fg='#4CAF50').pack(pady=10)
        
        self.enhanced_canvas = tk.Label(enh_frame, bg='#1e1e1e')
        self.enhanced_canvas.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Tab 2: Crack Detection
        tab2 = tk.Frame(self.notebook, bg='#1e1e1e')
        self.notebook.add(tab2, text='  Crack Detection  ')
        
        crack_frame = tk.Frame(tab2, bg='#2d2d2d')
        crack_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        tk.Label(crack_frame, text="Detected Cracks", font=('Segoe UI', 14, 'bold'),
                bg='#2d2d2d', fg='#FF9800').pack(pady=10)
        
        self.crack_canvas = tk.Label(crack_frame, bg='#1e1e1e')
        self.crack_canvas.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Tab 3: Final Result
        tab3 = tk.Frame(self.notebook, bg='#1e1e1e')
        self.notebook.add(tab3, text='  Final Restored  ')
        
        final_frame = tk.Frame(tab3, bg='#2d2d2d')
        final_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        tk.Label(final_frame, text="Final Restored Image", font=('Segoe UI', 14, 'bold'),
                bg='#2d2d2d', fg='#4CAF50').pack(pady=10)
        
        self.final_canvas = tk.Label(final_frame, bg='#1e1e1e')
        self.final_canvas.pack(fill='both', expand=True, padx=10, pady=10)
    
    def load_image_file(self):
        """Load image file"""
        file_path = filedialog.askopenfilename(
            title='Select Heritage Artifact Image',
            filetypes=[
                ('Image Files', '*.jpg *.jpeg *.png *.bmp *.tiff'),
                ('All Files', '*.*')
            ]
        )
        
        if file_path:
            self.input_path = file_path
            self.original_image = load_image(file_path)
            self.display_image(self.original_image, self.original_canvas)
            
            filename = os.path.basename(file_path)
            self.file_label.config(text=f"✓ {filename}", fg='#4CAF50')
            self.update_status(f"Loaded: {filename}")
    
    def display_image(self, image, canvas, max_size=(600, 600)):
        """Display image in canvas"""
        if image is None:
            return
        
        # Resize image to fit canvas
        h, w = image.shape[:2]
        scale = min(max_size[0]/w, max_size[1]/h)
        new_w, new_h = int(w*scale), int(h*scale)
        
        resized = cv2.resize(image, (new_w, new_h))
        
        # Convert to PIL format
        if len(resized.shape) == 2:
            pil_image = Image.fromarray(resized)
        else:
            pil_image = Image.fromarray(resized)
        
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(pil_image)
        
        canvas.config(image=photo)
        canvas.image = photo  # Keep reference
    
    def update_status(self, message):
        """Update status bar"""
        self.status_bar.config(text=message)
        self.root.update()
    
    def run_enhancement(self):
        """Run image enhancement"""
        if self.original_image is None:
            messagebox.showwarning("No Image", "Please load an image first!")
            return
        
        self.update_status("🎨 Enhancing image...")
        self.progress.start()
        
        self.enhanced_image = self.enhancer.enhance_pipeline(self.original_image)
        self.display_image(self.enhanced_image, self.enhanced_canvas)
        
        self.progress.stop()
        self.update_status("✓ Enhancement complete")
        self.notebook.select(0)  # Switch to comparison tab
    
    def run_crack_detection(self):
        """Run crack detection"""
        if self.enhanced_image is None:
            self.run_enhancement()
        
        self.update_status("🔍 Detecting cracks...")
        self.progress.start()
        
        self.crack_mask = self.crack_detector.predict(self.enhanced_image)
        
        # Create colored visualization
        crack_vis = cv2.applyColorMap(self.crack_mask, cv2.COLORMAP_HOT)
        crack_vis = cv2.cvtColor(crack_vis, cv2.COLOR_BGR2RGB)
        
        self.display_image(crack_vis, self.crack_canvas)
        
        self.progress.stop()
        self.update_status("✓ Crack detection complete")
        self.notebook.select(1)  # Switch to crack tab
    
    def run_texture_classification(self):
        """Run texture classification"""
        if self.enhanced_image is None:
            self.run_enhancement()
        
        self.update_status("🧱 Classifying texture...")
        self.progress.start()
        
        result = self.texture_classifier.predict(self.enhanced_image)
        
        self.progress.stop()
        self.update_status(f"✓ Texture: {result['class']} ({result['confidence']:.0%})")
        
        messagebox.showinfo("Texture Classification", 
                           f"Texture Type: {result['class']}\n"
                           f"Confidence: {result['confidence']:.1%}")
    
    def run_inpainting(self):
        """Run inpainting"""
        if self.crack_mask is None:
            self.run_crack_detection()
        
        if np.sum(self.crack_mask) == 0:
            messagebox.showinfo("No Damage", "No cracks detected. Skipping inpainting.")
            return
        
        self.update_status("🖌️ Inpainting damaged regions...")
        self.progress.start()
        
        working = self.enhanced_image if self.enhanced_image is not None else self.original_image
        self.final_image = self.inpainting_gan.inpaint(working, self.crack_mask)
        self.display_image(self.final_image, self.final_canvas)
        
        self.progress.stop()
        self.update_status("✓ Restoration complete")
        self.notebook.select(2)  # Switch to final tab
    
    def run_all_steps(self):
        """Run complete pipeline"""
        if self.original_image is None:
            messagebox.showwarning("No Image", "Please load an image first!")
            return
        
        self.update_status("⚡ Running complete pipeline...")
        
        # Run all steps
        self.run_enhancement()
        self.root.after(500, lambda: self.run_crack_detection())
        self.root.after(1000, lambda: self.run_texture_classification())
        self.root.after(1500, lambda: self.run_inpainting())
    
    def save_results(self):
        """Save all results"""
        if self.final_image is None:
            messagebox.showwarning("No Results", "Please run the pipeline first!")
            return
        
        output_dir = filedialog.askdirectory(title="Select Output Directory")
        if not output_dir:
            return
        
        try:
            save_image(self.enhanced_image, f'{output_dir}/1_enhanced.jpg')
            save_image(self.crack_mask, f'{output_dir}/2_crack_mask.jpg')
            save_image(self.final_image, f'{output_dir}/3_final_restored.jpg')
            
            messagebox.showinfo("Success", f"Results saved to:\n{output_dir}")
            self.update_status(f"✓ Results saved to {output_dir}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save: {str(e)}")
    
    def show_detailed_report(self):
        """Show detailed matplotlib report"""
        if self.final_image is None:
            messagebox.showwarning("No Results", "Please run the pipeline first!")
            return
        
        fig = plt.figure(figsize=(16, 10))
        fig.patch.set_facecolor('#1e1e1e')
        
        plt.subplot(2, 3, 1)
        plt.imshow(self.original_image)
        plt.title('Original', color='white', fontsize=14, fontweight='bold')
        plt.axis('off')
        
        plt.subplot(2, 3, 2)
        plt.imshow(self.enhanced_image)
        plt.title('Enhanced', color='white', fontsize=14, fontweight='bold')
        plt.axis('off')
        
        plt.subplot(2, 3, 3)
        plt.imshow(self.crack_mask, cmap='hot')
        plt.title('Crack Detection', color='white', fontsize=14, fontweight='bold')
        plt.axis('off')
        
        plt.subplot(2, 3, 4)
        overlay = self.enhanced_image.copy()
        overlay[self.crack_mask > 0] = [255, 0, 0]
        plt.imshow(overlay)
        plt.title('Cracks Highlighted', color='white', fontsize=14, fontweight='bold')
        plt.axis('off')
        
        plt.subplot(2, 3, 5)
        plt.imshow(self.final_image)
        plt.title('Final Restored', color='white', fontsize=14, fontweight='bold')
        plt.axis('off')
        
        plt.subplot(2, 3, 6)
        h, w = self.original_image.shape[:2]
        target_h = 400
        target_w = int(w * (target_h / h))
        comparison = np.hstack([
            cv2.resize(self.original_image, (target_w, target_h)),
            cv2.resize(self.final_image, (target_w, target_h))
        ])
        plt.imshow(comparison)
        plt.title('Before → After', color='white', fontsize=14, fontweight='bold')
        plt.axis('off')
        
        plt.suptitle('🏛️ Art & Heritage Restoration - Detailed Report', 
                    color='white', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

def main():
    root = tk.Tk()
    app = ModernHeritageRestorationGUI(root)
    root.mainloop()

if __name__ == '__main__':
    main()

