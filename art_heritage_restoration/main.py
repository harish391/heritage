"""
🏛️ Art & Heritage Restoration System - ULTIMATE EDITION
Complete with all upload methods and step-by-step visualization
"""
import cv2
import numpy as np
import sys
import os
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageGrab
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import time
from datetime import datetime
import threading

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.image_enhancement import ImageEnhancer
from src.crack_segmentation import CrackSegmentationModel
from src.texture_classification import TextureClassifier
from src.style_transfer import StyleTransferModel
from src.inpainting import InpaintingGAN
from src.utils import load_image, save_image, create_mask

class UltimateHeritageRestorationUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🏛️ Heritage Restoration System - Ultimate Edition")
        
        # Window setup
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        width = int(screen_width * 0.9)
        height = int(screen_height * 0.9)
        x = (screen_width - width) // 2
        y = (screen_height - height) // 2
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        self.root.configure(bg='#0a0a0a')
        
        # Variables
        self.original_image = None
        self.enhanced_image = None
        self.crack_mask = None
        self.final_image = None
        self.input_path = None
        self.current_step = 0
        self.total_steps = 5
        self.is_processing = False
        
        # Initialize pipeline
        self.init_pipeline()
        
        # Create UI
        self.create_ultimate_ui()
        
        # Setup keyboard shortcuts
        self.root.bind('<Control-v>', self.paste_from_clipboard)
        self.root.bind('<Control-o>', lambda e: self.load_from_file_dialog())
        
        # Start clock
        self.update_clock()
        
        # Show welcome message
        self.show_welcome()
    
    def init_pipeline(self):
        """Initialize restoration pipeline"""
        print("Initializing AI Pipeline...")
        self.enhancer = ImageEnhancer()
        self.crack_detector = CrackSegmentationModel()
        self.texture_classifier = TextureClassifier()
        self.style_transfer = StyleTransferModel()
        self.inpainting_gan = InpaintingGAN()
        print("Pipeline Ready!")
    
    def create_ultimate_ui(self):
        """Create ultimate UI"""
        # ============ HEADER ============
        header = tk.Frame(self.root, bg='#1a1a2e', height=100)
        header.pack(fill='x')
        header.pack_propagate(False)
        
        # Title with gradient effect
        title_canvas = tk.Canvas(header, bg='#1a1a2e', height=100, highlightthickness=0)
        title_canvas.pack(fill='both')
        
        for i in range(100):
            color = self.gradient_color('#1a1a2e', '#16213e', i/100)
            title_canvas.create_line(0, i, 3000, i, fill=color, width=1)
        
        title_canvas.create_text(502, 27, text="🏛️ HERITAGE RESTORATION AI", 
                                font=('Arial', 28, 'bold'), fill='#0a0a0a')
        title_canvas.create_text(500, 25, text="🏛️ HERITAGE RESTORATION AI", 
                                font=('Arial', 28, 'bold'), fill='#00d4ff')
        
        title_canvas.create_text(501, 66, text="Step-by-Step Cultural Heritage Preservation", 
                                font=('Arial', 11), fill='#1a1a1a')
        title_canvas.create_text(500, 65, text="Step-by-Step Cultural Heritage Preservation", 
                                font=('Arial', 11), fill='#00ff88')
        
        # ============ MAIN CONTAINER ============
        main = tk.Frame(self.root, bg='#0a0a0a')
        main.pack(fill='both', expand=True, padx=15, pady=10)
        
        # LEFT PANEL - Upload & Controls
        left = tk.Frame(main, bg='#16213e', width=350)
        left.pack(side='left', fill='y', padx=(0, 10))
        left.pack_propagate(False)
        self.create_upload_panel(left)
        
        # CENTER PANEL - Image Display
        center = tk.Frame(main, bg='#0a0a0a')
        center.pack(side='left', fill='both', expand=True, padx=10)
        self.create_display_panel(center)
        
        # RIGHT PANEL - Progress & Stats
        right = tk.Frame(main, bg='#16213e', width=300)
        right.pack(side='right', fill='y')
        right.pack_propagate(False)
        self.create_progress_panel(right)
        
        # ============ STATUS BAR ============
        self.create_status_bar()
    
    def create_upload_panel(self, parent):
        """Create upload methods panel"""
        # Title
        tk.Label(parent, text="📤 UPLOAD IMAGE", font=('Arial', 14, 'bold'),
                bg='#16213e', fg='#00d4ff', pady=15).pack()
        
        # Upload methods frame
        upload_frame = tk.Frame(parent, bg='#1a1a2e', padx=15, pady=15)
        upload_frame.pack(fill='x', padx=10, pady=5)
        
        # Method 1: File Dialog
        self.create_button(upload_frame, "📁 BROWSE FILES (Ctrl+O)", 
                          self.load_from_file_dialog, '#00d4ff').pack(fill='x', pady=5)
        
        # Method 2: Paste
        self.create_button(upload_frame, "📋 PASTE FROM CLIPBOARD (Ctrl+V)", 
                          self.paste_from_clipboard, '#00ff88').pack(fill='x', pady=5)
        
        # Method 3: Manual Path
        self.create_button(upload_frame, "⌨️ ENTER FILE PATH", 
                          self.enter_path_manually, '#8338ec').pack(fill='x', pady=5)
        
        # Method 4: Sample
        self.create_button(upload_frame, "🎨 LOAD SAMPLE IMAGE", 
                          self.load_sample_image, '#ff006e').pack(fill='x', pady=5)
        
        # File info
        self.file_info = tk.Label(upload_frame, text="No image loaded", 
                                 bg='#1a1a2e', fg='#888', font=('Arial', 9), 
                                 wraplength=280, justify='left', pady=10)
        self.file_info.pack(fill='x')
        
        # Separator
        tk.Frame(parent, bg='#00d4ff', height=2).pack(fill='x', pady=15)
        
        # PROCESSING CONTROLS
        tk.Label(parent, text="⚙️ PROCESSING", font=('Arial', 14, 'bold'),
                bg='#16213e', fg='#ffbe0b', pady=10).pack()
        
        control_frame = tk.Frame(parent, bg='#1a1a2e', padx=15, pady=15)
        control_frame.pack(fill='x', padx=10, pady=5)
        
        # Individual steps
        self.create_button(control_frame, "1️⃣ Enhance Image", 
                          self.step1_enhance, '#2196F3').pack(fill='x', pady=3)
        self.create_button(control_frame, "2️⃣ Detect Cracks", 
                          self.step2_cracks, '#FF9800').pack(fill='x', pady=3)
        self.create_button(control_frame, "3️⃣ Classify Texture", 
                          self.step3_texture, '#9C27B0').pack(fill='x', pady=3)
        self.create_button(control_frame, "4️⃣ Restore Damage", 
                          self.step4_restore, '#4CAF50').pack(fill='x', pady=3)
        
        tk.Frame(control_frame, bg='#555', height=1).pack(fill='x', pady=10)
        
        # Run all button
        self.run_all_btn = self.create_button(control_frame, "⚡ RUN ALL STEPS", 
                                              self.run_all_steps, '#00ff88', height=3)
        self.run_all_btn.pack(fill='x', pady=5)
        
        # Separator
        tk.Frame(parent, bg='#00d4ff', height=2).pack(fill='x', pady=15)
        
        # EXPORT
        tk.Label(parent, text="💾 EXPORT", font=('Arial', 14, 'bold'),
                bg='#16213e', fg='#00d4ff', pady=10).pack()
        
        export_frame = tk.Frame(parent, bg='#1a1a2e', padx=15, pady=15)
        export_frame.pack(fill='x', padx=10, pady=5)
        
        self.create_button(export_frame, "💾 Save Results", 
                          self.save_results, '#607D8B').pack(fill='x', pady=3)
        self.create_button(export_frame, "📊 View Report", 
                          self.show_report, '#00BCD4').pack(fill='x', pady=3)
    
    def create_display_panel(self, parent):
        """Create image display panel"""
        # Notebook for tabs
        style = ttk.Style()
        style.configure('Custom.TNotebook', background='#0a0a0a')
        style.configure('Custom.TNotebook.Tab', 
                       background='#16213e', foreground='white',
                       padding=[15, 8], font=('Arial', 10, 'bold'))
        style.map('Custom.TNotebook.Tab',
                 background=[('selected', '#00d4ff')],
                 foreground=[('selected', '#000')])
        
        self.notebook = ttk.Notebook(parent, style='Custom.TNotebook')
        self.notebook.pack(fill='both', expand=True)
        
        # Tab 1: Original
        tab1 = tk.Frame(self.notebook, bg='#0a0a0a')
        self.create_image_view(tab1, "ORIGINAL IMAGE")
        self.original_canvas = self.image_label
        self.notebook.add(tab1, text='  📷 ORIGINAL  ')
        
        # Tab 2: Enhanced
        tab2 = tk.Frame(self.notebook, bg='#0a0a0a')
        self.create_image_view(tab2, "ENHANCED IMAGE")
        self.enhanced_canvas = self.image_label
        self.notebook.add(tab2, text='  🎨 ENHANCED  ')
        
        # Tab 3: Cracks
        tab3 = tk.Frame(self.notebook, bg='#0a0a0a')
        self.create_image_view(tab3, "CRACK DETECTION")
        self.crack_canvas = self.image_label
        self.notebook.add(tab3, text='  🔍 CRACKS  ')
        
        # Tab 4: Final
        tab4 = tk.Frame(self.notebook, bg='#0a0a0a')
        self.create_image_view(tab4, "FINAL RESTORED")
        self.final_canvas = self.image_label
        self.notebook.add(tab4, text='  ✨ RESTORED  ')
    
    def create_image_view(self, parent, title):
        """Create image view with title"""
        frame = tk.Frame(parent, bg='#1a1a2e', relief='ridge', bd=3)
        frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        tk.Label(frame, text=title, font=('Arial', 16, 'bold'),
                bg='#1a1a2e', fg='#00d4ff', pady=12).pack()
        
        self.image_label = tk.Label(frame, bg='#0a0a0a', text="No image")
        self.image_label.pack(fill='both', expand=True, padx=10, pady=10)
    
    def create_progress_panel(self, parent):
        """Create progress and stats panel"""
        # Title
        tk.Label(parent, text="📊 LIVE PROGRESS", font=('Arial', 14, 'bold'),
                bg='#16213e', fg='#00d4ff', pady=15).pack()
        
        # Progress frame
        progress_frame = tk.Frame(parent, bg='#1a1a2e', padx=15, pady=15)
        progress_frame.pack(fill='x', padx=10, pady=5)
        
        # Step indicator
        self.step_label = tk.Label(progress_frame, text="⏳ Waiting to start...", 
                                   font=('Arial', 11, 'bold'),
                                   bg='#1a1a2e', fg='#00d4ff', pady=10)
        self.step_label.pack()
        
        # Progress bar
        self.progress_bar = ttk.Progressbar(progress_frame, mode='determinate', 
                                           length=250, maximum=100)
        self.progress_bar.pack(pady=10)
        
        # Progress percentage
        self.progress_percent = tk.Label(progress_frame, text="0%", 
                                        font=('Arial', 20, 'bold'),
                                        bg='#1a1a2e', fg='#00ff88')
        self.progress_percent.pack(pady=5)
        
        # Step checklist
        tk.Label(progress_frame, text="STEPS COMPLETED:", 
                font=('Arial', 10, 'bold'),
                bg='#1a1a2e', fg='#aaa', pady=10).pack()
        
        self.steps_frame = tk.Frame(progress_frame, bg='#1a1a2e')
        self.steps_frame.pack(fill='x', pady=5)
        
        self.step_indicators = {}
        steps = [
            "📥 Image Loaded",
            "🎨 Enhanced",
            "🔍 Cracks Detected",
            "🧱 Texture Analyzed",
            "✨ Fully Restored"
        ]
        
        for step in steps:
            indicator = tk.Label(self.steps_frame, text=f"⚪ {step}",
                               font=('Arial', 9), bg='#1a1a2e', fg='#666',
                               anchor='w', padx=10, pady=3)
            indicator.pack(fill='x')
            self.step_indicators[step] = indicator
        
        # Separator
        tk.Frame(parent, bg='#00d4ff', height=2).pack(fill='x', pady=15)
        
        # STATISTICS
        tk.Label(parent, text="📈 STATISTICS", font=('Arial', 14, 'bold'),
                bg='#16213e', fg='#ffbe0b', pady=10).pack()
        
        stats_frame = tk.Frame(parent, bg='#1a1a2e', padx=15, pady=15)
        stats_frame.pack(fill='x', padx=10, pady=5)
        
        self.stats_widgets = {}
        stats = [
            ("⏱️ Time", "0.0s"),
            ("📏 Size", "0×0"),
            ("🔍 Cracks", "0"),
            ("🧱 Texture", "Unknown")
        ]
        
        for label, value in stats:
            card = tk.Frame(stats_frame, bg='#16213e', relief='solid', bd=1)
            card.pack(fill='x', pady=5)
            
            tk.Label(card, text=label, font=('Arial', 9),
                    bg='#16213e', fg='#aaa', pady=3).pack()
            
            val_label = tk.Label(card, text=value, font=('Arial', 14, 'bold'),
                                bg='#16213e', fg='#00d4ff', pady=3)
            val_label.pack()
            
            self.stats_widgets[label] = val_label
        
        # Clock
        clock_frame = tk.Frame(parent, bg='#1a1a2e')
        clock_frame.pack(side='bottom', fill='x', padx=10, pady=10)
        
        self.clock = tk.Label(clock_frame, text="", font=('Courier', 18, 'bold'),
                             bg='#1a1a2e', fg='#00ff88', pady=10)
        self.clock.pack()
    
    def create_status_bar(self):
        """Create status bar"""
        bar = tk.Frame(self.root, bg='#16213e', height=35)
        bar.pack(side='bottom', fill='x')
        bar.pack_propagate(False)
        
        self.status = tk.Label(bar, text="🟢 Ready | Load an image to begin", 
                              font=('Arial', 10), bg='#16213e', fg='#00ff88',
                              anchor='w', padx=15)
        self.status.pack(side='left', fill='both', expand=True)
        
        self.version = tk.Label(bar, text="v2.0 Ultimate", 
                               font=('Arial', 9), bg='#16213e', fg='#666', padx=15)
        self.version.pack(side='right')
    
    # ============ HELPER METHODS ============
    
    def create_button(self, parent, text, command, color, height=2):
        """Create styled button"""
        btn = tk.Button(parent, text=text, command=command,
                       font=('Arial', 10, 'bold'), bg=color, fg='white',
                       relief='flat', cursor='hand2', height=height,
                       activebackground=color, bd=0)
        
        lighter = self.lighten_color(color)
        btn.bind('<Enter>', lambda e: btn.config(bg=lighter))
        btn.bind('<Leave>', lambda e: btn.config(bg=color))
        
        return btn
    
    def gradient_color(self, c1, c2, ratio):
        """Generate gradient color"""
        r1, g1, b1 = int(c1[1:3], 16), int(c1[3:5], 16), int(c1[5:7], 16)
        r2, g2, b2 = int(c2[1:3], 16), int(c2[3:5], 16), int(c2[5:7], 16)
        
        r = int(r1 + (r2 - r1) * ratio)
        g = int(g1 + (g2 - g1) * ratio)
        b = int(b1 + (b2 - b1) * ratio)
        
        return f'#{r:02x}{g:02x}{b:02x}'
    
    def lighten_color(self, color):
        """Lighten a color"""
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        r = min(255, int(r * 1.2))
        g = min(255, int(g * 1.2))
        b = min(255, int(b * 1.2))
        return f'#{r:02x}{g:02x}{b:02x}'
    
    def display_image(self, image, canvas):
        """Display image in canvas"""
        if image is None:
            return
        
        canvas.update()
        max_w = max(canvas.winfo_width(), 400)
        max_h = max(canvas.winfo_height(), 400)
        
        h, w = image.shape[:2]
        scale = min((max_w-40)/w, (max_h-40)/h, 1.0)
        new_w, new_h = int(w*scale), int(h*scale)
        
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        pil_img = Image.fromarray(resized)
        photo = ImageTk.PhotoImage(pil_img)
        
        canvas.config(image=photo, text="")
        canvas.image = photo
    
    def update_status(self, text, color='#00ff88'):
        """Update status bar"""
        self.status.config(text=text, fg=color)
        self.root.update()
    
    def update_progress(self, step, total):
        """Update progress bar"""
        percent = int((step / total) * 100)
        self.progress_bar['value'] = percent
        self.progress_percent.config(text=f"{percent}%")
        self.root.update()
    
    def mark_step_complete(self, step_name):
        """Mark step as complete"""
        if step_name in self.step_indicators:
            self.step_indicators[step_name].config(
                text=f"🟢 {step_name}", fg='#00ff88'
            )
    
    def update_stat(self, label, value):
        """Update statistic"""
        if label in self.stats_widgets:
            self.stats_widgets[label].config(text=str(value))
    
    def update_clock(self):
        """Update clock"""
        now = datetime.now().strftime("%H:%M:%S")
        if hasattr(self, 'clock'):
            self.clock.config(text=now)
        self.root.after(1000, self.update_clock)
    
    def show_welcome(self):
        """Show welcome dialog"""
        msg = ("Welcome to Heritage Restoration AI!\n\n"
               "📤 UPLOAD METHODS:\n"
               "• Browse Files (Ctrl+O)\n"
               "• Paste from Clipboard (Ctrl+V)\n"
               "• Enter File Path\n"
               "• Load Sample Image\n\n"
               "⚡ The AI will process your image step-by-step!")
        
        messagebox.showinfo("Welcome! 🏛️", msg)
    
    # ============ UPLOAD METHODS ============
    
    def load_from_file_dialog(self):
        """Load via file dialog"""
        try:
            self.root.update()
            
            path = filedialog.askopenfilename(
                parent=self.root,
                title='Select Heritage Image',
                initialdir=os.path.expanduser('~'),
                filetypes=[
                    ('Images', '*.jpg *.jpeg *.png *.bmp'),
                    ('All Files', '*.*')
                ]
            )
            
            if path:
                self.load_image(path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open dialog:\n{str(e)}")
    
    def paste_from_clipboard(self, event=None):
        """Paste from clipboard"""
        try:
            img = ImageGrab.grabclipboard()
            
            if img is None:
                messagebox.showinfo("No Image", 
                                  "No image in clipboard!\n\n"
                                  "Copy an image (Ctrl+C) then paste here (Ctrl+V)")
                return
            
            # Convert to numpy
            img_np = np.array(img)
            if len(img_np.shape) == 3:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
            # Save temp
            temp = os.path.join(os.path.dirname(__file__), 'data', 'clipboard.png')
            os.makedirs(os.path.dirname(temp), exist_ok=True)
            cv2.imwrite(temp, img_np)
            
            self.load_image(temp, source="Clipboard")
            
        except Exception as e:
            messagebox.showerror("Paste Error", f"Failed:\n{str(e)}")
    
    def enter_path_manually(self):
        """Enter path manually"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Enter File Path")
        dialog.geometry("550x180")
        dialog.configure(bg='#16213e')
        dialog.transient(self.root)
        dialog.grab_set()
        
        x = (dialog.winfo_screenwidth() - 550) // 2
        y = (dialog.winfo_screenheight() - 180) // 2
        dialog.geometry(f"+{x}+{y}")
        
        tk.Label(dialog, text="Enter Full Path to Image:", 
                font=('Arial', 12, 'bold'),
                bg='#16213e', fg='#00d4ff', pady=15).pack()
        
        entry = tk.Entry(dialog, font=('Arial', 11), width=50)
        entry.pack(padx=20, pady=10)
        entry.focus()
        
        def load():
            path = entry.get().strip().strip('"')
            if path:
                dialog.destroy()
                self.load_image(path)
        
        entry.bind('<Return>', lambda e: load())
        
        tk.Button(dialog, text="Load", command=load,
                 bg='#00d4ff', fg='white', font=('Arial', 10, 'bold'),
                 padx=30, pady=10).pack(pady=15)
    
    def load_sample_image(self):
        """Load sample image"""
        # Create sample
        img = np.random.randint(120, 160, (600, 800, 3), dtype=np.uint8)
        
        # Add texture
        for _ in range(30):
            x1, y1 = np.random.randint(0, 800), np.random.randint(0, 600)
            x2, y2 = np.random.randint(0, 800), np.random.randint(0, 600)
            cv2.line(img, (x1, y1), (x2, y2), (80, 80, 80), 2)
        
        sample_path = os.path.join(os.path.dirname(__file__), 'data', 'sample.jpg')
        os.makedirs(os.path.dirname(sample_path), exist_ok=True)
        cv2.imwrite(sample_path, img)
        
        self.load_image(sample_path, source="Sample")
    
    def load_image(self, path, source="File"):
        """Common load method"""
        if not os.path.exists(path):
            messagebox.showerror("Not Found", f"File not found:\n{path}")
            return
        
        try:
            self.update_status(f"📂 Loading from {source}...", '#00d4ff')
            
            self.original_image = load_image(path)
            self.input_path = path
            
            self.display_image(self.original_image, self.original_canvas)
            
            filename = os.path.basename(path)
            h, w = self.original_image.shape[:2]
            
            self.file_info.config(
                text=f"✅ Loaded from {source}\n"
                     f"📄 {filename}\n"
                     f"📏 {w}×{h} pixels",
                fg='#00ff88'
            )
            
            self.update_stat("📏 Size", f"{w}×{h}")
            self.mark_step_complete("📥 Image Loaded")
            self.update_progress(1, 5)
            
            self.update_status(f"✅ Loaded: {filename}", '#00ff88')
            self.notebook.select(0)
            
            # Reset other images
            self.enhanced_image = None
            self.crack_mask = None
            self.final_image = None
            
            messagebox.showinfo("Success! 🎉", 
                              f"Image loaded from {source}!\n\n"
                              f"File: {filename}\n"
                              f"Size: {w}×{h} pixels\n\n"
                              f"Ready to process!")
            
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed:\n{str(e)}")
            self.update_status("❌ Load failed", '#ff006e')
    
    # ============ PROCESSING METHODS ============
    
    def step1_enhance(self):
        """Step 1: Enhance"""
        if self.original_image is None:
            messagebox.showwarning("No Image", "Load an image first!")
            return
        
        self.step_label.config(text="🎨 Step 1/4: Enhancing...")
        self.update_status("🎨 Enhancing image...", '#00d4ff')
        
        start = time.time()
        
        self.enhanced_image = self.enhancer.enhance_pipeline(self.original_image)
        self.display_image(self.enhanced_image, self.enhanced_canvas)
        
        elapsed = time.time() - start
        self.update_stat("⏱️ Time", f"{elapsed:.2f}s")
        self.mark_step_complete("🎨 Enhanced")
        self.update_progress(2, 5)
        
        self.step_label.config(text="✅ Enhancement Complete!")
        self.update_status("✅ Enhancement complete", '#00ff88')
        self.notebook.select(1)
        
        messagebox.showinfo("Step 1 Complete ✓", 
                          f"Image enhanced!\n\n"
                          f"Applied:\n"
                          f"• CLAHE contrast enhancement\n"
                          f"• Bilateral denoising\n"
                          f"• Unsharp masking\n\n"
                          f"Time: {elapsed:.2f}s")
    
    def step2_cracks(self):
        """Step 2: Detect cracks"""
        if self.enhanced_image is None:
            self.step1_enhance()
        
        self.step_label.config(text="🔍 Step 2/4: Detecting cracks...")
        self.update_status("🔍 Detecting cracks...", '#ff006e')
        
        start = time.time()
        
        self.crack_mask = self.crack_detector.predict(self.enhanced_image)
        
        # Visualize
        vis = cv2.applyColorMap(self.crack_mask, cv2.COLORMAP_INFERNO)
        vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        self.display_image(vis, self.crack_canvas)
        
        num_cracks = np.count_nonzero(self.crack_mask > 50)
        
        elapsed = time.time() - start
        self.update_stat("⏱️ Time", f"{elapsed:.2f}s")
        self.update_stat("🔍 Cracks", str(num_cracks))
        self.mark_step_complete("🔍 Cracks Detected")
        self.update_progress(3, 5)
        
        self.step_label.config(text="✅ Crack Detection Complete!")
        self.update_status("✅ Cracks detected", '#00ff88')
        self.notebook.select(2)
        
        messagebox.showinfo("Step 2 Complete ✓", 
                          f"Crack detection complete!\n\n"
                          f"Found: {num_cracks} damaged pixels\n"
                          f"Method: U-Net edge detection\n\n"
                          f"Time: {elapsed:.2f}s")
    
    def step3_texture(self):
        """Step 3: Classify texture"""
        if self.enhanced_image is None:
            self.step1_enhance()
        
        self.step_label.config(text="🧱 Step 3/4: Analyzing texture...")
        self.update_status("🧱 Analyzing texture...", '#ffbe0b')
        
        result = self.texture_classifier.predict(self.enhanced_image)
        
        self.update_stat("🧱 Texture", result['class'].title())
        self.mark_step_complete("🧱 Texture Analyzed")
        self.update_progress(4, 5)
        
        self.step_label.config(text="✅ Texture Analysis Complete!")
        self.update_status(f"✅ Texture: {result['class']}", '#00ff88')
        
        messagebox.showinfo("Step 3 Complete ✓", 
                          f"Texture analyzed!\n\n"
                          f"Type: {result['class'].title()}\n"
                          f"Confidence: {result['confidence']:.1%}\n\n"
                          f"Classification complete!")
    
    def step4_restore(self):
        """Step 4: Restore"""
        if self.crack_mask is None:
            self.step2_cracks()
        
        if np.sum(self.crack_mask) == 0:
            messagebox.showinfo("No Damage", "No cracks to restore!")
            return
        
        self.step_label.config(text="✨ Step 4/4: Restoring...")
        self.update_status("✨ Restoring damage...", '#00ff88')
        
        start = time.time()
        
        working = self.enhanced_image if self.enhanced_image is not None else self.original_image
        self.final_image = self.inpainting_gan.inpaint(working, self.crack_mask)
        self.display_image(self.final_image, self.final_canvas)
        
        elapsed = time.time() - start
        self.update_stat("⏱️ Time", f"{elapsed:.2f}s")
        self.mark_step_complete("✨ Fully Restored")
        self.update_progress(5, 5)
        
        self.step_label.config(text="🎉 ALL STEPS COMPLETE!")
        self.update_status("🎉 Restoration complete!", '#00ff88')
        self.notebook.select(3)
        
        messagebox.showinfo("Step 4 Complete ✓", 
                          f"Restoration complete! 🎉\n\n"
                          f"All damage has been inpainted!\n"
                          f"Method: OpenCV inpainting\n\n"
                          f"Time: {elapsed:.2f}s\n\n"
                          f"Your heritage artifact has been restored!")
    
    def run_all_steps(self):
        """Run all steps sequentially"""
        if self.original_image is None:
            messagebox.showwarning("No Image", "Load an image first!")
            return
        
        self.update_status("⚡ Running complete pipeline...", '#ffbe0b')
        
        # Run steps with delays
        self.root.after(100, self.step1_enhance)
        self.root.after(1500, self.step2_cracks)
        self.root.after(3000, self.step3_texture)
        self.root.after(4000, self.step4_restore)
    
    def save_results(self):
        """Save results"""
        if self.final_image is None:
            messagebox.showwarning("No Results", "Complete the pipeline first!")
            return
        
        output = filedialog.askdirectory(title="Select Output Folder")
        if output:
            try:
                save_image(self.enhanced_image, f'{output}/1_enhanced.jpg')
                save_image(self.crack_mask, f'{output}/2_cracks.jpg')
                save_image(self.final_image, f'{output}/3_restored.jpg')
                
                messagebox.showinfo("Saved! 💾", 
                                  f"Results saved to:\n{output}\n\n"
                                  f"Files:\n"
                                  f"• 1_enhanced.jpg\n"
                                  f"• 2_cracks.jpg\n"
                                  f"• 3_restored.jpg")
                
                self.update_status(f"💾 Saved to {output}", '#00ff88')
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed:\n{str(e)}")
    
    def show_report(self):
        """Show detailed report"""
        if self.final_image is None:
            messagebox.showwarning("No Results", "Complete the pipeline first!")
            return
        
        fig = plt.figure(figsize=(16, 9), facecolor='#0a0a0a')
        
        images = [
            (self.original_image, 'Original', None),
            (self.enhanced_image, 'Enhanced', None),
            (self.crack_mask, 'Cracks', 'hot'),
            (self.final_image, 'Restored', None)
        ]
        
        for idx, (img, title, cmap) in enumerate(images, 1):
            ax = plt.subplot(2, 2, idx)
            if cmap:
                ax.imshow(img, cmap=cmap)
            else:
                ax.imshow(img)
            ax.set_title(title, color='white', fontsize=16, fontweight='bold', pad=15)
            ax.axis('off')
        
        plt.suptitle('🏛️ Heritage Restoration Complete Report', 
                    color='#00d4ff', fontsize=20, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.show()

def main():
    root = tk.Tk()
    app = UltimateHeritageRestorationUI(root)
    root.mainloop()

if __name__ == '__main__':
    main()
