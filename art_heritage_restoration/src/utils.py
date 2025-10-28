import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

def load_image(image_path, target_size=None):
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f'Could not load image from {image_path}')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if target_size:
        img = cv2.resize(img, target_size)
    return img

def save_image(image, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), image)
    print(f'Image saved to: {output_path}')

def create_mask(image, mask_type='random', mask_ratio=0.2):
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    if mask_type == 'random':
        num_patches = int(h * w * mask_ratio / 400)
        for _ in range(num_patches):
            x = np.random.randint(0, max(1, w - 20))
            y = np.random.randint(0, max(1, h - 20))
            patch_w = np.random.randint(10, 30)
            patch_h = np.random.randint(10, 30)
            mask[y:y+patch_h, x:x+patch_w] = 255
    return mask

def display_images(images, titles=None, figsize=(15, 5), cmap=None):
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    for i, (img, ax) in enumerate(zip(images, axes)):
        ax.imshow(img, cmap=cmap)
        ax.axis('off')
        if titles:
            ax.set_title(titles[i])
    plt.tight_layout()
    plt.show()

def create_directory_structure(base_path):
    dirs = ['data/input', 'data/output', 'data/masks', 'data/test_images',
            'models/crack_detection', 'models/style_transfer', 
            'models/texture_classification', 'models/inpainting']
    for dir_path in dirs:
        full_path = os.path.join(base_path, dir_path)
        os.makedirs(full_path, exist_ok=True)
    print('Directory structure created successfully!')
