import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
INPUT_DIR = os.path.join(DATA_DIR, 'input')
OUTPUT_DIR = os.path.join(DATA_DIR, 'output')
MODELS_DIR = os.path.join(BASE_DIR, '..', 'models')

# Image Enhancement Parameters
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE = (8, 8)
GAUSSIAN_KERNEL = (5, 5)

# Crack Segmentation Parameters
UNET_INPUT_SIZE = (256, 256, 3)
CRACK_BATCH_SIZE = 16
CRACK_EPOCHS = 50
CRACK_LEARNING_RATE = 1e-4

# Texture Classification Parameters
TEXTURE_INPUT_SIZE = (224, 224, 3)
TEXTURE_CLASSES = ['regular', 'irregular', 'damaged', 'pristine']
NUM_CLASSES = len(TEXTURE_CLASSES)
TEXTURE_BATCH_SIZE = 32
TEXTURE_EPOCHS = 30

# Style Transfer Parameters
STYLE_IMG_SIZE = (512, 512)
STYLE_LEARNING_RATE = 0.003
STYLE_ITERATIONS = 1000
CONTENT_WEIGHT = 1e3
STYLE_WEIGHT = 1e-2

# Inpainting Parameters
INPAINT_INPUT_SIZE = (256, 256, 3)
INPAINT_BATCH_SIZE = 8
INPAINT_EPOCHS = 100
GAN_LEARNING_RATE = 2e-4
DEVICE = 'cuda'
