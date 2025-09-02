import os
from pathlib import Path


class Config:
    # Paths
    BASE_DIR = Path(__file__).parent.parent.parent
    DATA_DIR = BASE_DIR / "dataset"
    TRAIN_DIR = DATA_DIR / "Training"
    TEST_DIR = DATA_DIR / "Testing"
    MODEL_DIR = BASE_DIR / "experiments" / "saved_models"
    LOG_DIR = BASE_DIR / "experiments"

    # Data parameters
    IMG_SIZE_CLF = (224, 224)
    IMG_SIZE_SEG = (256, 256)
    NUM_CLASSES = 4
    CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

    # Training parameters
    BATCH_SIZE = 32
    LEARNING_RATE_CLF = 1e-4
    LEARNING_RATE_SEG = 1e-3
    EPOCHS_CLF = 50
    EPOCHS_SEG = 100
    EARLY_STOPPING_PATIENCE = 10

    # Model parameters
    CLF_BACKBONE = 'efficientnet-b3'
    SEG_BACKBONE = 'attention_unet'
    DROPOUT_RATE = 0.5

    # Augmentation parameters
    AUGMENT_PROB = 0.7
    ROTATION_RANGE = 15
    BRIGHTNESS_RANGE = 0.2

    # Medical thresholds
    TUMOR_SIZE_THRESHOLD = 30  # mm
    IRREGULARITY_THRESHOLD = 0.7
    MALIGNANT_VOLUME_THRESHOLD = 14000  # mm³

    # Device
    DEVICE = 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu'

    # Validation
    VAL_SPLIT = 0.2
    CROSS_VAL_FOLDS = 5

    # Seeds
    RANDOM_SEED = 42