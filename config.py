import torch

class Config:
    # Directories
    #AUDIO_DIR = "../../../classifier/dataset/genres_original"
    AUDIO_DIR = "../AudioGeneration/dataset/genres_original"
    INFERENCE_AUDIO_DIR = "../AudioGenerationRifusion/dataset/genres_original"
    
    # Logging and Checkpoints
    LOG_FILE = "training_log.txt"
    CHECKPOINT_PATH = "./model_epoch_50.pth"

    # Device and Generator
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    #DEVICE = "cpu"
    GENERATOR = torch.Generator(device=DEVICE)
    
    # Data Parameters
    INPUT_CHANNELS = 1
    TEST_SIZE = 512
    BATCH_SIZE = 2
    NUM_CLASSES = 11
    VALIDATION_RATIO = 0.05
    TEST_RATIO = .8
    
    # Training Parameters
    DROPOUT_RATE = 0.25
    EPOCHS = 1000
    LEARNING_RATE = .0005
    PATIENCE = 4
    FACTOR = .8
    
    # Audio Processing Parameters
    # Audio Processing Parameters
    #SAMPLE_RATE = 44100
    #SECONDS = 30
    #N_FFT = 17640
    #HOP_LENGTH = 441
    #WIN_LENGTH = 4410
    #N_MELS = 512
    SAMPLE_RATE = 22025
    SECONDS = 30
    N_FFT = 2048
    HOP_LENGTH = 512
    WIN_LENGTH = 2048
    N_MELS = 128
    F_MIN = 40
    F_MAX = 11025
    
    # CLIP Model Parameters
    CLIP_N_HEAD = 12
    CLIP_N_EMBED = 16
    CLIP_NUM_LAYERS = 12
    CLIP_NUM_CATEGORIES = 10
    
    # Latent Shape (Needs to be defined)
    #LATENTS_HEIGHT = 64
    #LATENTS_WIDTH = 375
    LATENTS_HEIGHT = 16
    LATENTS_WIDTH = 161
    LATENTS_CHANNELS = 4
    LATENT_SHAPE = (BATCH_SIZE, LATENTS_CHANNELS, LATENTS_HEIGHT, LATENTS_WIDTH)