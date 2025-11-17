import os

# --- Base Paths ---
BASE_PATH = os.path.expanduser('~/yolo-animal-classifier')
SOUND_MODEL_DIR = os.path.expanduser('/home/aadhi/Real time')

# --- Model Paths ---
IMAGE_MODEL_PATH = os.path.join(BASE_PATH, 'models/final_best_fine_tuned.keras')
SOUND_MODEL_PATH = os.path.join(SOUND_MODEL_DIR, 'crnn_checkpoint_71.h5')
SOUND_LABELS_PATH = os.path.join(SOUND_MODEL_DIR, 'y_labels.npy')
YOLO_MODEL = 'yolov8n.pt' # Or your specific YOLO model

# --- Audio Settings ---
FS = 22050
N_MFCC = 40
RECORD_DURATION_SEC = 4.0
TARGET_SAMPLES = int(RECORD_DURATION_SEC * FS)
CHANNELS = 1
CHUNK_SIZE = 1024
AUDIO_THRESHOLD = 0.1 # Volume threshold for triggering

# --- Inference Settings ---
IMAGE_CONFIDENCE_THRESHOLD = 0.75
SOUND_CONFIDENCE_THRESHOLD = 0.90
NOISE_CLASS_NAME = "Background"
THREAT_CLASSES = ['Tiger', 'Elephant', 'Bear', 'Wild_Boar']

# --- Output ---
DETERRENT_SOUND_PATH = os.path.join(SOUND_MODEL_DIR, 'Detterent Voice/Deterrent.wav')
SNAPSHOT_DIR = os.path.join(BASE_PATH, "snapshot_keras")