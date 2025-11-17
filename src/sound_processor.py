import sounddevice as sd
import numpy as np
import os
import time
import librosa
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import subprocess
import queue
import config # Import our new config file

# --- Global Vars for this Thread ---
model = None
class_names = []
audio_buffer = np.zeros(config.TARGET_SAMPLES, dtype='float32')
is_recording = False
recording_start_time = 0.0

def setup_sound_model():
    """Loads the CRNN model and labels."""
    global model, class_names
    try:
        model = load_model(config.SOUND_MODEL_PATH)
        y_original_labels = np.load(config.SOUND_LABELS_PATH)
        le = LabelEncoder()
        le.fit(y_original_labels)
        class_names = le.classes_
        print(f"[Audio Thread] Sound model and labels loaded successfully. Classes: {class_names}")
    except Exception as e:
        print(f"[Audio Thread] CRITICAL ERROR: Failed to load sound model: {e}")

def convert_to_crnn_input(audio_data):
    """Converts a 4-second raw audio array to the 4D MFCC input."""
    y_fixed = librosa.util.fix_length(audio_data.squeeze(), size=config.TARGET_SAMPLES, mode='constant')
    mfccs = librosa.feature.mfcc(y=y_fixed, sr=config.FS, n_mfcc=config.N_MFCC)
    mfccs_transposed = mfccs.T
    input_3d = np.expand_dims(mfccs_transposed, axis=-1)
    final_crnn_input = np.expand_dims(input_3d, axis=0)
    return final_crnn_input

def play_deterrent():
    """Plays the deterrent sound using a non-blocking subprocess."""
    try:
        subprocess.Popen(['aplay', config.DETERRENT_SOUND_PATH], 
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("[Audio Thread] !!! ACTIVATING ACOUSTIC DETERRENT !!!")
    except Exception as e:
        print(f"[Audio Thread] ERROR playing sound: {e}")

def audio_callback(indata, frames, time_info, status):
    """This is the core function that runs in the background thread."""
    global audio_buffer, is_recording, recording_start_time
    
    if status:
        print(f"[Audio Thread] Status Warning: {status}")

    volume_rms = np.sqrt(np.mean(indata**2))
    
    # Rolling Buffer Management
    audio_buffer = np.roll(audio_buffer, -frames)
    audio_buffer[-frames:] = indata.squeeze()

    # Threshold Check
    if volume_rms > config.AUDIO_THRESHOLD and not is_recording:
        is_recording = True
        recording_start_time = time.time()
            
    # End-of-Event Check (4 seconds after the first trigger)
    if is_recording and (time.time() - recording_start_time) >= config.RECORD_DURATION_SEC:
        # We have a full 4-second buffer, send it for classification
        classify_audio(audio_buffer)
        is_recording = False

def classify_audio(audio_buffer_data):
    """Runs inference and places the result in the shared queue."""
    global sound_prediction_queue # This queue is passed from the main thread
    
    crnn_input = convert_to_crnn_input(audio_buffer_data)
    
    if model is None:
        print("[Audio Thread] Error: Sound model is not loaded.")
        return

    raw_prediction = model.predict(crnn_input, verbose=0)
    
    confidence = np.max(raw_prediction)
    predicted_index = np.argmax(raw_prediction)
    predicted_class = class_names[predicted_index]
    
    if confidence > config.SOUND_CONFIDENCE_THRESHOLD:
        # Put the valid prediction into the queue
        sound_prediction_queue.put({
            "class": predicted_class,
            "confidence": confidence
        })
    else:
        # Optional: Log ignored low-confidence sounds
        if predicted_class != config.NOISE_CLASS_NAME:
            print(f"[{time.strftime('%H:%M:%S')}] ...Ignoring low confidence sound ({predicted_class}: {confidence*100:.2f}%)...")


def start_audio_listener(queue):
    """Main function to be called by the main thread."""
    global sound_prediction_queue
    sound_prediction_queue = queue
    
    setup_sound_model()
    print("[Audio Thread] Listening for audio triggers...")
    try:
        with sd.InputStream(
            samplerate=config.FS, 
            blocksize=config.CHUNK_SIZE, 
            channels=config.CHANNELS, 
            dtype='float32', 
            callback=audio_callback
        ):
            while True:
                time.sleep(1)
    except Exception as e:
        print(f"[Audio Thread] Stream failed: {e}")