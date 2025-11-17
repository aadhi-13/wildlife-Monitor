import cv2
import numpy as np
import os
# Use the full tensorflow.keras API
from tensorflow import keras

# --- 1. CONFIGURATION ---
MODEL_PATH = '/home/aadhi/yolo-animal-classifier/models/final_best_fine_tuned.keras'
ANIMAL_CLASSES = [
    "Great_Hornbill",
    "Indian_Elephant",
    "King_Cobra",
    "Leopard",
    "Monitor_Lizard",
    "Nilgiri_Langur",
    "Sloth_Bear",
    "Spotted_Deer_Chital",
    "Wild_Boar"
]
TARGET_CLASSIFICATION_SIZE = (224, 224) 
CONFIDENCE_THRESHOLD = 0.75

# --- 2. GLOBAL MODEL VARIABLE ---
# This will be loaded by the setup function
GLOBAL_CLASSIFICATION_MODEL = None

# --- 3. FIX: CREATE THE MISSING SETUP FUNCTION ---
def setup_image_model():
    """
    Loads the Keras model into the global variable.
    This function is called by main_pipeline.py at startup.
    """
    global GLOBAL_CLASSIFICATION_MODEL
    
    if GLOBAL_CLASSIFICATION_MODEL is not None:
        print("Image model already loaded.")
        return

    try:
        GLOBAL_CLASSIFICATION_MODEL = keras.models.load_model(MODEL_PATH, compile=False)
        print(f"Successfully loaded Keras model from {MODEL_PATH}")
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load .keras model. {e}")
        print("Please check the MODEL_PATH and ensure TensorFlow is installed.")

# --- 4. CLASSIFICATION FUNCTION (With fixes from last time) ---
def classify_animal(cropped_image: np.ndarray) -> str:
    """
    Handles pre-processing and inference for your .keras model.
    """
    
    if GLOBAL_CLASSIFICATION_MODEL is None:
        return "Error: Model not loaded"
    
    try:
        # 1. Convert BGR (OpenCV) to RGB (TensorFlow)
        input_img_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        
        # 2. Resize to the model's expected input
        input_img_resized = cv2.resize(input_img_rgb, TARGET_CLASSIFICATION_SIZE)
        
        # 3. Convert to the tensor format (float32, [0, 255] range)
        input_data = input_img_resized.astype('float32')
        
        # 4. Add batch dimension
        input_tensor = np.expand_dims(input_data, axis=0) 

        # 5. Run model inference
        predictions = GLOBAL_CLASSIFICATION_MODEL.predict(input_tensor, verbose=0)
        scores = predictions[0]
        
        confidence = np.max(scores)
        predicted_index = np.argmax(scores)
        
        # 6. Apply filtering logic
        if confidence > CONFIDENCE_THRESHOLD:
            predicted_class = ANIMAL_CLASSES[predicted_index]
            return predicted_class
        else:
            return "cannot detect the animal properly"
        
    except Exception as e:
        print(f"Error during Keras classification: {e}")
        return "Classification Error"