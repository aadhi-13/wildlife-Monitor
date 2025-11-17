import cv2
import time
from ultralytics import YOLO
import numpy as np
import os
import threading
import queue

# Import our custom modules
import config # Your new config file
from src.classification_keras import classify_animal, setup_image_model
from src.sound_processor import start_audio_listener, play_deterrent

# --- Setup ---
os.makedirs(config.SNAPSHOT_DIR, exist_ok=True)
setup_image_model() # Loads the Keras image model

# Thread-safe queue to get results from the sound thread
sound_prediction_queue = queue.Queue()

# Start the audio listener thread
audio_thread = threading.Thread(target=start_audio_listener, args=(sound_prediction_queue,), daemon=True)
audio_thread.start()

def fuse_predictions(image_pred, sound_pred):
    """
    Applies your custom logic rules.
    Logic 1: If different, image model has priority.
    Logic 2: If one fails, use the other.
    """
    
    # Check for "failure" cases (noise or low confidence)
    img_failed = image_pred in ["cannot detect the animal properly", "Classification Error"]
    sound_failed = sound_pred in [config.NOISE_CLASS_NAME, "None"]

    # --- Logic 2: If one fails, use the other ---
    if img_failed and not sound_failed:
        return sound_pred # Image failed, use sound
    
    if not img_failed and sound_failed:
        return image_pred # Sound failed, use image

    # --- Logic 1: If both detected but different, image has priority ---
    if not img_failed and not sound_failed:
        return image_pred # Both valid, image wins

    # If both failed
    return "Unclassified"

def process_video_stream(source: str | int, model_name: str):
    """
    Main pipeline for multi-modal (Video + Audio) detection.
    """
    print("-" * 50)
    print(f"[Video Thread] Loading YOLO Detection Model: {model_name}")
    yolo_model = YOLO(model_name)
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[Video Thread] Error: Could not open video source at {source}")
        return

    frame_count = 0
    p_time = time.time()
    
    # This variable holds the latest sound prediction
    latest_sound_prediction = "None"
    
    print("[Video Thread] Starting real-time classification pipeline. Press 'q' to exit.")
    print("-" * 50)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1
        
        # --- 1. Check for new Sound Detections ---
        # Check the queue for new results from the audio thread (non-blocking)
        try:
            sound_result = sound_prediction_queue.get_nowait()
            latest_sound_prediction = sound_result["class"]
            print(f"[Audio Thread] -> DETECTED: {latest_sound_prediction} ({sound_result['confidence']:.2f})")
        except queue.Empty:
            pass # No new sound, just keep processing video

        # --- 2. Process Video Frame (YOLO) ---
        results = yolo_model(frame, verbose=False)
        
        # Default label
        final_label_text = "Unclassified"

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = box.conf[0].cpu().numpy()
                
                if conf > 0.5:
                    cropped_region = frame[y1:y2, x1:x2]
                    
                    if cropped_region.size > 0:
                        # --- 3. Classify Image Crop ---
                        image_prediction = classify_animal(cropped_region)
                        
                        # --- 4. APPLY FUSION LOGIC ---
                        fused_prediction = fuse_predictions(image_prediction, latest_sound_prediction)
                        
                        # Reset sound prediction after using it
                        latest_sound_prediction = "None"
                        
                        # Log the final decision
                        final_label_text = f"{fused_prediction} (V:{image_prediction} | A:{latest_sound_prediction})"
                        print(f"Frame {frame_count}: Fused Prediction: {fused_prediction}")

                        # --- 5. Trigger Deterrent if Threat ---
                        if fused_prediction in config.THREAT_CLASSES:
                            play_deterrent()
                        
                        # --- 6. Draw Bounding Box ---
                        color = (0, 255, 0)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        (text_width, text_height), _ = cv2.getTextSize(final_label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                        cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width + 10, y1), color, cv2.FILLED)
                        cv2.putText(frame, final_label_text, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

        # Calculate and display FPS
        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("Multi-Modal Wildlife Monitor (Press 'q' to exit)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\nVideo processing finished. Resources released.")

if __name__ == "__main__":
    process_video_stream(source=0, model_name=config.YOLO_MODEL) # source=0 for webcam