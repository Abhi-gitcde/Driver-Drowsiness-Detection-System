import cv2
import numpy as np
import time
import pygame
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# Initialize pygame for alarm sound
pygame.mixer.init()
pygame.mixer.music.load("C:/Users/abhir/Downloads/alarm.wav")  # You'll need to add an alarm sound file

# Load pre-trained models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# For closed eyes detection, we'll use a CNN model
# You'll need to download or train this model separately
# Here we're assuming it exists
try:
    eye_status_model = load_model('eye_status_model.h5')
    use_deep_learning = True
    print("Deep learning model for eye status loaded successfully")
except:
    use_deep_learning = False
    print("Using traditional CV methods for eye detection")

# Constants
EYE_AR_THRESH = 0.3  # Eye aspect ratio threshold
EYE_AR_CONSEC_FRAMES = 10  # Number of consecutive frames for drowsiness
COUNTER = 0
ALARM_ON = False

def detect_drowsiness_cv(eyes, frame_gray):
    """Detect drowsiness using traditional CV methods"""
    if len(eyes) == 0:
        return True  # No eyes detected, might be closed or looking away
    
    # Check if eyes are open by looking at the eye regions
    for (ex, ey, ew, eh) in eyes:
        eye_roi = frame_gray[ey:ey+eh, ex:ex+ew]
        
        # Apply threshold to find the iris
        _, threshold = cv2.threshold(eye_roi, 70, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # If no contours found or very small, eyes might be closed
        if len(contours) == 0 or cv2.contourArea(max(contours, key=cv2.contourArea)) < 10:
            return True
    
    return False

def detect_drowsiness_dl(eyes, frame):
    """Detect drowsiness using the deep learning model"""
    eyes_closed = 0
    
    for (ex, ey, ew, eh) in eyes:
        eye_roi = frame[ey:ey+eh, ex:ex+ew]
        if eye_roi.size == 0:
            continue
            
        eye_roi = cv2.resize(eye_roi, (64, 64))
        eye_roi = img_to_array(eye_roi)
        eye_roi = preprocess_input(eye_roi)
        eye_roi = np.expand_dims(eye_roi, axis=0)
        
        pred = eye_status_model.predict(eye_roi)
        if pred[0][0] > 0.5:  # Assuming model output: 0=open, 1=closed
            eyes_closed += 1
    
    # If all detected eyes are closed
    return eyes_closed == len(eyes) and len(eyes) > 0

def main():
    global COUNTER, ALARM_ON
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    print("Driver drowsiness detection started. Press 'q' to quit.")
    
    start_time = time.time()
    frame_count = 0
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break
            
        frame_count += 1
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Process each face
        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Region of interest for eyes
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            
            # Detect eyes
            eyes = eye_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(20, 20)
            )
            
            # Draw rectangles around eyes
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)
            
            # Detect drowsiness
            is_drowsy = False
            if use_deep_learning:
                is_drowsy = detect_drowsiness_dl(eyes, roi_color)
            else:
                is_drowsy = detect_drowsiness_cv(eyes, roi_gray)
            
            if is_drowsy:
                COUNTER += 1
                # If eyes closed for sufficient number of frames, alert
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    # If alarm not on, turn it on
                    if not ALARM_ON:
                        ALARM_ON = True
                        if pygame.mixer.music.get_busy() == 0:
                            pygame.mixer.music.play(-1)  # Play in loop
                    
                    # Add warning text
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                COUNTER = 0
                # If alarm is on, turn it off
                if ALARM_ON:
                    ALARM_ON = False
                    pygame.mixer.music.stop()
        
        # Calculate and display FPS
        if frame_count % 10 == 0:
            end_time = time.time()
            fps = 10 / (end_time - start_time)
            start_time = end_time
            cv2.putText(frame, f"FPS: {fps:.2f}", (frame.shape[1] - 120, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display the resulting frame
        cv2.imshow('Driver Drowsiness Detection', frame)
        
        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.music.stop()

if __name__ == "__main__":
    main()
