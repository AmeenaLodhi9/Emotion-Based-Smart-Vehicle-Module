import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter

# Load the TFLite model and allocate tensors for emotion detection.
try:
    emotion_interpreter = Interpreter(model_path="emotion_detection_model_100epochs_opt.tflite")
    emotion_interpreter.allocate_tensors()
except Exception as e:
    print(f"Error loading emotion model: {e}")
    exit(1)

# Get input and output tensors for emotion detection.
emotion_input_details = emotion_interpreter.get_input_details()
emotion_output_details = emotion_interpreter.get_output_details()
emotion_input_shape = emotion_input_details[0]['shape']

# Load the Haarcascades face classifier.
face_classifier = cv2.CascadeClassifier('haarcascades_models/haarcascade_frontalface_default.xml')

# Class labels for emotions.
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Open a video capture object.
cap = cv2.VideoCapture(0)

while True:
    try:
        # Read a frame from the video capture.
        ret, frame = cap.read()

        # Convert the frame to grayscale for face detection.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame.
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # Extract the region of interest (ROI) for emotion detection.
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (emotion_input_shape[1], emotion_input_shape[2]), interpolation=cv2.INTER_AREA)

            # Preprocess the ROI for emotion detection.
            roi = roi_gray.astype('float') / 255.0
            roi = np.expand_dims(roi, axis=0)

            # Set the input tensor for emotion detection and invoke the interpreter.
            emotion_interpreter.set_tensor(emotion_input_details[0]['index'], roi)
            emotion_interpreter.invoke()

            # Get the output tensor and determine the predicted emotion.
            emotion_preds = emotion_interpreter.get_tensor(emotion_output_details[0]['index'])
            emotion_label = class_labels[emotion_preds.argmax()]

            # Draw a rectangle around the detected face and display the predicted emotion.
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            emotion_label_position = (x, y)
            cv2.putText(frame, emotion_label, emotion_label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame with emotion detection.
        cv2.imshow('Emotion Detector', frame)

        # Break the loop if 'q' key is pressed.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(f"Error in main loop: {e}")
        break

# Release resources.
cap.release()
cv2.destroyAllWindows()
