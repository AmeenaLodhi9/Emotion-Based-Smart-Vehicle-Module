from tensorflow.keras.preprocessing.image import img_to_array
import cv2
from tflite_runtime.interpreter import Interpreter
import numpy as np
import time

# Load the TFLite model and allocate tensors for emotion detection
emotion_interpreter = Interpreter(model_path="trained model/emotion_detection_model_100epochs_no_opt.tflite")
emotion_interpreter.allocate_tensors()

# Get input and output details for emotion detection
emotion_input_details = emotion_interpreter.get_input_details()
emotion_output_details = emotion_interpreter.get_output_details()

# Load the TFLite model and allocate tensors for drowsiness detection
drowsiness_interpreter = Interpreter(model_path="trained model/drowsiness_detection_model_100epochs_no_opt.tflite")
drowsiness_interpreter.allocate_tensors()

# Get input and output details for drowsiness detection
drowsiness_input_details = drowsiness_interpreter.get_input_details()
drowsiness_output_details = drowsiness_interpreter.get_output_details()

class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Load the pre-trained face cascade classifier
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

car_speed = 50  # Initial car speed

while True:
    ret, frame = cap.read()
    labels = []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    start = time.time()

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray_emotion = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        # Get image ready for emotion prediction
        roi_emotion = roi_gray_emotion.astype('float') / 255.0
        roi_emotion = img_to_array(roi_emotion)
        roi_emotion = np.expand_dims(roi_emotion, axis=0)

        # Predict emotion
        emotion_interpreter.set_tensor(emotion_input_details[0]['index'], roi_emotion)
        emotion_interpreter.invoke()
        emotion_preds = emotion_interpreter.get_tensor(emotion_output_details[0]['index'])

        emotion_label = class_labels[emotion_preds.argmax()]
        emotion_label_position = (x, y)
        speed_label_position = (x, y + h + 30)

        cv2.putText(frame, f"Emotion: {emotion_label}", emotion_label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Get image ready for drowsiness prediction
        roi_drowsiness = cv2.resize(roi_gray, (227, 227), interpolation=cv2.INTER_AREA)
        roi_drowsiness = roi_drowsiness.astype('float32') / 255.0
        roi_drowsiness = np.stack((roi_drowsiness,)*3, axis=-1)
        roi_drowsiness = np.expand_dims(roi_drowsiness, axis=0)

        # Predict drowsiness
        drowsiness_interpreter.set_tensor(drowsiness_input_details[0]['index'], roi_drowsiness)
        drowsiness_interpreter.invoke()
        drowsiness_preds = drowsiness_interpreter.get_tensor(drowsiness_output_details[0]['index'])[0][0]

        # Adjust car speed based on emotion and drowsiness
        car_speed = 50 + (emotion_preds[0][3] * 20) - (drowsiness_preds * 30)

        # Display drowsiness label
        drowsiness_label = "Drowsy" if drowsiness_preds > 0.5 else "Not Drowsy"
        label_position_drowsiness = (x, y - 20)
        cv2.putText(frame, f"Drowsiness: {drowsiness_label}", label_position_drowsiness, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Update car speed within a certain range
    car_speed = max(0, min(int(car_speed), 100))

    # Display car speed
    cv2.putText(frame, f"Car Speed: {car_speed}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    end = time.time()
    print("Total time=", end - start)

    cv2.imshow('Emotion and Drowsiness Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press q to exit
        break

cap.release()
cv2.destroyAllWindows()
