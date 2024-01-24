from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
from scipy.spatial import distance as dist

# Load the pre-trained emotion detection model
emotion_model = load_model('trained model/emotion_detection_model.h5')

# Load the pre-trained face cascade classifier
face_classifier = cv2.CascadeClassifier('haarcascades_models/haarcascade_frontalface_default.xml')

# Load the pre-trained drowsiness detection model
# You need to replace 'path_to_drowsiness_model' with the actual path to your drowsiness detection model
drowsiness_model = load_model('path_to_drowsiness_model.h5')

class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
gender_labels = ['Male', 'Female']

def eye_aspect_ratio(eye):
    # Compute the euclidean distances between the two sets of vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # Compute the euclidean distance between the horizontal eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    return ear

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    labels = []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        # Get image ready for emotion prediction
        roi = roi_gray.astype('float') / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        # Predict emotion
        preds = emotion_model.predict(roi)[0]
        emotion_label = class_labels[preds.argmax()]

        # Get eye region for drowsiness detection
        eyes = cv2.resize(roi_gray, (224, 224))
        eyes = eyes.astype('float') / 255.0
        eyes = img_to_array(eyes)
        eyes = np.expand_dims(eyes, axis=0)

        # Predict drowsiness
        drowsiness_preds = drowsiness_model.predict(eyes)[0]
        drowsiness_label = "Drowsy" if drowsiness_preds[0] > 0.5 else "Awake"

        # Display emotion and drowsiness labels
        label_position_emotion = (x, y - 20)
        label_position_drowsiness = (x, y - 40)
        cv2.putText(frame, f"Emotion: {emotion_label}", label_position_emotion, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f"Drowsiness: {drowsiness_label}", label_position_drowsiness, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Emotion and Drowsiness Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
