import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter

# Load the drowsiness detection TFLite model
drowsiness_interpreter = Interpreter(model_path="trained model/drowsiness_detection_model_100epochs_no_opt.tflite")
drowsiness_interpreter.allocate_tensors()

# Get input and output details
drowsiness_input_details = drowsiness_interpreter.get_input_details()
drowsiness_output_details = drowsiness_interpreter.get_output_details()

face_classifier = cv2.CascadeClassifier('haarcascades_models/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]

        # Resize the ROI to the input size expected by the model and normalize
        roi_drowsiness = cv2.resize(roi_gray, (227, 227), interpolation=cv2.INTER_AREA)
        roi_drowsiness = roi_drowsiness.astype('float32') / 255.0

        # Convert grayscale ROI to 3-channel (RGB) format
        roi_drowsiness = np.stack((roi_drowsiness,)*3, axis=-1)

        # Add the batch dimension
        roi_drowsiness = np.expand_dims(roi_drowsiness, axis=0)

        # Set tensor input for drowsiness detection
        drowsiness_interpreter.set_tensor(drowsiness_input_details[0]['index'], roi_drowsiness)
        drowsiness_interpreter.invoke()

        # Get drowsiness prediction
        drowsiness_preds = drowsiness_interpreter.get_tensor(drowsiness_output_details[0]['index'])[0][0]

        # Display drowsiness label
        label_position_drowsiness = (x, y - 20)
        cv2.putText(frame, f"Drowsiness: {drowsiness_preds}", label_position_drowsiness, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Drowsiness Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()