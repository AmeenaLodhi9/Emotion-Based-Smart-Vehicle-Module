# https://youtu.be/NJpS-sFGLng
"""
Convert keras trained h5 model to tflite for deployment onto edge devices. 

"""

import tensorflow as tf
import keras
from keras.models import load_model

emotion_model = load_model('trained model/emotion_detection_model.h5', compile=False)


converter = tf.lite.TFLiteConverter.from_keras_model(emotion_model)
#converter.optimizations = [tf.lite.Optimize.DEFAULT] #Uses default optimization strategy to reduce the model size
tflite_model = converter.convert()
open("emotion_detection_model_100epochs_no_opt.tflite", "wb").write(tflite_model)

