import tensorflow as tf
import numpy as np
model = tf.keras.models.load_model('models/cnn_final.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
open('models/cnn_quant.tflite','wb').write(tflite_model)
print('Saved TFLite model to models/cnn_quant.tflite')
