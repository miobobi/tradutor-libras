import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import os

# evaluate landmarks model
model = tf.keras.models.load_model('models/landmark_mlp.h5')
raw = np.load('data/landmarks.npy', allow_pickle=True)
X = np.stack([r[0] for r in raw], axis=0)
y = np.array([r[1] for r in raw])

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, stratify=y, random_state=42)
pred = np.argmax(model.predict(X_test), axis=1)
print(classification_report(y_test, pred, target_names=['A','B','C']))
print(confusion_matrix(y_test, pred))
