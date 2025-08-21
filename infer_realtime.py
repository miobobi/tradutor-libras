import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from utils import extract_landmark_vector_from_image
import os

# Configurações
MODE = os.environ.get('HAND_MODE', 'landmark')  # 'landmark' ou 'cnn'
MODEL_PATH = 'models/hand_sign_mlp.h5'
conf_threshold = 0.75  # Limiar de confiança mínimo
# Classes reais do modelo (A, B, C, etc.), "None" é adicionada dinamicamente
classes_model = ['A', 'B', 'C']
classes = classes_model + ['None']

# Carrega modelo
if not os.path.exists(MODEL_PATH):
    print(f"❌ Modelo não encontrado em {MODEL_PATH}")
    exit()
model = tf.keras.models.load_model(MODEL_PATH)

# Inicializa mediapipe
mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0)

with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.6) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        lab = 'None'  # Classe padrão quando nada é detectado
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if MODE == 'landmark':
            vec = extract_landmark_vector_from_image(frame)
            if vec is not None:
                p = model.predict(vec.reshape(1, -1), verbose=0)
                conf = np.max(p)
                if conf > conf_threshold:
                    lab = classes_model[np.argmax(p)]

        else:  # CNN
            res = hands.process(frame_rgb)
            if res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0]
                h, w, _ = frame.shape
                xs = [p.x for p in lm.landmark]
                ys = [p.y for p in lm.landmark]
                x1, x2 = int(min(xs) * w) - 20, int(max(xs) * w) + 20
                y1, y2 = int(min(ys) * h) - 20, int(max(ys) * h) + 20
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                crop = frame[y1:y2, x1:x2]
                if crop.size != 0:
                    img = cv2.resize(crop, (224, 224))
                    p = model.predict(img.reshape(1, 224, 224, 3) / 255.0, verbose=0)
                    conf = np.max(p)
                    if conf > conf_threshold:
                        lab = classes_model[np.argmax(p)]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(frame, f"Letra: {lab}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.imshow('Infer', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
