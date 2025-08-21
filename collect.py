import cv2
import mediapipe as mp
import os
 
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
 
OUT_DIR = 'data/images'
os.makedirs(OUT_DIR, exist_ok=True)
 
# Labels que você quer capturar
LABELS = ['A', 'B', 'C']
for lbl in LABELS:
    os.makedirs(os.path.join(OUT_DIR, lbl), exist_ok=True)
 
label = None
counters = {lbl: 0 for lbl in LABELS}
 
cap = cv2.VideoCapture(0)
 
with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
 
        frame = cv2.flip(frame, 1)  # Espelha
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
 
        # Desenha landmarks na tela
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
 
        # Mostra instruções
        cv2.putText(frame, "Pressione A/B/C para escolher label, ESPACO para salvar, ESC para sair",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
 
        if label:
            cv2.putText(frame, f"Label atual: {label}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
 
        cv2.imshow("Coletor de Imagens", frame)
        key = cv2.waitKey(1) & 0xFF
 
        # Seleção de label
        if key in [ord(l.lower()) for l in LABELS]:
            label = chr(key).upper()
            print(f"Label selecionado: {label}")
 
        # Salvar imagem apenas se tiver mão detectada
        elif key == ord(' '):
            if label and result.multi_hand_landmarks:
                h, w, _ = frame.shape
                hand = result.multi_hand_landmarks[0]
                xs = [lm.x * w for lm in hand.landmark]
                ys = [lm.y * h for lm in hand.landmark]
 
                x_min, x_max = int(min(xs)), int(max(xs))
                y_min, y_max = int(min(ys)), int(max(ys))
 
                # Recorte com margem
                margin = 20
                x_min = max(0, x_min - margin)
                y_min = max(0, y_min - margin)
                x_max = min(w, x_max + margin)
                y_max = min(h, y_max + margin)
 
                crop = frame[y_min:y_max, x_min:x_max]
                if crop.size > 0:
                    save_path = os.path.join(OUT_DIR, label, f"{counters[label]:04d}.jpg")
                    cv2.imwrite(save_path, crop)
                    print(f"Imagem salva: {save_path}")
                    counters[label] += 1
                else:
                    print("[ERRO] Recorte inválido. Mão não detectada corretamente.")
            else:
                print("[ERRO] Nenhuma mão detectada para salvar.")
 
        # Sair
        elif key == 27:  # ESC
            break
 
cap.release()
cv2.destroyAllWindows()