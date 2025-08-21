import mediapipe as mp
import numpy as np
import os
import cv2

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

LABEL_MAP = {'A': 0, 'B': 1, 'C': 2}  # Ajuste suas classes aqui

def save_debug_landmarks(img, landmarks, out_path):
    """Desenha landmarks detectados na imagem e salva em debug/."""
    debug_img = img.copy()
    mp_drawing.draw_landmarks(
        debug_img,
        landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, debug_img)

def extract_landmark_vector_from_image(img, debug_path=None):
    """Extrai vetores de landmarks normalizados (XY) a partir de uma imagem."""
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.1  # mais tolerante
    ) as hands:
        # Pré-processamento: aumenta brilho e contraste
        img = cv2.convertScaleAbs(img, alpha=1.5, beta=30)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        if not res.multi_hand_landmarks:
            return None
        
        lm = res.multi_hand_landmarks[0]
        
        # Se pediram debug, salvar imagem com landmarks desenhados
        if debug_path is not None:
            save_debug_landmarks(img, lm, debug_path)

        h, w, _ = img.shape
        xs = np.array([p.x for p in lm.landmark]) * w
        ys = np.array([p.y for p in lm.landmark]) * h
        
        # Centro e tamanho da mão
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0
        bw = max(x_max - x_min, 1)
        bh = max(y_max - y_min, 1)
        
        vec = []
        for p in lm.landmark:
            vx = (p.x * w - cx) / bw
            vy = (p.y * h - cy) / bh
            vec.extend([vx, vy])
        
        return np.array(vec, dtype=np.float32) if len(vec) == 42 else None

def build_landmark_dataset(image_root='data/images', out_dir='data', debug_dir='debug'):
    """Cria um dataset de landmarks a partir das imagens organizadas por classe."""
    X, y = [], []
    total_imgs = 0
    processed_imgs = 0

    for label_name, label_idx in LABEL_MAP.items():
        folder = os.path.join(image_root, label_name)
        if not os.path.exists(folder):
            print(f"⚠ Pasta {folder} não encontrada, pulando...")
            continue
        
        files = os.listdir(folder)
        total_imgs += len(files)

        for fname in files:
            path = os.path.join(folder, fname)
            img = cv2.imread(path)
            if img is None:
                print(f"⚠ Imagem {path} inválida, pulando...")
                continue

            debug_path = os.path.join(debug_dir, label_name, fname)  # salvar debug se detectar

            vec = extract_landmark_vector_from_image(img, debug_path=debug_path)
            if vec is not None:
                X.append(vec)
                y.append(label_idx)
                processed_imgs += 1
                print(f"✅ OK: {path}")
            else:
                print(f"❌ Falha: nenhuma mão detectada em {path}")

    if X and y:
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int64)
        
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, 'X.npy'), X)
        np.save(os.path.join(out_dir, 'y.npy'), y)
        
        print(f"\n✅ Dataset salvo em {out_dir}/X.npy e {out_dir}/y.npy")
        print(f"📊 {processed_imgs}/{total_imgs} imagens processadas com sucesso.")
        print(f"🖼 Imagens de debug salvas em {debug_dir}/")
    else:
        print("❌ Nenhum landmark foi extraído.")

if __name__ == '__main__':
    # 🔎 Debug em apenas uma imagem para testar primeiro
    test_img_path = "data/images/A/seu_arquivo.jpg"
    if os.path.exists(test_img_path):
        img = cv2.imread(test_img_path)
        vec = extract_landmark_vector_from_image(img, debug_path="debug/test_debug.jpg")
        print("\n--- TESTE EM UMA IMAGEM ---")
        print("Shape do vetor:", vec.shape if vec is not None else "Nenhuma mão detectada")
        if vec is not None:
            print("Imagem de debug salva em debug/test_debug.jpg")
    
    # Depois processa todas
    build_landmark_dataset()
