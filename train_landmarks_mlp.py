import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import os

DATA_X = "data/X.npy"
DATA_Y = "data/y.npy"

# Verifica se o dataset existe
if not (os.path.exists(DATA_X) and os.path.exists(DATA_Y)):
    print("❌ Dataset não encontrado. Execute utils.py primeiro para gerar X.npy e y.npy")
    exit()

print("✅ Dataset encontrado.")

# Carrega dados
X = np.load(DATA_X)
y = np.load(DATA_Y)

print(f"📊 Dataset carregado: {X.shape[0]} amostras, vetor de {X.shape[1]} dimensões.")

# Define número correto de classes automaticamente
num_classes = int(np.max(y) + 1)
print(f"📊 Número de classes detectadas: {num_classes}")

# One-hot encoding para as classes
y_cat = to_categorical(y, num_classes)

# Normalização opcional (garante float32)
X = X.astype("float32")

# Define modelo MLP
model = Sequential([
    Dense(128, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Treinamento com cuidado se poucas amostras
if X.shape[0] < 10:
    print("⚠ Atenção: poucas amostras disponíveis, resultados podem não generalizar.")
history = model.fit(
    X, y_cat,
    epochs=30,
    batch_size=min(32, X.shape[0]),  # evita batch maior que dataset
    validation_split=0.2 if X.shape[0] > 5 else 0.0,  # se dataset muito pequeno, sem validação
    verbose=1
)

# Salvar modelo
os.makedirs("models", exist_ok=True)
model.save("models/hand_sign_mlp.h5")
print("✅ Modelo salvo em models/hand_sign_mlp.h5")
