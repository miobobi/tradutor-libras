# 🤟 Tradutor de Libras com IA

Este projeto utiliza **Python + TensorFlow + MediaPipe + OpenCV** para reconhecer sinais de Libras a partir da câmera em tempo real.  
A ideia é treinar modelos de IA (MLP com landmarks e CNN com imagens) e usá-los para prever letras do alfabeto em Libras.  

Feito para a **Feira de Ciências 2025** 🧪✨

---

## 📂 Estrutura do Projeto

```
hand_sign_project_with_runner/
│── data/                # Onde ficam os datasets (X.npy e y.npy)
│── debug/               # Imagens de debug geradas durante a coleta
│── models/              # Modelos treinados (.h5)
│── utils.py             # Funções de processamento (extração de landmarks, etc.)
│── train_landmarks_mlp.py  # Script para treinar modelo baseado em landmarks
│── infer_realtime.py    # Script para rodar a câmera em tempo real
│── requirements.txt     # Lista de dependências do projeto
│── README.md            # Este arquivo 😃
```

---

## ⚙️ Instalação

Clone este repositório e entre na pasta:

```bash
git clone https://github.com/seu-usuario/hand_sign_project_with_runner.git
cd hand_sign_project_with_runner
```

Crie um ambiente virtual (recomendado):

```bash
python -m venv venv
```

Ative o ambiente virtual:

- **Windows (PowerShell)**  
  ```bash
  venv\Scripts\Activate.ps1
  ```

- **Linux/Mac**  
  ```bash
  source venv/bin/activate
  ```

Instale as dependências:

```bash
pip install -r requirements.txt
```

---

## 🧠 Treinando o Modelo

Se ainda não tiver o dataset (`X.npy` e `y.npy`):

```bash
python utils.py
```

Isso gera os arquivos em `data/`.

Depois, treine o modelo MLP com:

```bash
python train_landmarks_mlp.py
```

O modelo treinado será salvo em:

```
models/hand_sign_mlp.h5
```

---

## 📸 Executando em Tempo Real

Para abrir a câmera e começar a prever sinais:

```bash
python infer_realtime.py
```

- Pressione **Q** para sair.  
- O texto no canto superior esquerdo mostrará a letra reconhecida.  

---

## 🛠 Tecnologias Utilizadas

- [Python 3.11+](https://www.python.org/)  
- [TensorFlow/Keras](https://www.tensorflow.org/) → treino e inferência do modelo  
- [MediaPipe](https://developers.google.com/mediapipe) → detecção dos landmarks da mão  
- [OpenCV](https://opencv.org/) → captura da câmera e visualização  
- [NumPy](https://numpy.org/) → manipulação dos dados  

---

## 📌 Próximos Passos

- Aumentar o dataset (mais imagens para melhorar a acurácia)  
- Adicionar novas letras e sinais da Libras  
- Criar uma interface mais amigável (GUI/Web)  
- Publicar em nuvem ou rodar em dispositivo móvel  

---

## 👨‍💻 Autores

- Projeto desenvolvido por **Bob/Giovanna Adriano de Carvalho** para a Bentotec Feira de Ciências 2025.  

---
