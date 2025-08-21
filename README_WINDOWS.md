# Instruções específicas para Windows e uso de GPU (resumo)


## 1) Drivers e CUDA (apenas se quiser usar GPU para TensorFlow)

- Verifique se sua GPU NVIDIA é compatível com a versão do CUDA requerida pela versão do TensorFlow.
- Instale drivers NVIDIA recentes.
- Instale CUDA Toolkit (ex.: 11.x) e cuDNN correspondentes.
- Recomenda-se seguir a documentação oficial do TensorFlow para compatibilidade: https://www.tensorflow.org/install/gpu

## 2) Instalar TensorFlow com suporte a GPU (opcional)

- No ambiente virtual, instale:
  pip install tensorflow  # versões recentes já detectam GPU se drivers estiverem corretos

## 3) Executando no Windows

- Crie e ative o venv:
  python -m venv venv
  venv\Scripts\activate
- Instale dependências:
  pip install -r requirements.txt
- Rode o coletor:
  python collect.py
- Extraia landmarks:
  python utils.py
- Treine MLP:
  python train_landmarks_mlp.py
- Teste inferência:
  set HAND_MODE=landmark && python infer_realtime.py

## 4) Problemas comuns

- OpenCV não abre câmera: verifique se outros apps não estão usando a câmera e se drivers estão OK.
- MediaPipe falha: atualize a biblioteca ou ajuste min_detection_confidence.
- Erros de memória ao treinar CNN: reduza batch size para 8 ou 16.

