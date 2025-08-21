#!/usr/bin/env bash
set -e
echo "=== Hand Sign Project — Runner (Linux/macOS) ==="
echo "1) Criando/ativando venv (venv)..."
python -m venv venv
source venv/bin/activate
echo "2) Instalando dependências..."
pip install -r requirements.txt
echo "3) Coleta de dados: abra a câmera e gere imagens nas pastas data/images/A,B,C"
echo "   Execute: python collect.py"
echo "Pressione a/b/c para selecionar label e SPACE para salvar crops. Quando terminar, pressione q."
read -p "Pressione Enter quando quiser continuar para extrair landmarks..."
echo "4) Extraindo landmarks..."
python utils.py
echo "5) Treinar landmark MLP (rápido):"
python train_landmarks_mlp.py
echo "6) (Opcional) Treinar CNN (se tiver muitas imagens):"
echo "   python train_cnn.py"
echo "7) Testar inferência em tempo real:"
echo "   export HAND_MODE=landmark && python infer_realtime.py"
echo "Runner finalizado."
