\
    @echo off
    echo === Hand Sign Project - Runner (Windows) ===
    echo 1) Criando venv...
    python -m venv venv
    echo Ative o ambiente virtual com: venv\Scripts\activate
    echo 2) Instale as dependências: pip install -r requirements.txt
    echo 3) Rode coletor para gerar imagens: python collect.py
    echo    - Pressione a/b/c para selecionar label; SPACE para salvar; q para sair.
    pause
    echo 4) Extraindo landmarks...
    python utils.py
    echo 5) Treinar landmark MLP (rápido):
    python train_landmarks_mlp.py
    echo 6) (Opcional) Treinar CNN (se tiver muitas imagens):
    echo    python train_cnn.py
    echo 7) Testar inferência em tempo real:
    echo    set HAND_MODE=landmark && python infer_realtime.py
    pause
