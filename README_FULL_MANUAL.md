
        Principais passos resumidos:

        1) Criar ambiente virtual
2) pip install -r requirements.txt
3) rodar collect.py para coletar imagens nas pastas data/images/A, B, C
4) rodar python utils.py para gerar data/landmarks.npy
5) treinar com train_landmarks_mlp.py ou train_cnn.py
6) testar infer_realtime.py
