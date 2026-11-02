#!/usr/bin/env sh
# Baixa o dataset original do Kaggle para data/raw.
# Só é necessário para recriar o dado do zero: normalmente basta `dvc pull`.
# Requer credenciais do Kaggle em ~/.kaggle (https://www.kaggle.com/settings).
set -eu

DATASET="henrysue/online-shoppers-intention"
TARGET_DIR="data/raw"

mkdir -p "$TARGET_DIR"
kaggle datasets download -d "$DATASET" -p "$TARGET_DIR" --unzip
echo "Dataset salvo em $TARGET_DIR. Versione com: dvc add $TARGET_DIR/online_shoppers_intention.csv"
