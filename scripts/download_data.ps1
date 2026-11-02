# Baixa o dataset original do Kaggle para data/raw (equivalente ao download_data.sh).
# Só é necessário para recriar o dado do zero: normalmente basta `dvc pull`.
$ErrorActionPreference = "Stop"

$Dataset = "henrysue/online-shoppers-intention"
$TargetDir = "data/raw"

New-Item -ItemType Directory -Force $TargetDir | Out-Null
kaggle datasets download -d $Dataset -p $TargetDir --unzip
Write-Output "Dataset salvo em $TargetDir. Versione com: dvc add $TargetDir/online_shoppers_intention.csv"
