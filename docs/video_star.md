# Roteiro do vídeo — método STAR (5 minutos)

Tempo total: **4:50**, com 10 s de margem. Grave a tela em 1080p, com o terminal em fonte grande (≥ 16 pt).

## Preparação (antes de gravar)

```bash
docker compose down -v                     # MLflow zerado: a demo mostra tudo nascendo
docker compose build pipeline api          # builds já em cache, para não esperar ao vivo
```

Abas abertas: repositório no GitHub, VS Code com o projeto, terminal, `localhost:5000` (abre após o passo 3),
`localhost:8000/docs`.

---

## S — Situation (0:00 – 0:40)

**Tela:** README no GitHub (topo, tabela de pilares).

> "Um e-commerce quer saber, ainda durante a navegação, quais sessões vão terminar em compra, para agir na
> hora certa com um cupom ou um atendimento. Usamos o dataset Online Shoppers Purchasing Intention: 12.330
> sessões reais, com páginas visitadas, tempo em cada tipo de página, taxas de rejeição e de saída, mês e tipo
> de visitante. Só 15,6% das sessões compram, então as classes são desbalanceadas."

## T — Task (0:40 – 1:20)

**Tela:** seção "Como rodar" do README.

> "O desafio não era o modelo mais sofisticado, e sim a engenharia: qualquer pessoa precisa conseguir clonar o
> repositório e reproduzir o mesmo resultado. Os requisitos: Clean Code com type hints, dependências travadas
> com Poetry, dados versionados com DVC, tudo rodando em Docker, e experimentos rastreados no MLflow, com o
> melhor modelo promovido no Model Registry."

## A — Action (1:20 – 3:50)

### 1. Código (1:20 – 1:55) — VS Code

- Árvore `src/purchase_propensity/`: `data`, `features`, `models`, `pipelines`, `api`
- Abrir `models/trainer.py`: classe `ModelTrainer`, type hints, docstrings, funções curtas
- Abrir `models/factory.py`: o pré-processamento fica **dentro** do `Pipeline` do Scikit-Learn, então o mesmo
  objeto é treinado e servido

> "Usamos POO onde há estado, como o trainer e o registry, e funções puras onde não há, como limpeza, split e
> métricas. O Ruff garante docstrings e complexidade baixa, o mypy roda em modo strict e o CI valida tudo a
> cada PR."

### 2. Reprodutibilidade (1:55 – 2:15) — VS Code

- `pyproject.toml`: grupos `main` e `dev`, Python 3.12 fixado
- `params.yaml`: seed única e hiperparâmetros

> "O `poetry.lock` está commitado e os segredos ficam no `.env`. No repositório vai só o `.env.example`."

### 3. DVC + Docker ao vivo (2:15 – 3:20) — terminal

```bash
poetry run dvc dag
docker compose up -d mlflow
docker compose run --rm pipeline dvc repro -f -s preprocess train evaluate register
```

Enquanto roda, mostrar `dvc.yaml` e o `Dockerfile`:

> "O DVC versiona o dataset no DagsHub e define quatro estágios, cada um com dependências de dados, código e
> parâmetros. Se eu mudar só um hiperparâmetro, ele reexecuta só o que depende dele. O Dockerfile é
> multi-stage: o Poetry instala as dependências no builder a partir do lock, e a imagem final roda como
> usuário não-root. O mesmo arquivo gera o container do pipeline e o da API."

Ao terminar:

```bash
poetry run dvc status        # "Data and pipelines are up to date"
```

### 4. MLflow (3:20 – 3:50) — `localhost:5000`

- Experimento `purchase-propensity`: 3 runs, colunas `test_pr_auc` e `cv_pr_auc`
- Abrir a run do HistGradientBoosting: params, tags `data_version` e `git commit`, artefatos `model/` e
  `evaluation/`
- Aba **Models**: `purchase-propensity-model`, versão com alias **champion**

> "Cada run guarda parâmetros, métricas de validação cruzada e de teste, o modelo com signature e os gráficos.
> As tags ligam a run ao commit e à versão exata dos dados. O estágio register escolhe o melhor por PR-AUC e
> move o alias champion para a nova versão."

## R — Result (3:50 – 4:50)

**Tela:** `localhost:8000/docs`, executar `POST /predict` com o exemplo e depois a tabela de Resultados no README.

```bash
docker compose up -d api
```

> "O campeão foi o HistGradientBoosting: ROC-AUC de 0,93 e PR-AUC de 0,74 no teste, contra 0,16 de um modelo
> aleatório. A API carrega esse modelo e responde a probabilidade de compra de cada sessão. Rodamos o
> pipeline no Windows e no container Linux e as métricas foram idênticas: a reprodutibilidade está
> garantida pela seed, pelo lock e pelo DVC."

**Lições aprendidas (falar 2 ou 3):**

- O MLflow 3 serializa modelos com `skops`, que bloqueia tipos não auditados: liberamos explicitamente só as
  estruturas das árvores, em vez de voltar para pickle.
- Finais de linha diferentes entre Windows e Linux mudavam o hash dos arquivos no DVC; fixamos LF.
- Com classes desbalanceadas, acurácia engana. PR-AUC foi o critério certo de promoção.

> "O pipeline funciona de ponta a ponta: do dado versionado ao modelo em produção, com um único comando."
