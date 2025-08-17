# Projeto de Detecção de Fraudes em Cibersegurança

Este projeto tem como objetivo desenvolver um pipeline completo de Ciência de Dados e Machine Learning para detectar intrusões e atividades fraudulentas em dados de segurança cibernética.

O projeto segue uma abordagem modular e escalável, cobrindo desde extração e tratamento de dados até avaliação de métricas de negócio e implantação do modelo otimizado.

---

# Instalação e Requisitos

```
# Clone o repositório
git clone https://github.com/seuusuario/projeto-fraude-cybersecurity.git
cd projeto-fraude-cybersecurity

# Crie um ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Instale as dependências
pip install -r requirements.txt

```

---

# Estrutura do Projeto

```
configs/          → Configurações e paths
data/             → Datasets raw, processed e transformed
logs/             → Logs de EDA e processos
models/           → Modelos treinados (.joblib)
notebooks/        → Análises gráficas e estatísticas
scripts/          → Scripts de pipeline e modelagem
src/etl/          → Funções de Extract, Transform e Load
src/utils/        → Funções utilitárias, como descrições de dataset

```

---

# Pipeline do Projeto

1️⃣ Extração e Análise Exploratória

- Funções implementadas em extract.py:
    - describe_dataset(df): Gera sumário completo (tipos, missing values, estatísticas, correlações, problemas de qualidade).
    - detect_outliers(df): Identifica outliers com Z-score e IQR, classificando severidade e sugerindo ação.
- Logs automáticos em logs/.

2️⃣ Transformação de Dados

- Script transform.py responsável por:
    - Limpeza de colunas redundantes.
    - Padronização de tipos.
    - Preparação de features para modelagem.
- Saída salva em múltiplos formatos: CSV, Parquet e banco de dados.

3️⃣ Análises Exploratórias e Estatísticas

- Notebooks em notebooks/:
    1. 01_graphic_analysis.ipynb → Análise gráfica (distribuições, relações entre variáveis, padrões visuais).
    2. 02_statistical_analysis.ipynb → Testes de hipóteses, correlações e significância estatística.
    3. 03_business_metrics.ipynb → Construção de métricas de impacto no negócio (fraudes x não fraudes).

4️⃣ Seleção e Treinamento de Modelos

- Implementado em train.py e 04_model_selection.ipynb:
    - Modelos avaliados com RandomizedSearchCV.
    - Otimização de hiperparâmetros.
    - Threshold ajustado para maximizar Recall mantendo Precision ≥ 0.7.
    - Melhor modelo exportado em .joblib.

Exemplo de resultado final do modelo otimizado:

```
Threshold: 0.20
Precision: 0.730
Recall   : 0.809
F1       : 0.767
ROC AUC  : 0.882

```

---

# Principais Métricas de Negócio

- Prioridade no Recall (Sensibilidade) → Detectar o maior número de fraudes reais possível.
- Precision mínima de 0.7 → Evitar excesso de falsos positivos.
- F1-score como equilíbrio entre Precision e Recall.
- ROC AUC para avaliar capacidade geral de separação entre classes.

---

# Exportação e Reuso do Modelo

```
import joblib

# Salvar modelo
joblib.dump(best_model, "models/fraud_detection_model.joblib")

# Carregar modelo
model = joblib.load("models/fraud_detection_model.joblib")

```

---

# Tecnologia e Bibliotecas

- Python 3.12+
- Pandas → Manipulação e análise de dados
- Joblib → Serialização e armazenamento de modelos
- Matplotlib → Visualização gráfica básica
- Seaborn → Visualizações estatísticas avançadas
- Scikit-Learn → Modelagem e avaliação de Machine Learning
- KaggleHub → Integração com datasets do Kaggle
- Plotly → Visualizações interativas
- SciPy → Cálculos científicos e estatísticos

---

# Próximos Passos / Melhorias Futuras

- Implementar alertas em tempo real para detecção de fraudes.
- Explorar modelos ensemble ou deep learning.
- Adicionar monitoramento de performance do modelo em produção.
- Expandir pipeline para múltiplos datasets de cibersegurança.