# Predicao de Diabetes Tipo 2 - Inteligencia Computacional

**Trabalho Final - Inteligencia Computacional**  
**CEFET-MG - 2026/1**  
**Prof. Alisson Marques da Silva**

---

## Descricao

Este projeto implementa uma analise comparativa de quatro tecnicas de Inteligencia Computacional para predicao de Diabetes Mellitus Tipo 2:

- **MLP** (Multi-Layer Perceptron / Rede Neural Artificial)
- **SVM** (Support Vector Machine / Maquina de Vetores de Suporte)
- **Random Forest** (Floresta Aleatoria)
- **Regressao Logistica** (Baseline interpretavel)

Utilizando o dataset **PIMA Indians Diabetes Database** (UCI Machine Learning Repository), o projeto executa um pipeline completo de Machine Learning incluindo pre-processamento, busca de hiperparametros via GridSearchCV, avaliacao comparativa e geracao de graficos e relatorios.

---

## Requisitos

- **Python 3.9 ou superior**
- **Anaconda** (recomendado para Windows 11)

As seguintes bibliotecas sao necessarias (todas inclusas no Anaconda):

```
numpy
pandas
matplotlib
seaborn
scikit-learn
imbalanced-learn
```

---

## Como Executar

### Metodo 1 - Script Python (Multiplataforma)

1. Abra o **Anaconda Prompt** (ou terminal com Python instalado)
2. Navegue ate a pasta do projeto:
   ```bash
   cd diabetes-ic/src
   ```
3. Execute o script:
   ```bash
   python main.py
   ```

### Metodo 2 - Duplo Clique no Windows (Mais Facil)

1. Basta dar **duplo clique** no arquivo `EXECUTAR.bat`
2. O script executa automaticamente e abre a pasta de resultados ao final

---

## Estrutura do Projeto

```
diabetes-ic/
|-- README.md                          # Este arquivo
|-- requirements.txt                   # Dependencias Python
|-- EXECUTAR.bat                       # Script one-click para Windows
|
|-- dataset/
|   |-- diabetes.csv                   # Dataset PIMA (incluso)
|
|-- src/
|   |-- main.py                        # Script principal (executa TUDO)
|
|-- article/
|   |-- artigo.md                      # Artigo cientifico completo
|
|-- results/                           # GERADO automaticamente pelo script
|   |-- figuras/                       # 7 graficos (PNG, 300 DPI)
|   |   |-- 01_correlacao.png
|   |   |-- 02_distribuicao_classes.png
|   |   |-- 03_matrizes_confusao.png
|   |   |-- 04_curvas_roc.png
|   |   |-- 05_comparacao_metricas.png
|   |   |-- 06_stability_boxplot.png
|   |   |-- 07_feature_importance.png
|   |
|   |-- tabelas/                       # 3 relatorios TXT
|       |-- resultados_comparativos.txt
|       |-- melhores_parametros.txt
|       |-- relatorio_completo.txt
```

---

## Fluxo de Execucao

O script `main.py` executa automaticamente as seguintes etapas:

1. **Carregamento dos dados** - Le o arquivo CSV (ou faz download automatico)
2. **Analise exploratoria** - Estatisticas descritivas, zeros impossiveis, desbalanceamento
3. **Pre-processamento** - Imputacao, divisao treino/val/teste (60/20/20), StandardScaler, SMOTE
4. **Treinamento** - GridSearchCV 5-fold para cada um dos 4 algoritmos
5. **Repeticoes** - 10 execucoes para modelos estocasticos (MLP e RF)
6. **Avaliacao** - Calculo de Acuracia, Precisao, Recall, F1-Score, AUC-ROC no teste
7. **Graficos** - Geracao de 7 figuras em alta resolucao (300 DPI)
8. **Relatorio** - Exportacao de 3 arquivos TXT com resultados detalhados

---

## Resultados Esperados

O script completa em aproximadamente **1-2 minutos** e gera:

- **7 graficos** na pasta `results/figuras/`
- **3 relatorios** na pasta `results/tabelas/`
- Prints detalhados no console com todas as metricas

---

## Instalacao de Dependencias (se necessario)

Se alguma biblioteca estiver faltando, execute:

```bash
pip install -r requirements.txt
```

Ou instale manualmente:

```bash
conda install numpy pandas matplotlib seaborn scikit-learn
conda install -c conda-forge imbalanced-learn
```

---

## Tecnologias Utilizadas

| Tecnologia        | Versao Minima | Finalidade                      |
|-------------------|---------------|---------------------------------|
| Python            | 3.9+          | Linguagem principal             |
| NumPy             | 1.21+         | Computacao numerica             |
| Pandas            | 1.3+          | Manipulacao de dados            |
| Matplotlib        | 3.4+          | Visualizacao de dados           |
| Seaborn           | 0.11+         | Graficos estatisticos           |
| Scikit-learn      | 1.0+          | Algoritmos de ML                |
| Imbalanced-learn  | 0.9+          | Balanceamento de classes (SMOTE)|

---

## Dataset

**PIMA Indians Diabetes Database**
- Fonte: UCI Machine Learning Repository
- Amostras: 768
- Atributos: 8 numericos
- Alvo: Binario (0 = sem diabetes, 1 = com diabetes)
- Uso: Pesquisa academica (dominio publico)

---

## Autores

- Rafael A. S. Ferreira
- Joao M. G. Lisboa
- Gabriel V. Silva

**Instituicao:** CEFET-MG  
**Disciplina:** Inteligencia Computacional  
**Professor:** Alisson Marques da Silva

---

## Artigo

O artigo cientifico completo em formato LaTeX (modelo IEEEtran) esta disponivel em `artigo/artigo_diabetes.tex`, na raiz do projeto. O PDF compilado `artigo_diabetes.pdf` (dentro desta pasta) contem o artigo final com as figuras e tabelas geradas automaticamente pelo pipeline.

## Apresentacao (Slides)

Os slides da apresentacao final estao disponiveis em `artigo/slides/slides_diabetes.tex`, na raiz do projeto. O PDF compilado `slides_diabetes.pdf` (dentro desta pasta) contem 15 slides com os principais pontos do trabalho.

## Licenca

Este projeto e de uso academico para a disciplina de Inteligencia Computacional do CEFET-MG.
