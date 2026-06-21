# Plano de Execução — Trabalho de Inteligência Computacional

## Objetivo
Criar um projeto completo de predição de diabetes tipo 2 usando técnicas de IC, empacotado em ZIP < 50MB, extremamente fácil de executar no Anaconda/Windows 11.

## Entregáveis Finais
- ZIP `diabetes-ic.zip` com tudo incluso (< 50MB)
- Código one-click (`python main.py` ou `EXECUTAR.bat`)
- Artigo científico completo (.md)
- Dataset incluso no projeto

## Estrutura do Projeto
```
diabetes-ic/
├── README.md
├── requirements.txt
├── EXECUTAR.bat
├── dataset/
│   └── diabetes.csv          # Dataset incluso (~25KB)
├── src/
│   └── main.py               # Script único completo
├── article/
│   └── artigo.md
└── results/                  # Criado automaticamente
    ├── figuras/
    └── tabelas/
```

## Stage 1 — Criação do Código (Subagent: coder_ml)
- Criar `main.py` único e completo com todo o pipeline
- Dataset embutido (fazer download automático do UCI via requests, ou gerar)
- 4 algoritmos: MLP, SVM, Random Forest, Logistic Regression
- GridSearchCV, métricas, gráficos, tabelas, relatório de resultados
- Tudo salva na pasta `results/`

## Stage 2 — Criação do Artigo (Subagent: writer_ic)
- Artigo científico completo em markdown
- Baseado nos resultados que o código produzirá
- Máximo 6 páginas, todas as seções exigidas

## Stage 3 — Empacotamento
- README.md, requirements.txt, EXECUTAR.bat
- ZIP final
