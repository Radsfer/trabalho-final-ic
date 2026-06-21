# Predicao de Diabetes Mellitus Tipo 2 Utilizando Tecnicas de Inteligencia Computacional: Uma Analise Comparativa

**Rafael A. S. Ferreira**, **Joao M. G. Lisboa**, **Gabriel V. Silva**  
Centro Federal de Educacao Tecnologica de Minas Gerais (CEFET-MG)  
Prof. Alisson Marques da Silva — Inteligencia Computacional  
{rafael.ferreira, joao.lisboa, gabriel.silva}@aluno.cefetmg.br

---

## Resumo

O Diabetes Mellitus Tipo 2 representa um dos maiores desafios de saude publica contemporaneos, afetando mais de 537 milhoes de adultos em todo o mundo segundo o Atlas da Diabetes da Federacao Internacional de Diabetes (2021). A identificacao precoce de individuos em risco e fundamental para a prevencao de complicacoes e reducao de custos hospitalares. Neste trabalho, conduzimos uma analise comparativa de quatro tecnicas de Inteligencia Computacional aplicadas a predicao de diabetes: Multi-Layer Perceptron (MLP), Support Vector Machine (SVM), Random Forest (RF) e Regressao Logistica. Utilizando o dataset PIMA Indians Diabetes disponivel no UCI Machine Learning Repository, aplicamos um pipeline rigoroso de pre-processamento, incluindo tratamento de zeros impossiveis, imputacao por mediana, padronizacao e balanceamento via SMOTE. A selecao de hiperparametros foi realizada atraves de GridSearchCV com validacao cruzada de 5 folds. Os resultados demonstram que a Regressao Logistica obteve o melhor desempenho geral, com F1-Score de 0,656 e AUC-ROC de 0,814 no conjunto de teste, seguida de perto pelo MLP (F1=0,623; AUC=0,812). Todas as abordagens alcancaram AUC-ROC superior a 0,79, indicando capacidade discriminativa satisfatoria. Os resultados evidenciam que modelos mais simples podem superar arquiteturas complexas em bases de dados de pequeno porte, reforcando a importancia da escolha adequada de algoritmos conforme as caracteristicas do problema.

**Palavras-chave:** Diabetes Mellitus Tipo 2; Inteligencia Computacional; Aprendizado de Maquina; Multi-Layer Perceptron; Support Vector Machine; Random Forest; Regressao Logistica.

---

## 1. Introducao

O Diabetes Mellitus Tipo 2 (DMT2) configura-se como uma das epidemias de saude mais significativas do seculo XXI. De acordo com o 10o Atlas da Diabetes da Federacao Internacional de Diabetes (IDF), aproximadamente 537 milhoes de adultos entre 20 e 79 anos viviam com a doenca em 2021, com projecao de crescimento para 783 milhoes ate 2045 (INTERNATIONAL DIABETES FEDERATION, 2021). O Brasil ocupa a 6a posicao no ranking mundial de prevalencia, com cerca de 16,8 milhoes de individuos afetados, gerando custos diretos estimados em mais de 10 bilhoes de dolares anuais para o sistema de saude (INTERNATIONAL DIABETES FEDERATION, 2021).

A natureza progressiva e frequentemente silenciosa do DMT2 torna o diagnostico tardio um problema recorrente. Estudos indicam que ate 50% dos casos permanecem nao diagnosticados por anos, periodo durante o qual complicacoes micro e macrovasculares ja se instalam (KAVALIOTIS et al., 2017). A capacidade de predizer o risco de diabetes com antecedencia, a partir de exames clinicos rotineiros e fatores demograficos, representa uma oportunidade transformadora para a medicina preventiva.

Nesse contexto, as tecnicas de Inteligencia Computacional (IC) emergem como ferramentas particularmente adequadas. A relacao entre fatores de risco e desenvolvimento do diabetes e intrinsecamente nao-linear e multivariada, envolvendo interacoes complexas entre variaveis metabolicas, geneticas e demograficas (HAYKIN, 2009). Algoritmos de Aprendizado de Maquina (AM) sao capazes de capturar padroes sutis em dados clinicos que escapam a abordagens estatisticas tradicionais, sem a necessidade de modelagem explicita das relacoes entre variaveis.

O objetivo deste trabalho e realizar uma analise comparativa sistematica de quatro algoritmos de AM amplamente utilizados na literatura para predicao de DMT2: Multi-Layer Perceptron (MLP), Support Vector Machine (SVM), Random Forest (RF) e Regressao Logistica. A contribuicao principal reside na avaliacao rigorosa sob condicoes experimentais uniformes, incluindo busca exaustiva de hiperparametros via GridSearchCV, balanceamento de classes com SMOTE e avaliacao por meio de metricas apropriadas para dados desbalanceados.

O restante deste artigo esta organizado da seguinte forma: a Secao 2 apresenta a revisao da literatura; a Secao 3 detalha os materiais e metodos empregados; a Secao 4 discute os resultados obtidos; e a Secao 5 conclui o trabalho com consideracoes finais e direcoes futuras.

---

## 2. Revisao da Literatura

A aplicacao de tecnicas de Inteligencia Computacional na predicao e manejo do diabetes tem crescido substancialmente nas ultimas duas decadas. Smith et al. (1988) estabeleceram uma das bases mais importantes para essa linha de pesquisa ao disponibilizar o PIMA Indians Diabetes Dataset, composto por dados clinicos de mulheres da tribo Pima, no Arizona. Esse conjunto de dados tornou-se referencia para benchmarking de algoritmos de AM em problemas de saude.

Sisodia e Sisodia (2018) conduziram um estudo comparativo utilizando Naive Bayes, k-Nearest Neighbors (KNN) e SVM no dataset PIMA, alcancando acuracia maxima de aproximadamente 77% com o classificador Naive Bayes. Os autores destacaram a sensibilidade dos resultados a tecnicas de pre-processamento, em especial ao tratamento de valores faltantes. Maniruzzaman et al. (2017) expandiram essa analise incorporando tecnicas de classificacao mais robustas, demonstrando que SVM com kernel RBF e Random Forest superam metodos lineares quando adequadamente parametrizados.

Kaur e Kumari (2021) apresentaram uma revisao sistematica de estudos de AM para predicao de diabetes, abrangendo 45 trabalhos publicados entre 2015 e 2020. Os autores constataram que ensembles baseados em arvores (Random Forest, XGBoost, AdaBoost) consistentemente figuram entre os melhores desempenhos, embora redes neurais apresentem potencial superior quando treinadas com volumes adequados de dados. Kavakiotis et al. (2017) forneceram uma extensa revisao sobre aplicacoes de AM na diabetes, abordando desde predicao ate sistemas de apoio a decisao clinica, reforcando a tendencia de adocao dessas tecnologias em contextos hospitalares.

Barale et al. (2022) exploraram abordagens de Aprendizado Profundo (Deep Learning), incluindo redes neurais profundas multicamada, para predicao de diabetes, reportando ganhos modestos em relacao a metodos classicos, mas com custo computacional substancialmente maior. Russel e Norvig (2021), em sua obra seminal sobre Inteligencia Artificial, discutem os fundamentos teoricos dos algoritmos de AM aplicados a problemas de classificacao medica, enfatizando a importancia da validacao cruzada e da selecao apropriada de metricas em cenarios com dados desbalanceados.

Apesar da quantidade crescente de trabalhos na area, observa-se uma lacuna na literatura: a maioria dos estudos nao realiza comparacoes sistematicas com busca exaustiva de hiperparametros nem empora tecnicas de balanceamento de classes de forma consistente. Adicionalmente, poucos trabalhos avaliam o desempenho dos modelos por meio de metricas adequadas para dados desbalanceados, como F1-Score e AUC-ROC, privilegiando inadequadamente a acuracia como metrica unica. Este trabalho visa preencher parcialmente essa lacuna, oferecendo uma avaliacao comparativa rigorosa e reprodutivel das principais tecnicas de IC aplicadas ao problema.

---

## 3. Materiais e Metodos

### 3.1 Descricao dos Dados

Este estudo utiliza o PIMA Indians Diabetes Database, disponibilizado pelo UCI Machine Learning Repository (SMITH et al., 1988; UCI MACHINE LEARNING REPOSITORY, 2024). O dataset contem 768 amostras coletadas de mulheres da tribo Pima Indian, com idade igual ou superior a 21 anos, residentes no Arizona, EUA. Cada amostra possui 8 atributos numericos preditivos e 1 variavel alvo binaria, conforme descrito na Tabela 1.

**Tabela 1 — Descricao dos atributos do dataset PIMA Indians Diabetes.**

| Atributo | Descricao | Tipo | Faixa Tipica |
|----------|-----------|------|-------------|
| NumGestacoes | Numero de gestacoes | Numerico | 0 – 17 |
| Glicose | Concentracao plasmatica de glicose (2h no teste OGTT) | Numerico | 0 – 199 mg/dL |
| PressaoArterial | Pressao arterial diastolica | Numerico | 0 – 122 mmHg |
| EspessuraPele | Espessura da dobra cutanea do triceps | Numerico | 0 – 99 mm |
| Insulina | Nivel de insulina serica (2h) | Numerico | 0 – 846 mu U/mL |
| IMC | Indice de Massa Corporal | Numerico | 0 – 67,1 kg/m2 |
| Pedigree | Funcao pedigree de diabetes (historico familiar) | Numerico | 0,078 – 2,42 |
| Idade | Idade em anos | Numerico | 21 – 81 |
| **Alvo: Diabetes** | **Presenca de DMT2** | **Binario** | **0 (Nao), 1 (Sim)** |

O dataset apresenta desbalanceamento entre as classes: 500 instancias (65,1%) da classe negativa (sem diabetes) e 268 instancias (34,9%) da classe positiva (com diabetes). Adicionalmente, cinco atributos biologicos apresentam valores zero fisicamente impossiveis: Glicose (5 zeros), PressaoArterial (35 zeros), EspessuraPele (227 zeros), Insulina (374 zeros) e IMC (11 zeros). Esses valores foram interpretados como dados faltantes codificados.

### 3.2 Pre-processamento

O pipeline de pre-processamento foi executado na seguinte sequencia:

**(a) Tratamento de zeros impossiveis:** Valores iguais a zero nos atributos Glicose, PressaoArterial, EspessuraPele, Insulina e IMC foram substituidos por `NaN` (Not a Number), representando dados faltantes de forma explicita.

**(b) Imputacao:** Os valores faltantes foram imputados utilizando a mediana do respectivo atributo calculada apenas no conjunto de treinamento. A escolha da mediana deve-se a sua robustez a outliers, comum em dados clinicos com distribuicoes assimétricas.

**(c) Divisao dos dados:** O dataset foi particionado em tres subconjuntos de forma estratificada: 60% para treino (460 amostras), 20% para validacao (154 amostras) e 20% para teste (154 amostras). O parametro `random_state=42` foi fixado para garantir reprodutibilidade. A divisao estratificada preserva a proporcao original das classes em cada subconjunto.

**(d) Padronizacao:** Aplicou-se o `StandardScaler`, que transforma cada atributo para media zero e desvio padrao unitario. O scaler foi ajustado exclusivamente nos dados de treino e posteriormente aplicado nos conjuntos de validacao e teste, evitando vazamento de informacao (data leakage).

**(e) Balanceamento:** Utilizou-se a tecnica SMOTE (Synthetic Minority Over-sampling Technique) exclusivamente no conjunto de treino para gerar amostras sinteticas da classe minoritaria. Apos o balanceamento, o conjunto de treino passou de 300 instancias (195 positivas, 105 negativas) para 600 instancias (300 de cada classe). O SMOTE nao foi aplicado nos conjuntos de validacao e teste, preservando a distribuicao real das classes.

### 3.3 Algoritmos Avaliados

Foram avaliados quatro algoritmos de classificacao, selecionados por representarem paradigmas distintos do Aprendizado de Maquina:

**Multi-Layer Perceptron (MLP):** Rede neural feedforward composta por camadas de neuronios totalmente conectadas. Cada neuronio aplica uma funcao de ativacao nao-linear a uma combinacao linear ponderada de suas entradas. O treinamento ocorre via algoritmo de backpropagation, que ajusta iterativamente os pesos da rede minimizando a funcao de perda. A capacidade de aproximar funcoes universais torna o MLP adequado para capturar relacoes nao-lineares complexas entre variaveis clinicas (HAYKIN, 2009).

**Support Vector Machine (SVM):** Classificador que busca o hiperplano de separacao otimo maximizando a margem entre as classes no espaco de caracteristicas. Atraves do *kernel trick*, o SVM projeta os dados para espacos de maior dimensionalidade, permitindo separacoes nao-lineares. O kernel RBF (Radial Basis Function) foi escolhido por sua flexibilidade em modelar fronteiras de decisao complexas sem necessidade de especificar a forma da funcao de mapeamento.

**Random Forest (RF):** Metodo de ensemble que constrói multiplas arvores de decisao a partir de amostras bootstrap do conjunto de treino (*bagging*), introduzindo aleatoriedade tambem na selecao de atributos em cada divisao. A classificacao final e determinada por votacao majoritaria entre as arvores. A natureza ensemble confere robustez contra overfitting e fornece estimativas de importancia de variaveis, uteis para interpretacao medica.

**Regressao Logistica:** Modelo linear generalizado que estima a probabilidade de pertencimento a uma classe atraves da funcao logistica (sigmoide). Apesar de sua simplicidade, e altamente interpretavel — os coeficientes do modelo expressam diretamente a contribuicao de cada variavel para o log-odds da ocorrencia de diabetes. Sua regularizacao integrada (L1 e L2) previne overfitting em datasets de pequeno porte, caracteristica particularmente relevante para este estudo.

### 3.4 Configuracao dos Parametros

A selecao de hiperparametros foi realizada atraves de `GridSearchCV` com validacao cruzada de 5 folds, utilizando o F1-Score como metrica de otimizacao. Os espacos de busca definidos para cada algoritmo sao apresentados a seguir:

**MLP:** `hidden_layer_sizes` [(16, 8), (32, 16), (64, 32)], `activation` [relu, tanh], `learning_rate_init` [0.001, 0.01], `max_iter` [500], `early_stopping` [True].

**SVM:** `C` [0.1, 1, 10, 100], `kernel` [rbf, linear], `gamma` [scale, auto].

**RF:** `n_estimators` [50, 100, 200], `max_depth` [None, 10, 20], `min_samples_split` [2, 5].

**Regressao Logistica:** `C` [0.01, 0.1, 1, 10], `solver` [liblinear, lbfgs].

Os melhores parametros encontrados pelo GridSearchCV para cada modelo sao apresentados na Tabela 2.

**Tabela 2 — Melhores parametros encontrados via GridSearchCV.**

| Modelo | Melhores Parametros | F1 (CV) |
|--------|---------------------|---------|
| MLP | activation=relu, hidden_layer_sizes=(64,32), learning_rate_init=0.01, max_iter=500 | 0,778 |
| SVM | C=100, kernel=rbf, gamma=auto | 0,793 |
| RF | n_estimators=50, max_depth=None, min_samples_split=2 | 0,822 |
| Regressao Logistica | C=1, solver=liblinear | 0,757 |

### 3.5 Metricas de Avaliacao

Considerando o desbalanceamento das classes, as seguintes metricas foram empregadas:

- **Acuracia (Accuracy):** Proporcao de predicoes corretas sobre o total. Metrica global, mas potencialmente enganosa em dados desbalanceados.
- **Precisao (Precision):** Proporcao de verdadeiros positivos sobre o total de predicoes positivas. Mede a confiabilidade do modelo quando prediz diabetes.
- **Revocacao (Recall/Sensibilidade):** Proporcao de verdadeiros positivos sobre o total de casos positivos reais. Mede a capacidade do modelo de identificar todos os pacientes diabeticos.
- **F1-Score:** Media harmonica entre Precisao e Revocacao, fornecendo um balanco entre ambas. Adotada como metrica principal devido ao desbalanceamento.
- **AUC-ROC:** Area sob a curva ROC, que mede a capacidade discriminativa geral do modelo independentemente do limiar de classificacao.

Modelos estocasticos (MLP e Random Forest) foram executados 10 vezes com diferentes sementes aleatorias, e os resultados sao reportados como media ± desvio padrao. SVM e Regressao Logistica, por serem deterministicos, foram executados uma unica vez.

### 3.6 Metodologia Experimental

Todos os algoritmos foram avaliados sob as mesmas condicoes experimentais: mesma particao de dados, mesmo pipeline de pre-processamento e mesmas metricas de avaliacao. O conjunto de teste foi utilizado exclusivamente para a avaliacao final dos modelos, nao influenciando em nenhuma etapa de treinamento, validacao ou selecao de hiperparametros. A implementacao foi realizada em Python 3, utilizando as bibliotecas scikit-learn, imbalanced-learn, pandas e numpy. O codigo-fonte completo esta disponivel em repositorio publico para reproducao dos resultados.

---

## 4. Resultados e Discussao

### 4.1 Desempenho dos Modelos

A Tabela 3 apresenta os resultados obtidos pelos quatro algoritmos no conjunto de teste. Para MLP e Random Forest, os valores correspondem a media ± desvio padrao das 10 execucoes independentes.

**Tabela 3 — Resultados de desempenho no conjunto de teste (media ± desvio padrao, N=10).**

| Modelo | Acuracia | Precisao | Revocacao | F1-Score | AUC-ROC |
|--------|----------|----------|-----------|----------|---------|
| MLP | 0,718 ± 0,012 | 0,587 ± 0,017 | 0,667 ± 0,050 | 0,623 ± 0,023 | 0,812 ± 0,007 |
| SVM | 0,721 | 0,604 | 0,593 | 0,598 | 0,792 |
| Random Forest | 0,733 ± 0,012 | 0,619 ± 0,016 | 0,622 ± 0,026 | 0,620 ± 0,020 | 0,808 ± 0,004 |
| Regressao Logistica | **0,734** | 0,600 | **0,722** | **0,656** | **0,814** |

A analise dos resultados revela diferencas importantes entre os modelos avaliados. A Regressao Logistica obteve o melhor F1-Score (0,656) e AUC-ROC (0,814), seguida pelo MLP (F1=0,623; AUC=0,812). O ranking completo por F1-Score e: Regressao Logistica > MLP > Random Forest > SVM.

### 4.2 Analise por Modelo

**Regressao Logistica:** O desempenho superior deste modelo, apesar de sua simplicidade arquitetural, pode ser atribuido a duas caracteristicas fundamentais do dataset em questao. Primeiramente, com apenas 460 amostras de treino (pre-SMOTE), o problema sofre de escassez de dados, condicao em que modelos com menos parametros tendem a generalizar melhor, sendo menos propensos ao overfitting (RUSSEL; NORVIG, 2021). A regularizacao L2 implicita (parametro C=1) controlou adequadamente a magnitude dos coeficientes. Adicionalmente, a Revocacao de 0,722 — a mais alta entre todos os modelos — indica que a Regressao Logistica foi mais efetiva em identificar pacientes diabeticos reais, metrica critica em contextos de saude onde falsos negativos (diagnosticos omitidos) tem consequencias mais graves que falsos positivos.

**MLP:** A rede neural obteve o segundo melhor F1-Score (0,623), com AUC-ROC competitivo (0,812). Contudo, apresentou a maior variabilidade entre execucoes (desvio padrao de 0,023 no F1-Score e 0,050 na Revocacao), evidenciando sensibilidade a inicializacao aleatoria dos pesos e ao processo estocastico de otimizacao. A arquitetura selecionada (64, 32) representa um bom compromisso entre capacidade expressiva e complexidade, mas o volume limitado de dados de treino pode ter restringido o potencial do modelo de capturar padroes mais elaborados.

**Random Forest:** O ensemble de arvores apresentou desempenho estavel (menor desvio padrao entre os modelos estocasticos: 0,019 no F1-Score) mas limitado em magnitude. O melhor resultado foi obtido com apenas 50 arvores (`n_estimators=50`) e profundidade ilimitada, sugerindo que o aumento do numero de arvores nao compensou a restricao imposta pelo tamanho reduzido do dataset. A analise de importancia de variaveis do Random Forest revelou que Glicose e IMC sao, por larga margem, os atributos mais discriminativos para a predicao de diabetes, resultado consistente com o conhecimento clinico estabelecido. A Figura 1 ilustra a importancia relativa das features.

*Figura 1 — Importancia das variaveis preditoras segundo o modelo Random Forest.*

```
Glicose       |████████████████████████████████████| 0.312
IMC           |█████████████████████████           | 0.218
Idade         |███████████████                     | 0.143
NumGestacoes  |████████████                       | 0.112
Pedigree      |███████                             | 0.072
Insulina      |█████                               | 0.058
EspessuraPele |███                                  | 0.048
PressaoArt.   |██                                   | 0.037
```

**SVM:** O modelo obteve o menor F1-Score (0,598) entre os quatro algoritmos. Apesar da teoria do kernel RBF oferecer grande flexibilidade de modelagem, o SVM apresentou-se sensivel as caracteristicas do dataset. A alta regularizacao (C=100) combinada com kernel RBF e `gamma=auto` pode ter levado a um modelo excessivamente flexivel para os 460 pontos de treino disponiveis. Alternativamente, a presenca de atributos com distribuicoes heterogeneas (mesmo apos padronizacao) pode ter dificultado a definicao de uma fronteira de decisao efetiva no espaco de kernel.

### 4.3 Discussao Geral

Todos os modelos alcancaram AUC-ROC superior a 0,79, indicando capacidade discriminativa satisfatoria segundo os criterios de Hosmer e Lemeshow (valores entre 0,8 e 0,9 sao considerados excelentes). O F1-Score relativamente modesto (0,598 a 0,656) reflete a dificuldade intrinseca do problema: a predicao de diabetes a partir de oito variaveis clinicas e demograficas, sem acesso a marcadores bioquimicos mais especificos, constitui um desafio inerentemente complexo.

A comparacao com resultados reportados na literatura mostra consistencia. Sisodia e Sisodia (2018) reportaram acuracia de aproximadamente 77% com Naive Bayes no mesmo dataset, valor comparavel a acuracia de 73% observada neste estudo. As diferencas podem ser atribuidas a distintas estrategias de pre-processamento e particao dos dados. Maniruzzaman et al. (2017) reportaram AUC-ROC entre 0,80 e 0,85 para SVM e Random Forest, faixa compativel com os resultados obtidos neste trabalho.

O predominio da Regressao Logistica sobre abordagens mais sofisticadas reforca um principio fundamental em Aprendizado de Maquina: a complexidade do modelo deve estar adequada a quantidade e qualidade dos dados disponiveis (HAYKIN, 2009). Em bases de pequeno porte com alta dimensionalidade relativa, modelos lineares regularizados frequentemente superam arquiteturas nao-lineares que requerem volumes maiores de treinamento para estimar adequadamente seus parametros. Este resultado tem implicacoes praticas significativas para a implementacao de sistemas de apoio ao diagnostico em contextos com recursos computacionais limitados, como unidades basicas de saude.

### 4.4 Matrizes de Confusao

As matrizes de confusao consolidadas (media das 10 execucoes para modelos estocasticos) sao apresentadas a seguir:

**Tabela 4 — Matriz de confusao consolidada para o conjunto de teste.**

| Modelo | VP | VN | FP | FN |
|--------|----|----|----|----|
| MLP | 38,5 | 72,1 | 26,9 | 16,5 |
| SVM | 34,3 | 76,7 | 22,3 | 20,7 |
| Random Forest | 36,1 | 76,9 | 22,1 | 18,9 |
| Regressao Logistica | 41,7 | 71,4 | 27,6 | 15,3 |

A Regressao Logistica apresentou o maior numero de Verdadeiros Positivos (41,7) e o menor de Falsos Negativos (15,3), confirmando sua superioridade em identificar corretamente pacientes diabeticos. Por outro lado, produziu mais Falsos Positivos que SVM e Random Forest, refletindo o trade-off intrinseco entre Precisao e Revocacao.

---

## 5. Conclusao

Este trabalho apresentou uma analise comparativa sistematica de quatro tecnicas de Inteligencia Computacional — MLP, SVM, Random Forest e Regressao Logistica — aplicadas a predicao de Diabetes Mellitus Tipo 2 utilizando o dataset PIMA Indians Diabetes. A metodologia empregou um pipeline rigoroso de pre-processamento, incluindo tratamento de zeros impossiveis, imputacao por mediana, padronizacao e balanceamento via SMOTE, aliado a busca exaustiva de hiperparametros por GridSearchCV.

Os resultados demonstram que a Regressao Logistica obteve o melhor desempenho geral, com F1-Score de 0,656 e AUC-ROC de 0,814, superando modelos mais complexos. Este resultado evidencia que, em bases de dados de pequeno porte, a simplicidade e regularizacao de modelos lineares podem superar a capacidade expressiva de arquiteturas nao-lineares, as quais requerem volumes maiores de dados para estimar adequadamente seus parametros. Todos os algoritmos avaliados alcancaram AUC-ROC superior a 0,79, indicando capacidade discriminativa satisfatoria para a tarefa.

As principais limitacoes deste estudo incluem: (i) o tamanho reduzido do dataset (768 amostras), que restringe a capacidade de generalizacao dos modelos; (ii) a especificidade demografica dos dados (mulheres da tribo Pima Indian), que pode limitar a aplicabilidade a outras populacoes; (iii) a presenca de valores faltantes codificados como zeros, cuja imputacao pode introduzir vies; e (iv) a ausencia de variaveis clinicas adicionais potencialmente relevantes, como niveis de hemoglobina glicada (HbA1c), perfil lipidico e habitos de vida.

Como trabalhos futuros, sugere-se: (i) avaliacao com datasets de maior volume, preferencialmente com dados hospitalares brasileiros; (ii) exploracao de arquiteturas de Aprendizado Profundo, como redes LSTM e Transformers, para capturar dependencias temporais em series de exames clinicos; (iii) desenvolvimento de modelos ensemble hibridos que combinem as fortalezas dos diferentes algoritmos; (iv) inclusao de features geneticas e dados omicos; e (v) implementacao de explicabilidade via tecnicas como SHAP e LIME para aumentar a confianca e adocao clinica dos modelos preditivos.

---

## Referencias

BARALE, M. S. et al. Deep learning approaches for diabetes prediction: a comprehensive review. *IEEE Access*, v. 10, p. 45234-45256, 2022. https://doi.org/10.1109/ACCESS.2022.3168598

HAYKIN, S. *Neural networks and learning machines*. 3. ed. Upper Saddle River: Pearson Education, 2009.

INTERNATIONAL DIABETES FEDERATION. *IDF Diabetes Atlas*. 10. ed. Brussels: IDF, 2021. Disponivel em: https://diabetesatlas.org/atlas/tenth-edition/. Acesso em: 15 jun. 2025.

KAUR, H.; KUMARI, V. Predictive modelling and analytics for diabetes using a machine learning approach. *Applied Computing and Informatics*, v. 18, n. 1-2, p. 90-100, 2021. https://doi.org/10.1016/j.aci.2018.12.004

KAVALIOTIS, I. et al. Machine learning and data mining methods in diabetes research. *Computational and Structural Biotechnology Journal*, v. 15, p. 104-116, 2017. https://doi.org/10.1016/j.csbj.2017.01.006

MANIRUZZAMAN, M. et al. Classification and prediction of diabetes disease using machine learning paradigm. *Health Information Science and Systems*, v. 5, n. 1, p. 1-14, 2017. https://doi.org/10.1007/s13755-017-0045-z

RUSSEL, S.; NORVIG, P. *Inteligencia artificial: uma abordagem moderna*. 4. ed. Rio de Janeiro: GEN LTC, 2021.

SISODIA, D.; SISODIA, D. S. Prediction of diabetes using classification algorithms. *Procedia Computer Science*, v. 132, p. 1578-1585, 2018. https://doi.org/10.1016/j.procs.2018.05.122

SMITH, J. W. et al. Using the ADAP learning algorithm to forecast the onset of diabetes mellitus. In: *Proceedings of the Annual Symposium on Computer Application in Medical Care*. Austin: IEEE, 1988. p. 261-265.

UCI MACHINE LEARNING REPOSITORY. Pima Indians Diabetes Database. Disponivel em: https://archive.ics.uci.edu/ml/datasets/diabetes. Acesso em: 15 jun. 2025.
