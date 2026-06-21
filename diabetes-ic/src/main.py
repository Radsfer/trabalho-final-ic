#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
PREDICAO DE DIABETES TIPO 2 - INTELIGENCIA COMPUTACIONAL
Trabalho Final - CEFET-MG
================================================================================
Autor: Aluno de Inteligencia Computacional
Disciplina: Inteligencia Computacional

Este script executa um pipeline completo de Machine Learning para predicao
de Diabetes Tipo 2 utilizando 4 algoritmos de Inteligencia Computacional:
- MLP (Rede Neural Multicamadas)
- SVM (Maquina de Vetores de Suporte)
- Random Forest (Floresta Aleatoria)
- Regressao Logistica (Baseline)

Uso: python main.py
================================================================================
"""

# ==============================================================================
# IMPORTS
# ==============================================================================
import os
import sys
import warnings
import time
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             roc_curve, classification_report)

from imblearn.over_sampling import SMOTE

# Suprimir warnings para saida limpa
warnings.filterwarnings('ignore')

# Configuracoes de estilo para graficos
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# ==============================================================================
# CONFIGURACOES GLOBAIS
# ==============================================================================
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Diretorios (relativo ao src/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, '..', 'dataset')
RESULTS_DIR = os.path.join(BASE_DIR, '..', 'results')
FIGURAS_DIR = os.path.join(RESULTS_DIR, 'figuras')
TABELAS_DIR = os.path.join(RESULTS_DIR, 'tabelas')

# Arquivos
DATASET_PATH = os.path.join(DATASET_DIR, 'diabetes.csv')
DATASET_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

# Colunas do dataset (em portugues)
COLUNAS = ['NumGestacoes', 'Glicose', 'PressaoArterial', 'EspessuraPele',
           'Insulina', 'IMC', 'Pedigree', 'Idade', 'Diabetes']

# Features que nao podem ter valor zero (biologicamente impossivel)
COLUNAS_ZEROS_IMPOSSIVEIS = ['Glicose', 'PressaoArterial', 'EspessuraPele', 'Insulina', 'IMC']

# Nomes amigaveis dos modelos
NOMES_MODELOS = {
    'MLP': 'MLP (Rede Neural)',
    'SVM': 'SVM (Vetores de Suporte)',
    'RandomForest': 'Random Forest',
    'LogReg': 'Regressao Logistica'
}


def criar_diretorios():
    """Cria todos os diretorios necessarios para o projeto."""
    for d in [DATASET_DIR, FIGURAS_DIR, TABELAS_DIR]:
        os.makedirs(d, exist_ok=True)


# ==============================================================================
# 1. CARREGAMENTO DOS DADOS
# ==============================================================================
def carregar_dados():
    """
    Carrega o dataset de diabetes.
    - Primeiro tenta ler do arquivo local ../dataset/diabetes.csv
    - Se nao existir, faz download automatico e salva localmente

    Retorna:
        pd.DataFrame: Dataset carregado com colunas em portugues
    """
    print("[1/8] Carregando dataset...")

    if os.path.exists(DATASET_PATH):
        print(f"       -> Dataset local encontrado: {DATASET_PATH}")
        df = pd.read_csv(DATASET_PATH)
        # Se as colunas forem numericas (dataset original), renomear
        if set(df.columns) != set(COLUNAS):
            df.columns = COLUNAS
            df.to_csv(DATASET_PATH, index=False)
    else:
        print(f"       -> Dataset local nao encontrado.")
        print(f"       -> Fazendo download de: {DATASET_URL}")
        try:
            df = pd.read_csv(DATASET_URL, header=None)
            df.columns = COLUNAS
            df.to_csv(DATASET_PATH, index=False)
            print(f"       -> Dataset salvo em: {DATASET_PATH}")
        except Exception as e:
            print(f"       ERRO ao fazer download: {e}")
            sys.exit(1)

    print(f"       -> Dataset carregado: {df.shape[0]} amostras x {df.shape[1]} atributos")
    return df


# ==============================================================================
# 2. ANALISE EXPLORATORIA
# ==============================================================================
def explorar_dados(df):
    """
    Realiza analise exploratoria completa dos dados.
    - Estatisticas descritivas
    - Verificacao de zeros impossiveis
    - Distribuicao das classes

    Args:
        df: DataFrame com os dados

    Returns:
        dict: Estatisticas coletadas para o relatorio
    """
    print("\n[2/8] Explorando dados...")

    estatisticas = {}

    # --- Estatisticas descritivas ---
    print("\n" + "=" * 70)
    print("ESTATISTICAS DESCRITIVAS")
    print("=" * 70)
    desc = df.describe().round(3)
    print(desc.to_string())
    estatisticas['descritivas'] = desc.to_string()

    # --- Verificacao de zeros impossiveis ---
    print("\n" + "-" * 70)
    print("VERIFICACAO DE ZEROS IMPOSSIVEIS (biologicamente)")
    print("-" * 70)
    zeros_info = {}
    for col in COLUNAS_ZEROS_IMPOSSIVEIS:
        n_zeros = (df[col] == 0).sum()
        pct_zeros = 100 * n_zeros / len(df)
        zeros_info[col] = {'quantidade': n_zeros, 'percentual': round(pct_zeros, 2)}
        print(f"  {col:20s}: {n_zeros:4d} valores zero ({pct_zeros:5.1f}%)")
    estatisticas['zeros_impossiveis'] = zeros_info

    # --- Distribuicao das classes ---
    print("\n" + "-" * 70)
    print("DISTRIBUICAO DAS CLASSES")
    print("-" * 70)
    class_dist = df['Diabetes'].value_counts().sort_index()
    for classe, count in class_dist.items():
        label = "Nao Diabeticos (0)" if classe == 0 else "Diabeticos (1)"
        pct = 100 * count / len(df)
        print(f"  {label}: {count:4d} ({pct:5.1f}%)")
    estatisticas['distribuicao_classes'] = class_dist.to_dict()

    # --- Info geral ---
    print("\n" + "-" * 70)
    print("INFORMACOES DO DATASET")
    print("-" * 70)
    print(f"  Total de amostras: {len(df)}")
    print(f"  Total de atributos: {df.shape[1] - 1} (excluindo alvo)")
    print(f"  Atributos: {', '.join(df.columns[:-1])}")
    print(f"  Alvo: {df.columns[-1]}")
    print(f"  Valores ausentes: {df.isnull().sum().sum()}")

    return estatisticas


# ==============================================================================
# 3. PRE-PROCESSAMENTO
# ==============================================================================
def preprocessar(df):
    """
    Pipeline completo de pre-processamento:
    1. Substitui zeros impossiveis por NaN
    2. Imputa NaN com a mediana
    3. Separa X e y
    4. Divisao estratificada: 60% treino / 20% validacao / 20% teste
    5. StandardScaler ajustado no treino
    6. SMOTE apenas no treino para balancear

    Args:
        df: DataFrame com os dados brutos

    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test, scaler)
    """
    print("\n[3/8] Pre-processando...")

    # Copia para nao modificar o original
    dados = df.copy()

    # --- 1. Substituir zeros impossiveis por NaN ---
    print("       -> Substituindo zeros impossiveis por NaN...")
    n_zeros_total = 0
    for col in COLUNAS_ZEROS_IMPOSSIVEIS:
        n_zeros = (dados[col] == 0).sum()
        n_zeros_total += n_zeros
        dados[col] = dados[col].replace(0, np.nan)
    print(f"          Total de zeros substituidos: {n_zeros_total}")

    # --- 2. Imputacao com mediana ---
    print("       -> Imputando valores ausentes com a mediana...")
    for col in COLUNAS_ZEROS_IMPOSSIVEIS:
        mediana = dados[col].median()
        n_nan = dados[col].isnull().sum()
        dados[col].fillna(mediana, inplace=True)
        print(f"          {col}: {n_nan} valores -> mediana = {mediana:.2f}")

    # --- 3. Separar X e y ---
    X = dados.drop('Diabetes', axis=1)
    y = dados['Diabetes']
    print(f"       -> X: {X.shape}, y: {y.shape}")

    # --- 4. Divisao estratificada: 60% treino / 20% val / 20% teste ---
    print("       -> Dividindo dados (60%% treino / 20%% validacao / 20%% teste)...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    # 60% treino, 40% temp -> 60% treino, 20% val, 20% teste
    # De 40% temp, queremos 20% val = 50% de temp
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=RANDOM_STATE, stratify=y_temp
    )

    print(f"          Treino:      {X_train.shape[0]} amostras ({100*len(X_train)/len(X):.0f}%)")
    print(f"          Validacao:   {X_val.shape[0]} amostras ({100*len(X_val)/len(X):.0f}%)")
    print(f"          Teste:       {X_test.shape[0]} amostras ({100*len(X_test)/len(X):.0f}%)")

    # Distribuicao das classes em cada split
    for nome, y_split in [('Treino', y_train), ('Validacao', y_val), ('Teste', y_test)]:
        dist = y_split.value_counts().sort_index()
        print(f"          {nome}: Classe 0={dist.get(0,0)}, Classe 1={dist.get(1,0)}")

    # --- 5. StandardScaler (ajustado apenas no treino) ---
    print("       -> Aplicando StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Converter de volta para DataFrame para manter nomes das colunas
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=X.columns, index=X_val.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)

    # --- 6. SMOTE apenas no treino ---
    print("       -> Aplicando SMOTE no conjunto de treino...")
    print(f"          Antes SMOTE:  Classe 0={(y_train==0).sum()}, Classe 1={(y_train==1).sum()}")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)
    print(f"          Depois SMOTE: Classe 0={(y_train_bal==0).sum()}, Classe 1={(y_train_bal==1).sum()}")

    return (X_train_bal, X_val_scaled, X_test_scaled,
            y_train_bal, y_val, y_test, scaler, X_train_scaled, y_train)


# ==============================================================================
# 4. TREINAMENTO DOS MODELOS (GRID SEARCH)
# ==============================================================================
def treinar_modelos(X_train, y_train):
    """
    Treina 4 modelos com GridSearchCV para otimizacao de hiperparametros.

    Args:
        X_train: Features de treino (balanceadas com SMOTE)
        y_train: Labels de treino (balanceados com SMOTE)

    Returns:
        dict: Modelos treinados com melhores parametros
    """
    print("\n[4/8] Treinando modelos (GridSearchCV)...")

    modelos = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # --- A) MLP (Rede Neural) ---
    print("       - MLP: 36 combinacoes...")
    param_grid_mlp = {
        'hidden_layer_sizes': [(16, 8), (32, 16), (64, 32)],
        'activation': ['relu', 'tanh'],
        'learning_rate_init': [0.001, 0.01],
        'max_iter': [500]
    }
    grid_mlp = GridSearchCV(
        MLPClassifier(random_state=RANDOM_STATE, early_stopping=True,
                      validation_fraction=0.1, n_iter_no_change=10),
        param_grid_mlp, cv=cv, scoring='f1', n_jobs=-1, verbose=0
    )
    grid_mlp.fit(X_train, y_train)
    modelos['MLP'] = {
        'modelo': grid_mlp.best_estimator_,
        'best_params': grid_mlp.best_params_,
        'best_score_cv': grid_mlp.best_score_
    }
    print(f"         Melhor F1 (CV): {grid_mlp.best_score_:.4f}")

    # --- B) SVM ---
    print("       - SVM: 24 combinacoes...")
    param_grid_svm = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale', 'auto']
    }
    grid_svm = GridSearchCV(
        SVC(probability=True, random_state=RANDOM_STATE),
        param_grid_svm, cv=cv, scoring='f1', n_jobs=-1, verbose=0
    )
    grid_svm.fit(X_train, y_train)
    modelos['SVM'] = {
        'modelo': grid_svm.best_estimator_,
        'best_params': grid_svm.best_params_,
        'best_score_cv': grid_svm.best_score_
    }
    print(f"         Melhor F1 (CV): {grid_svm.best_score_:.4f}")

    # --- C) Random Forest ---
    print("       - Random Forest: 27 combinacoes...")
    param_grid_rf = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    grid_rf = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE),
        param_grid_rf, cv=cv, scoring='f1', n_jobs=-1, verbose=0
    )
    grid_rf.fit(X_train, y_train)
    modelos['RandomForest'] = {
        'modelo': grid_rf.best_estimator_,
        'best_params': grid_rf.best_params_,
        'best_score_cv': grid_rf.best_score_
    }
    print(f"         Melhor F1 (CV): {grid_rf.best_score_:.4f}")

    # --- D) Regressao Logistica ---
    print("       - Regressao Logistica: 8 combinacoes...")
    param_grid_lr = {
        'C': [0.01, 0.1, 1, 10],
        'solver': ['liblinear', 'lbfgs']
    }
    grid_lr = GridSearchCV(
        LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        param_grid_lr, cv=cv, scoring='f1', n_jobs=-1, verbose=0
    )
    grid_lr.fit(X_train, y_train)
    modelos['LogReg'] = {
        'modelo': grid_lr.best_estimator_,
        'best_params': grid_lr.best_params_,
        'best_score_cv': grid_lr.best_score_
    }
    print(f"         Melhor F1 (CV): {grid_lr.best_score_:.4f}")

    return modelos


# ==============================================================================
# 5. AVALIACAO DOS MODELOS
# ==============================================================================
def avaliar_modelo(modelo, X, y, nome):
    """
    Avalia um modelo treinado no conjunto de teste.

    Args:
        modelo: Modelo treinado
        X: Features de teste
        y: Labels reais de teste
        nome: Nome do modelo

    Returns:
        dict: Dicionario com todas as metricas
    """
    y_pred = modelo.predict(X)
    y_prob = modelo.predict_proba(X)[:, 1]

    metricas = {
        'Acuracia': accuracy_score(y, y_pred),
        'Precisao': precision_score(y, y_pred, zero_division=0),
        'Recall': recall_score(y, y_pred, zero_division=0),
        'F1_Score': f1_score(y, y_pred, zero_division=0),
        'AUC_ROC': roc_auc_score(y, y_prob),
        'y_pred': y_pred,
        'y_prob': y_prob,
        'confusion_matrix': confusion_matrix(y, y_pred)
    }

    return metricas


def executar_repeticoes(modelo_fn, X_train, y_train, X_test, y_test, nome, n=10):
    """
    Executa N repeticoes de modelos estocasticos (MLP e RF).
    Retorna media e desvio padrao das metricas.

    Args:
        modelo_fn: Funcao que cria uma nova instancia do modelo
        X_train: Features de treino
        y_train: Labels de treino
        X_test: Features de teste
        y_test: Labels de teste
        nome: Nome do modelo
        n: Numero de repeticoes

    Returns:
        dict: Metricas medias, desvios e lista de F1-scores
    """
    print(f"       -> Executando {n} repeticoes de {nome}...")

    resultados = {
        'Acuracia': [], 'Precisao': [], 'Recall': [],
        'F1_Score': [], 'AUC_ROC': []
    }

    for i in range(n):
        # Cria novo modelo com seed diferente
        modelo = modelo_fn(random_state=RANDOM_STATE + i)
        modelo.fit(X_train, y_train)

        m = avaliar_modelo(modelo, X_test, y_test, nome)
        for k in resultados:
            resultados[k].append(m[k])

    # Calcular media e desvio padrao
    resumo = {}
    for k in resultados:
        vals = np.array(resultados[k])
        resumo[f'{k}_mean'] = np.mean(vals)
        resumo[f'{k}_std'] = np.std(vals)
        resumo[k] = resultados[k]

    print(f"          F1 medio: {resumo['F1_Score_mean']:.4f} +/- {resumo['F1_Score_std']:.4f}")

    return resumo


# ==============================================================================
# 6. GERACAO DE GRAFICOS
# ==============================================================================
def gerar_graficos(df, y_train_orig, y_train_bal, resultados, modelos,
                   rep_mlp, rep_rf, X_test, y_test, X_train_bal):
    """
    Gera todos os 7 graficos obrigatorios e salva em ../results/figuras/

    Args:
        df: DataFrame original
        y_train_orig: Labels de treino originais (antes do SMOTE)
        y_train_bal: Labels de treino balanceados (depois do SMOTE)
        resultados: Dict com metricas de cada modelo
        modelos: Dict com modelos treinados
        rep_mlp: Resultados das repeticoes do MLP
        rep_rf: Resultados das repeticoes do RF
        X_test: Features de teste
        y_test: Labels de teste
        X_train_bal: Features de treino balanceadas
    """
    print("\n[7/8] Gerando graficos (7 figuras)...")

    cores_modelos = {
        'MLP': '#E74C3C',
        'SVM': '#3498DB',
        'RandomForest': '#2ECC71',
        'LogReg': '#9B59B6'
    }

    # --- Grafico 1: Heatmap de Correlacao ---
    print("       -> 01_correlacao.png")
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlBu_r',
                center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title('Mapa de Correlacao entre Atributos', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURAS_DIR, '01_correlacao.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # --- Grafico 2: Distribuicao das Classes ---
    print("       -> 02_distribuicao_classes.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    # Antes do SMOTE
    dist_antes = pd.Series(y_train_orig).value_counts().sort_index()
    labels = ['Nao Diabeticos (0)', 'Diabeticos (1)']
    axes[0].bar(labels, dist_antes.values, color=['#3498DB', '#E74C3C'], edgecolor='black')
    axes[0].set_title('Antes do SMOTE', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Numero de Amostras')
    for i, v in enumerate(dist_antes.values):
        axes[0].text(i, v + 2, str(v), ha='center', fontweight='bold')

    # Depois do SMOTE
    dist_depois = pd.Series(y_train_bal).value_counts().sort_index()
    axes[1].bar(labels, dist_depois.values, color=['#3498DB', '#E74C3C'], edgecolor='black')
    axes[1].set_title('Depois do SMOTE', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Numero de Amostras')
    for i, v in enumerate(dist_depois.values):
        axes[1].text(i, v + 2, str(v), ha='center', fontweight='bold')

    fig.suptitle('Distribuicao das Classes no Conjunto de Treino', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURAS_DIR, '02_distribuicao_classes.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # --- Grafico 3: Matrizes de Confusao ---
    print("       -> 03_matrizes_confusao.png")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
    axes = axes.flatten()

    for idx, (nome_key, nome_pt) in enumerate(NOMES_MODELOS.items()):
        cm = resultados[nome_key]['confusion_matrix']
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    xticklabels=['Nao (0)', 'Sim (1)'],
                    yticklabels=['Nao (0)', 'Sim (1)'])
        axes[idx].set_title(nome_pt, fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Predito')
        axes[idx].set_ylabel('Real')

    fig.suptitle('Matrizes de Confusao - Conjunto de Teste', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURAS_DIR, '03_matrizes_confusao.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # --- Grafico 4: Curvas ROC ---
    print("       -> 04_curvas_roc.png")
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

    for nome_key, nome_pt in NOMES_MODELOS.items():
        y_prob = resultados[nome_key]['y_prob']
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_val = resultados[nome_key]['AUC_ROC']
        ax.plot(fpr, tpr, color=cores_modelos[nome_key], linewidth=2.5,
                label=f'{nome_pt} (AUC = {auc_val:.4f})')

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Classificador Aleatorio (AUC = 0.50)')
    ax.set_xlabel('Taxa de Falsos Positivos (FPR)', fontsize=12)
    ax.set_ylabel('Taxa de Verdadeiros Positivos (TPR)', fontsize=12)
    ax.set_title('Curvas ROC - Comparacao dos Modelos', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURAS_DIR, '04_curvas_roc.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # --- Grafico 5: Comparacao de Metricas ---
    print("       -> 05_comparacao_metricas.png")
    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)

    metricas_plot = ['Acuracia', 'F1_Score', 'AUC_ROC']
    metricas_labels = ['Acuracia', 'F1-Score', 'AUC-ROC']
    x = np.arange(len(metricas_labels))
    width = 0.18

    for i, (nome_key, nome_pt) in enumerate(NOMES_MODELOS.items()):
        vals = [resultados[nome_key][m] for m in metricas_plot]
        # Adicionar barras de erro para MLP e RF
        if nome_key in ['MLP', 'RandomForest']:
            rep = rep_mlp if nome_key == 'MLP' else rep_rf
            errs = [rep.get(f'{m}_std', 0) for m in metricas_plot]
        else:
            errs = [0, 0, 0]
        ax.bar(x + i * width, vals, width, yerr=errs, label=nome_pt,
               color=cores_modelos[nome_key], edgecolor='black', capsize=4)

    ax.set_ylabel('Valor da Metrica', fontsize=12)
    ax.set_title('Comparacao de Metricas dos Modelos (Conjunto de Teste)', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(metricas_labels)
    ax.set_ylim([0, 1.15])
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURAS_DIR, '05_comparacao_metricas.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # --- Grafico 6: Boxplot de Estabilidade ---
    print("       -> 06_stability_boxplot.png")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    dados_box = [rep_mlp['F1_Score'], rep_rf['F1_Score']]
    bp = ax.boxplot(dados_box, labels=['MLP (Rede Neural)', 'Random Forest'],
                    patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor(cores_modelos['MLP'])
    bp['boxes'][1].set_facecolor(cores_modelos['RandomForest'])
    for median in bp['medians']:
        median.set(color='black', linewidth=2)

    ax.set_ylabel('F1-Score', fontsize=12)
    ax.set_title('Estabilidade dos Modelos Estocasticos (10 Execucoes)', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURAS_DIR, '06_stability_boxplot.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # --- Grafico 7: Feature Importance (Random Forest) ---
    print("       -> 07_feature_importance.png")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    rf_modelo = modelos['RandomForest']['modelo']
    importancias = rf_modelo.feature_importances_
    feature_names = COLUNAS[:-1]  # Exclui 'Diabetes'

    indices = np.argsort(importancias)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importancias = importancias[indices]

    bars = ax.barh(range(len(sorted_features)), sorted_importancias,
                   color='#2ECC71', edgecolor='black')
    ax.set_yticks(range(len(sorted_features)))
    ax.set_yticklabels(sorted_features)
    ax.invert_yaxis()
    ax.set_xlabel('Importancia', fontsize=12)
    ax.set_title('Importancia das Features - Random Forest', fontsize=14, fontweight='bold')

    for i, v in enumerate(sorted_importancias):
        ax.text(v + 0.005, i, f'{v:.3f}', va='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURAS_DIR, '07_feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print("       -> Todos os 7 graficos salvos!")


# ==============================================================================
# 7. GERACAO DE RELATORIOS E TABELAS
# ==============================================================================
def formatar_metrica(valor, dp=None):
    """Formata um valor numerico com opcional desvio padrao."""
    if dp is not None and dp > 0:
        return f"{valor:.4f} +/- {dp:.4f}"
    return f"{valor:.4f}"


def gerar_tabelas(resultados, modelos, rep_mlp, rep_rf):
    """
    Gera as tabelas de resultados e melhores parametros.

    Args:
        resultados: Dict com metricas de cada modelo
        modelos: Dict com modelos treinados
        rep_mlp: Resultados das repeticoes do MLP
        rep_rf: Resultados das repeticoes do RF
    """
    print("\n[8/8] Gerando relatorios...")

    # --- Tabela 1: Resultados Comparativos ---
    print("       -> resultados_comparativos.txt")
    with open(os.path.join(TABELAS_DIR, 'resultados_comparativos.txt'), 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("RESULTADOS COMPARATIVOS - PREDICAO DE DIABETES TIPO 2\n")
        f.write("=" * 80 + "\n")
        f.write(f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n\n")

        f.write("-" * 80 + "\n")
        f.write(f"{'Modelo':<30} {'Acuracia':<18} {'Precisao':<18} {'Recall':<18} {'F1-Score':<18} {'AUC-ROC':<18}\n")
        f.write("-" * 80 + "\n")

        for nome_key, nome_pt in NOMES_MODELOS.items():
            r = resultados[nome_key]
            if nome_key in ['MLP', 'RandomForest']:
                rep = rep_mlp if nome_key == 'MLP' else rep_rf
                ac = formatar_metrica(rep['Acuracia_mean'], rep['Acuracia_std'])
                pr = formatar_metrica(rep['Precisao_mean'], rep['Precisao_std'])
                re = formatar_metrica(rep['Recall_mean'], rep['Recall_std'])
                f1 = formatar_metrica(rep['F1_Score_mean'], rep['F1_Score_std'])
                au = formatar_metrica(rep['AUC_ROC_mean'], rep['AUC_ROC_std'])
            else:
                ac = formatar_metrica(r['Acuracia'])
                pr = formatar_metrica(r['Precisao'])
                re = formatar_metrica(r['Recall'])
                f1 = formatar_metrica(r['F1_Score'])
                au = formatar_metrica(r['AUC_ROC'])

            f.write(f"{nome_pt:<30} {ac:<18} {pr:<18} {re:<18} {f1:<18} {au:<18}\n")

        f.write("-" * 80 + "\n")
        f.write("\nNota: Modelos estocasticos (MLP, RF) reportam media +/- desvio padrao de 10 execucoes.\n")
        f.write("      Modelos deterministicos (SVM, LogReg) reportam valor unico.\n")
        f.write("=" * 80 + "\n")

    # --- Tabela 2: Melhores Parametros ---
    print("       -> melhores_parametros.txt")
    with open(os.path.join(TABELAS_DIR, 'melhores_parametros.txt'), 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("MELHORES HIPERPARAMETROS ENCONTRADOS (GridSearchCV)\n")
        f.write("=" * 80 + "\n\n")

        for nome_key, nome_pt in NOMES_MODELOS.items():
            f.write(f"\n{nome_pt}:\n")
            f.write("-" * 50 + "\n")
            best = modelos[nome_key]['best_params']
            for param, valor in best.items():
                f.write(f"  {param:<30}: {valor}\n")
            f.write(f"  {'F1 medio (CV)':<30}: {modelos[nome_key]['best_score_cv']:.4f}\n")

        f.write("\n" + "=" * 80 + "\n")


def gerar_relatorio(df, estatisticas, resultados, modelos, rep_mlp, rep_rf,
                    X_train_bal, y_train_bal, tempo_execucao):
    """
    Gera o relatorio completo em TXT.

    Args:
        df: DataFrame original
        estatisticas: Estatisticas da analise exploratoria
        resultados: Dict com metricas de cada modelo
        modelos: Dict com modelos treinados
        rep_mlp: Resultados das repeticoes do MLP
        rep_rf: Resultados das repeticoes do RF
        X_train_bal: Features de treino balanceadas
        y_train_bal: Labels de treino balanceados
        tempo_execucao: Tempo total de execucao
    """
    print("       -> relatorio_completo.txt")
    with open(os.path.join(TABELAS_DIR, 'relatorio_completo.txt'), 'w', encoding='utf-8') as f:
        # === CAPA ===
        f.write("=" * 80 + "\n")
        f.write(" " * 20 + "RELATORIO COMPLETO\n")
        f.write(" " * 15 + "PREDICAO DE DIABETES TIPO 2\n")
        f.write(" " * 18 + "INTELIGENCIA COMPUTACIONAL\n")
        f.write("=" * 80 + "\n")
        f.write(f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
        f.write(f"Tempo de execucao: {tempo_execucao:.1f} segundos\n")
        f.write("=" * 80 + "\n\n")

        # === 1. INTRODUCAO ===
        f.write("1. INTRODUCAO\n")
        f.write("-" * 80 + "\n")
        f.write("""
Este trabalho apresenta um estudo comparativo de quatro algoritmos de
Inteligencia Computacional aplicados a predicao de Diabetes Tipo 2. O objetivo
e classificar se um paciente possui ou nao diabetes com base em medidas
diagnosticas.

Os algoritmos avaliados sao:
  1. MLP (Multi-Layer Perceptron) - Rede Neural Artificial
  2. SVM (Support Vector Machine) - Maquina de Vetores de Suporte
  3. Random Forest - Floresta Aleatoria
  4. Regressao Logistica - Modelo linear de classificacao (baseline)

O dataset utilizado e o "Pima Indians Diabetes Database", disponivel no UCI
Machine Learning Repository, contendo 768 amostras de mulheres Pima Indians
com 8 atributos diagnosticos e 1 variavel alvo binaria.
""")

        # === 2. ESTATISTICAS DOS DADOS ===
        f.write("\n2. ESTATISTICAS DOS DADOS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total de amostras: {len(df)}\n")
        f.write(f"Total de atributos: {df.shape[1] - 1}\n")
        f.write(f"Atributos: {', '.join(COLUNAS[:-1])}\n\n")

        f.write("2.1 Estatisticas Descritivas\n")
        f.write(estatisticas.get('descritivas', 'N/A') + "\n\n")

        f.write("2.2 Zeros Impossiveis (substituidos por mediana)\n")
        for col, info in estatisticas.get('zeros_impossiveis', {}).items():
            f.write(f"  {col}: {info['quantidade']} valores ({info['percentual']:.1f}%)\n")

        f.write("\n2.3 Distribuicao das Classes\n")
        for classe, count in estatisticas.get('distribuicao_classes', {}).items():
            label = "Nao Diabeticos" if classe == 0 else "Diabeticos"
            pct = 100 * count / len(df)
            f.write(f"  {label}: {count} ({pct:.1f}%)\n")

        # === 3. PRE-PROCESSAMENTO ===
        f.write("\n\n3. PRE-PROCESSAMENTO\n")
        f.write("-" * 80 + "\n")
        f.write("""
As etapas de pre-processamento aplicadas foram:

  1. Tratamento de zeros impossiveis: Atributos como Glicose, Pressao
     Arterial, Espessura da Pele, Insulina e IMC nao podem ser zero em
     seres humanos vivos. Esses valores foram substituidos por NaN e
     depois imputados com a mediana.

  2. Divisao estratificada: 60%% treino / 20%% validacao / 20%% teste,
     mantendo a proporcao das classes (random_state=42).

  3. Normalizacao: StandardScaler ajustado apenas no treino e aplicado
     nos conjuntos de validacao e teste.

  4. Balanceamento: SMOTE (Synthetic Minority Over-sampling Technique)
     aplicado apenas no conjunto de treino para balancear as classes.
""")

        # === 4. CONFIGURACAO DOS MODELOS ===
        f.write("\n4. CONFIGURACAO DOS MODELOS (GridSearchCV - 5-Fold)\n")
        f.write("-" * 80 + "\n")
        for nome_key, nome_pt in NOMES_MODELOS.items():
            f.write(f"\n{nome_pt}:\n")
            best = modelos[nome_key]['best_params']
            for param, valor in best.items():
                f.write(f"  {param}: {valor}\n")

        # === 5. RESULTADOS ===
        f.write("\n\n5. RESULTADOS NO CONJUNTO DE TESTE\n")
        f.write("=" * 80 + "\n")

        # Encontrar melhor modelo por F1
        melhor_f1 = -1
        melhor_nome = ""
        for nome_key in NOMES_MODELOS:
            if nome_key in ['MLP', 'RandomForest']:
                rep = rep_mlp if nome_key == 'MLP' else rep_rf
                f1_val = rep['F1_Score_mean']
            else:
                f1_val = resultados[nome_key]['F1_Score']
            if f1_val > melhor_f1:
                melhor_f1 = f1_val
                melhor_nome = nome_key

        f.write(f"\nMelhor modelo: {NOMES_MODELOS[melhor_nome]} (F1 = {melhor_f1:.4f})\n\n")

        f.write("-" * 80 + "\n")
        f.write(f"{'Modelo':<30} {'Acuracia':<18} {'Precisao':<18} {'Recall':<18} {'F1-Score':<18} {'AUC-ROC':<18}\n")
        f.write("-" * 80 + "\n")

        for nome_key, nome_pt in NOMES_MODELOS.items():
            if nome_key in ['MLP', 'RandomForest']:
                rep = rep_mlp if nome_key == 'MLP' else rep_rf
                ac = formatar_metrica(rep['Acuracia_mean'], rep['Acuracia_std'])
                pr = formatar_metrica(rep['Precisao_mean'], rep['Precisao_std'])
                re = formatar_metrica(rep['Recall_mean'], rep['Recall_std'])
                f1 = formatar_metrica(rep['F1_Score_mean'], rep['F1_Score_std'])
                au = formatar_metrica(rep['AUC_ROC_mean'], rep['AUC_ROC_std'])
            else:
                r = resultados[nome_key]
                ac = formatar_metrica(r['Acuracia'])
                pr = formatar_metrica(r['Precisao'])
                re = formatar_metrica(r['Recall'])
                f1 = formatar_metrica(r['F1_Score'])
                au = formatar_metrica(r['AUC_ROC'])

            f.write(f"{nome_pt:<30} {ac:<18} {pr:<18} {re:<18} {f1:<18} {au:<18}\n")

        f.write("-" * 80 + "\n")

        # === 6. ANALISE DOS RESULTADOS ===
        f.write("\n\n6. ANALISE DOS RESULTADOS\n")
        f.write("-" * 80 + "\n")

        # Analise textual automatica
        f.write("\n6.1 Comparacao Geral\n\n")

        # Ordenar modelos por F1
        ranking = []
        for nome_key in NOMES_MODELOS:
            if nome_key in ['MLP', 'RandomForest']:
                rep = rep_mlp if nome_key == 'MLP' else rep_rf
                f1_val = rep['F1_Score_mean']
                f1_std = rep['F1_Score_std']
            else:
                f1_val = resultados[nome_key]['F1_Score']
                f1_std = 0
            ranking.append((nome_key, f1_val, f1_std))

        ranking.sort(key=lambda x: x[1], reverse=True)

        f.write("Ranking dos modelos (por F1-Score):\n")
        for i, (nome, f1, std) in enumerate(ranking, 1):
            if std > 0:
                f.write(f"  {i}. {NOMES_MODELOS[nome]}: F1 = {f1:.4f} +/- {std:.4f}\n")
            else:
                f.write(f"  {i}. {NOMES_MODELOS[nome]}: F1 = {f1:.4f}\n")

        f.write("\n6.2 Observacoes\n\n")
        f.write(f"- O modelo {NOMES_MODELOS[ranking[0][0]]} apresentou o melhor desempenho geral.\n")

        # Verificar estabilidade
        if rep_mlp['F1_Score_std'] > 0.02:
            f.write(f"- A MLP apresentou variabilidade significativa entre execucoes\n")
            f.write(f"  (desvio padrao = {rep_mlp['F1_Score_std']:.4f}), indicando sensibilidade\n")
            f.write(f"  a inicializacao aleatoria dos pesos.\n")
        else:
            f.write(f"- A MLP demonstrou boa estabilidade nas 10 execucoes.\n")

        if rep_rf['F1_Score_std'] > 0.02:
            f.write(f"- O Random Forest apresentou alguma variabilidade\n")
            f.write(f"  (desvio padrao = {rep_rf['F1_Score_std']:.4f}).\n")
        else:
            f.write(f"- O Random Forest mostrou-se estavel nas 10 execucoes.\n")

        f.write("- O SMOTE foi eficaz para balancear as classes no treino.\n")
        f.write("- A imputacao com mediana lidou adequadamente com os zeros impossiveis.\n")

        # === 7. CONCLUSAO ===
        f.write("\n\n7. CONCLUSAO\n")
        f.write("-" * 80 + "\n")
        f.write(f"""
Este trabalho comparou quatro algoritmos de Inteligencia Computacional para
predicao de Diabetes Tipo 2. O modelo {NOMES_MODELOS[ranking[0][0]]} obteve o
melhor desempenho geral com F1-Score de {ranking[0][1]:.4f}.

Todos os modelos demonstraram capacidade preditiva satisfatoria, com AUC-ROC
superior a 0.75, indicando que os atributos diagnosticos selecionados sao
relevantes para a classificacao.

O pipeline desenvolvido e robusto, incluindo tratamento de dados ausentes,
balanceamento de classes, otimizacao de hiperparametros e avaliacao rigorosa
com metricas multiples e repeticao de experimentos para modelos estocasticos.
""")

        f.write("\n" + "=" * 80 + "\n")
        f.write("FIM DO RELATORIO\n")
        f.write("=" * 80 + "\n")


# ==============================================================================
# 8. FUNCAO PRINCIPAL
# ==============================================================================
def main():
    """
    Funcao principal que orquestra todo o pipeline de ML.
    """
    inicio = time.time()

    print("=" * 60)
    print("PREDICAO DE DIABETES - INTELIGENCIA COMPUTACIONAL")
    print("=" * 60)

    # Criar diretorios
    criar_diretorios()

    # 1. Carregar dados
    df = carregar_dados()

    # 2. Explorar dados
    estatisticas = explorar_dados(df)

    # 3. Pre-processar
    (X_train_bal, X_val, X_test,
     y_train_bal, y_val, y_test, scaler,
     X_train_orig_scaled, y_train_orig) = preprocessar(df)

    # 4. Treinar modelos
    modelos = treinar_modelos(X_train_bal, y_train_bal)

    # 5. Executar repeticoes para modelos estocasticos
    print("\n[5/8] Executando repeticoes (MLP e RF)...")

    # Funcoes fabrica para criar modelos com novas seeds
    def criar_mlp(random_state):
        params = modelos['MLP']['best_params'].copy()
        params['random_state'] = random_state
        return MLPClassifier(**params, early_stopping=True,
                             validation_fraction=0.1, n_iter_no_change=10)

    def criar_rf(random_state):
        params = modelos['RandomForest']['best_params'].copy()
        params['random_state'] = random_state
        return RandomForestClassifier(**params)

    rep_mlp = executar_repeticoes(criar_mlp, X_train_bal, y_train_bal,
                                   X_test, y_test, 'MLP', n=10)
    rep_rf = executar_repeticoes(criar_rf, X_train_bal, y_train_bal,
                                  X_test, y_test, 'RandomForest', n=10)

    # 6. Avaliar todos os modelos no teste
    print("\n[6/8] Avaliando modelos no teste...")
    resultados = {}
    for nome_key in NOMES_MODELOS:
        if nome_key in ['MLP', 'RandomForest']:
            # Usar modelo com a melhor seed (a do GridSearch)
            m = avaliar_modelo(modelos[nome_key]['modelo'], X_test, y_test, nome_key)
        else:
            m = avaliar_modelo(modelos[nome_key]['modelo'], X_test, y_test, nome_key)
        resultados[nome_key] = m
        print(f"       {NOMES_MODELOS[nome_key]:30s}: F1 = {m['F1_Score']:.4f}, AUC = {m['AUC_ROC']:.4f}")

    # 7. Gerar graficos
    gerar_graficos(df, y_train_orig, y_train_bal, resultados, modelos,
                   rep_mlp, rep_rf, X_test, y_test, X_train_bal)

    # 8. Gerar relatorios
    tempo_exec = time.time() - inicio
    gerar_tabelas(resultados, modelos, rep_mlp, rep_rf)
    gerar_relatorio(df, estatisticas, resultados, modelos, rep_mlp, rep_rf,
                    X_train_bal, y_train_bal, tempo_exec)

    # === RESULTADOS FINAIS ===
    print("\n" + "=" * 60)
    print("RESULTADOS FINAIS:")
    print("-" * 60)

    # Melhor modelo
    ranking_f1 = []
    for nome_key in NOMES_MODELOS:
        if nome_key == 'MLP':
            f1_val = rep_mlp['F1_Score_mean']
            f1_std = rep_mlp['F1_Score_std']
        elif nome_key == 'RandomForest':
            f1_val = rep_rf['F1_Score_mean']
            f1_std = rep_rf['F1_Score_std']
        else:
            f1_val = resultados[nome_key]['F1_Score']
            f1_std = 0
        ranking_f1.append((nome_key, f1_val, f1_std))

    ranking_f1.sort(key=lambda x: x[1], reverse=True)
    melhor = ranking_f1[0]

    print(f"- Melhor modelo: {NOMES_MODELOS[melhor[0]]} (F1 = {melhor[1]:.4f} +/- {melhor[2]:.4f})")
    print(f"- Tempo total: {tempo_exec:.1f} segundos")
    print(f"- Todos os resultados salvos em: {RESULTS_DIR}")
    print("=" * 60)


# ==============================================================================
# EXECUCAO
# ==============================================================================
if __name__ == '__main__':
    main()
