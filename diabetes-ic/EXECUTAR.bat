@echo off
chcp 65001 >nul
title Predicao de Diabetes - Inteligencia Computacional

echo ================================================
echo  PREDICAO DE DIABETES TIPO 2
echo  Inteligencia Computacional - CEFET-MG
echo ================================================
echo.
echo Iniciando execucao completa...
echo.

:: Verifica se Python esta instalado
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERRO] Python nao encontrado!
    echo Por favor, instale o Anaconda ou Python 3.9+.
    echo.
    pause
    exit /b 1
)

echo [OK] Python encontrado.
echo.

:: Navega para a pasta src
cd /d "%~dp0src"

:: Executa o script Python
echo Executando pipeline completo...
echo Isso pode levar 1-2 minutos.
echo.
python main.py

echo.
echo ================================================
if %errorlevel% == 0 (
    echo [SUCESSO] Execucao concluida!
    echo.
    echo Abrindo pasta de resultados...
    start "" "%~dp0results"
) else (
    echo [ERRO] Execucao falou. Verifique as mensagens acima.
    echo.
    pause
)

echo ================================================
pause
