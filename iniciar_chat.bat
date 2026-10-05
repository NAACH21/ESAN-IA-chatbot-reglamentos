@echo off
echo Iniciando Asistente Normativo ESAN...
cd /d "%~dp0"
python -m streamlit run app.py
pause
