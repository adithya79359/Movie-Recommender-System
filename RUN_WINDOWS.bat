@echo off
cd /d "%~dp0"
python -m pip install -r requirements-runtime.txt
if errorlevel 1 goto error
echo Open http://127.0.0.1:5000 in your browser.
python app.py
goto end
:error
echo Installation failed. Check Python and internet access.
pause
:end
