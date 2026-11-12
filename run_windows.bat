@echo off
cd /d "%~dp0"
if exist .venv\Scripts\activate.bat goto run

echo Creating virtual environment - first run only, takes a few minutes...
python -m venv .venv
if errorlevel 1 goto nopython
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
goto start

:run
call .venv\Scripts\activate.bat

:start
python app.py
pause
exit /b 0

:nopython
echo Python was not found. Install Python 3.10 or newer from python.org and tick "Add python.exe to PATH".
pause
exit /b 1
