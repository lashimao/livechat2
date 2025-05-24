@echo OFF
REM Script to install dependencies and start the Amadeus-like AI system

ECHO Starting Backend Service...
cd service/webrtc

REM Check if virtual environment exists, if not create it
IF NOT EXIST venv (
    ECHO Creating Python virtual environment in service/webrtc/venv...
    python -m venv venv
    IF ERRORLEVEL 1 (
        ECHO Failed to create Python virtual environment. Please ensure Python is installed and accessible.
        PAUSE
        EXIT /B 1
    )
)

REM Activate virtual environment
ECHO Activating Python virtual environment...
CALL .\venv\Scripts\activate

REM Install Python dependencies
ECHO Installing Python dependencies from requirements.txt...
pip install -r requirements.txt
IF ERRORLEVEL 1 (
    ECHO Failed to install Python dependencies.
    PAUSE
    EXIT /B 1
)

REM Start Python FastAPI backend server
ECHO Starting FastAPI backend server (will run in a new window)...
REM Using start "Backend Server" to run it in a new non-blocking window.
start "Backend Server" python server.py
IF ERRORLEVEL 1 (
    ECHO Failed to start backend server.
    PAUSE
    EXIT /B 1
)

REM Go back to the root directory
cd ..\..

ECHO Starting Frontend Application...

REM Install Node.js dependencies
ECHO Installing Node.js dependencies from package.json...
npm install
IF ERRORLEVEL 1 (
    ECHO Failed to install Node.js dependencies. Please ensure Node.js and npm are installed and accessible.
    PAUSE
    EXIT /B 1
)

REM Start the Electron application (adjust this command if needed - check package.json)
ECHO Starting Electron frontend...
npm run electron:dev
IF ERRORLEVEL 1 (
    ECHO Failed to start Electron frontend.
    PAUSE
    EXIT /B 1
)

ECHO Both backend and frontend startup processes have been initiated.
PAUSE
