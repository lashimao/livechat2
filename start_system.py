import os 
import subprocess 
import sys 
import platform 
import time 

# --- Configuration --- 
BACKEND_DIR = os.path.join(os.getcwd(), 'service', 'webrtc') 
FRONTEND_DIR = os.getcwd() 
PYTHON_VENV_DIR = os.path.join(BACKEND_DIR, 'venv') 
running_processes = [] # Global list to store subprocesses

# --- Helper Functions --- 
def print_message(message, level='INFO'): 
    print(f'[{level}] {message}') 

def run_command(command, cwd=None, shell=True, check=False, env=None): 
    """Runs a shell command.""" 
    print_message(f'Running command: "{" ".join(command) if isinstance(command, list) else command}" in {cwd or os.getcwd()}', 'CMD') 
    try: 
        process = subprocess.Popen(command, cwd=cwd, shell=shell, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True) 
        # Stream stdout and stderr 
        if process.stdout: 
            for stdout_line in iter(process.stdout.readline, ""): 
                if stdout_line.strip(): 
                    print_message(stdout_line.strip(), 'STDOUT') 
        if process.stderr: 
            for stderr_line in iter(process.stderr.readline, ""): 
                if stderr_line.strip(): 
                    print_message(stderr_line.strip(), 'STDERR') 
        process.stdout.close() 
        process.stderr.close() 
        return_code = process.wait() 
        if check and return_code != 0: 
            print_message(f'Command failed with exit code {return_code}: {" ".join(command) if isinstance(command, list) else command}', 'ERROR') 
            sys.exit(return_code) 
        return return_code 
    except FileNotFoundError: 
        print_message(f'Error: Command not found - {command[0] if isinstance(command, list) else command.split()[0]}. Please ensure it is installed and in your PATH.', 'ERROR') 
        sys.exit(1) 
    except Exception as e: 
        print_message(f'An error occurred while running command: {" ".join(command) if isinstance(command, list) else command}. Error: {e}', 'ERROR') 
        sys.exit(1) 

# --- Placeholder Functions --- 
def check_ffmpeg(): 
    print_message('Checking for FFmpeg in system PATH...') 
    try: 
        # Try to run ffmpeg -version. 
        # stdout and stderr are captured to prevent console clutter from the command itself. 
        process = subprocess.Popen(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True) 
        stdout, stderr = process.communicate(timeout=5) # Added timeout 
        if process.returncode == 0: 
            print_message('FFmpeg found.', 'INFO') 
            # Optionally print version: 
            # version_line = stdout.splitlines()[0] if stdout else "ffmpeg version not parsable" 
            # print_message(f'  {version_line}', 'INFO') 
        else: 
            # This case might not be hit if FileNotFoundError is raised first on some OS 
            print_message('WARNING: ffmpeg command ran with an error (might indicate an issue).', 'WARNING') 
            print_message('  Please ensure FFmpeg is correctly installed and in your PATH.', 'WARNING') 
            print_message('  Download from: https://ffmpeg.org/download.html', 'WARNING') 
            # No sys.exit here, allow script to continue but warn user 
    except FileNotFoundError: 
        print_message('ERROR: ffmpeg not found in PATH.', 'ERROR') 
        print_message('  This is required for audio processing by `pydub` (likely used by `fastrtc`).', 'ERROR') 
        print_message('  Please install FFmpeg and add its bin directory to your system PATH.', 'ERROR') 
        print_message('  Download from: https://ffmpeg.org/download.html', 'ERROR') 
        print_message('Exiting due to missing FFmpeg.', 'ERROR') 
        sys.exit(1) # Exit if FFmpeg is critical 
    except subprocess.TimeoutExpired: 
        print_message('WARNING: ffmpeg -version command timed out.', 'WARNING') 
        print_message('  This might indicate an issue with your FFmpeg installation.', 'WARNING') 
    except Exception as e: 
        print_message(f'An unexpected error occurred while checking for FFmpeg: {e}', 'ERROR') 
        print_message('  Script will continue, but audio processing might fail.', 'WARNING') 
    print_message('FFmpeg check finished.', 'INFO') 
    print_message('-' * 30, 'SEPARATOR') # Separator line 

def setup_and_start_backend(python_venv_executable): 
    print_message('--- Backend Setup & Start ---') 
    original_cwd = os.getcwd() 
    os.chdir(BACKEND_DIR) 
    print_message(f'Changed directory to: {os.getcwd()}', 'DEBUG') 

    # 1. Check/Create Virtual Environment 
    if not os.path.isdir(PYTHON_VENV_DIR): 
        print_message(f'Python virtual environment not found at {PYTHON_VENV_DIR}. Creating...', 'INFO') 
        # Use the system's default python to create venv 
        # Ensure 'python' or 'python3' is in PATH for this to work 
        py_exe_for_venv = sys.executable # Or 'python3' or 'python' if sys.executable is not desired 
        run_command([py_exe_for_venv, '-m', 'venv', 'venv'], cwd=BACKEND_DIR, check=True) 
        print_message('Virtual environment created.', 'INFO') 
    else: 
        print_message('Python virtual environment found.', 'INFO') 

    # 2. Determine pip path within venv 
    if platform.system() == "Windows": 
        pip_venv_executable = os.path.join(PYTHON_VENV_DIR, 'Scripts', 'pip.exe') 
    else: 
        pip_venv_executable = os.path.join(PYTHON_VENV_DIR, 'bin', 'pip') 

    # 3. Install Dependencies 
    requirements_file = os.path.join(BACKEND_DIR, 'requirements.txt') 
    if os.path.isfile(requirements_file): 
        print_message(f'Installing backend dependencies from {requirements_file}...', 'INFO') 
        run_command([pip_venv_executable, 'install', '-r', requirements_file], cwd=BACKEND_DIR, check=True) 
    else: 
        print_message(f'requirements.txt not found in {BACKEND_DIR}. Skipping pip install.', 'WARNING') 

    # 4. Launch Backend Server 
    server_file = os.path.join(BACKEND_DIR, 'server.py') 
    if os.path.isfile(server_file): 
        print_message(f'Starting backend server ({server_file}) in a new process...', 'INFO') 
        # For Windows, using 'start' with CREATE_NEW_CONSOLE flag to detach 
        # For other OS, Popen without wait should run it in background 
        if platform.system() == "Windows": 
            # Using CREATE_NEW_CONSOLE to run in a new window, somewhat detached. 
            # For truly detached, other methods like pythonw or specific library might be needed. 
            backend_p = subprocess.Popen([python_venv_executable, server_file], cwd=BACKEND_DIR, creationflags=subprocess.CREATE_NEW_CONSOLE) 
            running_processes.append(backend_p)
        else: 
            # On Linux/macOS, Popen usually detaches if not waited on. 
            # Stderr/Stdout can be DEVNULL if no output is desired from this script. 
            backend_p = subprocess.Popen([python_venv_executable, server_file], cwd=BACKEND_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) 
            running_processes.append(backend_p)
        print_message('Backend server process initiated.', 'INFO') 
    else: 
        print_message(f'server.py not found in {BACKEND_DIR}. Cannot start backend.', 'ERROR') 
        sys.exit(1) 

    os.chdir(original_cwd) # Go back to original directory 
    print_message(f'Changed directory back to: {os.getcwd()}', 'DEBUG') 
    print_message('--- Backend Setup & Start Finished ---', 'INFO') 
    print_message('-' * 30, 'SEPARATOR') 

def setup_and_start_frontend(): 
    print_message('--- Frontend Setup & Start ---') 
    original_cwd = os.getcwd() 
    # Ensure we are in the frontend directory (project root) 
    if os.getcwd() != FRONTEND_DIR: 
        os.chdir(FRONTEND_DIR) 
        print_message(f'Changed directory to: {os.getcwd()}', 'DEBUG') 

    # 1. Install Node.js Dependencies 
    package_json_file = os.path.join(FRONTEND_DIR, 'package.json') 
    if os.path.isfile(package_json_file): 
        print_message(f'Installing frontend dependencies from {package_json_file} (if needed via npm install)...', 'INFO') 
        # Using shell=True for npm commands is often more reliable on Windows 
        # Some systems might have 'npm.cmd' 
        npm_command = 'npm.cmd' if platform.system() == "Windows" else 'npm' 
        run_command([npm_command, 'install'], cwd=FRONTEND_DIR, check=True) 
    else: 
        print_message(f'package.json not found in {FRONTEND_DIR}. Cannot install frontend dependencies.', 'ERROR') 
        sys.exit(1) 

    # 2. Launch Frontend Application 
    print_message('Starting frontend application (npm run electron:dev)...', 'INFO') 
    # This will typically open a new window for the Electron app. 
    # We'll run this in a way that tries not to block the Python script, 
    # but its behavior (blocking or detaching) can depend on how 'npm run' handles subprocesses. 
    if platform.system() == "Windows": 
        # Using 'start' can help run it in a new process group, but npm might still attach. 
        # For a truly detached process, more complex handling or external tools are sometimes needed. 
        # Using Popen directly for better control if 'start' is problematic. 
        frontend_p = subprocess.Popen(['npm.cmd', 'run', 'electron:dev'], cwd=FRONTEND_DIR, shell=True, creationflags=subprocess.CREATE_NEW_CONSOLE) 
        running_processes.append(frontend_p)
    else: 
        frontend_p = subprocess.Popen(['npm', 'run', 'electron:dev'], cwd=FRONTEND_DIR, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) 
        running_processes.append(frontend_p)
    print_message('Frontend application process initiated.', 'INFO') 
    
    # If we changed directory, change back (though for frontend this might be the last step anyway) 
    if os.getcwd() != original_cwd: 
        os.chdir(original_cwd) 
        print_message(f'Changed directory back to: {os.getcwd()}', 'DEBUG') 
    print_message('--- Frontend Setup & Start Finished ---', 'INFO') 
    print_message('-' * 30, 'SEPARATOR') 

# --- Main Orchestration --- 
def main(): 
    print_message('Starting the Amadeus AI System setup...') 

    if platform.system() != "Windows": 
        print_message('This script is primarily designed for Windows. You might need to adapt it for other OS.', 'WARNING') 

    check_ffmpeg() 

    if platform.system() == "Windows": 
        python_venv_executable = os.path.join(PYTHON_VENV_DIR, 'Scripts', 'python.exe') 
    else: 
        # This script is Windows-focused, but for completeness if adapted: 
        python_venv_executable = os.path.join(PYTHON_VENV_DIR, 'bin', 'python') 

    try: 
        print_message('--- Starting Backend Setup ---') 
        setup_and_start_backend(python_venv_executable) 
        print_message('Backend process initiated.') 
        print_message('Giving backend 5 seconds to initialize...', 'INFO') 
        time.sleep(5) 

        print_message('--- Starting Frontend Setup ---') 
        setup_and_start_frontend() 
        print_message('Frontend process initiated.') 
        
        print_message('\nSystem startup sequence complete.', 'INFO') 
        print_message('Backend and Frontend should be running.', 'INFO') 
        print_message('Press Ctrl+C in this window to attempt to shut down all processes.', 'IMPORTANT') 

        while True: 
            # Check if any process has exited 
            all_running = True 
            for i, p in enumerate(running_processes): 
                if p.poll() is not None: # Process has terminated 
                    print_message(f'Process {i+1} (PID: {p.pid}) has terminated with code {p.returncode}.', 'WARNING') 
                    all_running = False 
            if not all_running and running_processes: # If at least one process was started and then stopped 
                print_message('One or more main processes have terminated. Exiting main script.', 'WARNING') 
                break 
            if not running_processes: # If no processes were even started (e.g. setup failed early) 
                 print_message('No main processes were started. Exiting.', 'WARNING') 
                 break
            time.sleep(10) # Check every 10 seconds 

    except KeyboardInterrupt: 
        print_message('Ctrl+C received. Attempting to shut down initiated processes...', 'INFO') 
    except Exception as e: 
        print_message(f'An unexpected error occurred in main: {e}', 'ERROR') 
    finally: 
        print_message('--- Initiating Shutdown of Subprocesses ---', 'INFO') 
        for i, p in enumerate(running_processes): 
            if p.poll() is None: # If process is still running 
                print_message(f'Terminating process {i+1} (PID: {p.pid})...', 'INFO') 
                try: 
                    if platform.system() == "Windows": 
                        # For processes started with CREATE_NEW_CONSOLE or detached, 
                        # p.terminate() might not be enough. taskkill is more forceful. 
                        subprocess.run(['taskkill', '/F', '/T', '/PID', str(p.pid)], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) 
                    else: 
                        p.terminate() # Send SIGTERM 
                    p.wait(timeout=5) # Wait for a bit 
                    print_message(f'Process {i+1} (PID: {p.pid}) terminated.', 'INFO') 
                except subprocess.TimeoutExpired: 
                    print_message(f'Process {i+1} (PID: {p.pid}) did not terminate gracefully, attempting to kill.', 'WARNING') 
                    p.kill() 
                    p.wait(timeout=5) 
                    print_message(f'Process {i+1} (PID: {p.pid}) killed.', 'INFO') 
                except Exception as e_term: 
                    print_message(f'Error terminating process {i+1} (PID: {p.pid}): {e_term}', 'ERROR') 
            else: 
                 print_message(f'Process {i+1} (PID: {p.pid}) had already terminated.', 'INFO') 
        print_message('--- Shutdown Complete ---', 'INFO') 
        print_message('Exiting startup script.') 

if __name__ == '__main__': 
    main()
