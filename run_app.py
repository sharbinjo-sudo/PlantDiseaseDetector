import subprocess
import time
import os
import signal
import sys

# change when your folder names change
DJANGO_PATH = r"C:\Users\sharb\OneDrive\Desktop\FullPlantDiseaseDetector\plant_api"
FLUTTER_EXE = r"C:\Users\sharb\OneDrive\Desktop\FullPlantDiseaseDetector\plant_disease\build\windows\x64\runner\Release\plant_disease.exe"

def run_django():
    """Start Django development server."""
    print("🚀 Starting Django backend...")
    return subprocess.Popen(
        ["python", "manage.py", "runserver", "127.0.0.1:8000"],
        cwd=DJANGO_PATH,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NEW_CONSOLE
    )

def run_flutter():
    """Launch Flutter desktop app."""
    print("🌿 Launching Flutter app...")
    return subprocess.Popen(
        [FLUTTER_EXE],
        cwd=os.path.dirname(FLUTTER_EXE),
        creationflags=subprocess.CREATE_NEW_CONSOLE
    )

def main():
    django_process = run_django()
    time.sleep(5)  # Give Django time to start

    flutter_process = run_flutter()

    try:
        flutter_process.wait()  # Wait until Flutter closes
    except KeyboardInterrupt:
        print("\n🛑 Interrupted. Closing apps...")

    # Stop Django
    if django_process.poll() is None:
        print("🔻 Stopping Django server...")
        os.kill(django_process.pid, signal.SIGTERM)

    print("✅ All processes closed. Bye!")

if __name__ == "__main__":
    main()
