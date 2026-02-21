import subprocess
import time
import os
import signal
import sys

# ===== ROOT FOLDER (where run_app.py is located) =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ===== AUTO PATHS =====
DJANGO_PATH = os.path.join(BASE_DIR, "plant_api")

FLUTTER_EXE = os.path.join(
    BASE_DIR,
    "plant_disease",
    "build",
    "windows",
    "x64",
    "runner",
    "Release",
    "plant_disease.exe"   # change only if your exe name is different
)

def run_django():
    print("🚀 Starting Django backend...")
    return subprocess.Popen(
        ["python", "manage.py", "runserver", "127.0.0.1:8000"],
        cwd=DJANGO_PATH,
        creationflags=subprocess.CREATE_NEW_CONSOLE
    )

def run_flutter():
    print("🌿 Launching Flutter app...")
    return subprocess.Popen(
        [FLUTTER_EXE],
        cwd=os.path.dirname(FLUTTER_EXE),
        creationflags=subprocess.CREATE_NEW_CONSOLE
    )

def main():

    # Safety checks
    if not os.path.exists(DJANGO_PATH):
        print("❌ plant_api folder not found.")
        sys.exit(1)

    if not os.path.exists(FLUTTER_EXE):
        print("❌ Flutter exe not found. Run 'flutter build windows --release' first.")
        sys.exit(1)

    django_process = run_django()
    time.sleep(4)  # Give Django time to start

    flutter_process = run_flutter()

    try:
        flutter_process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Interrupted. Closing apps...")

    if django_process.poll() is None:
        print("🔻 Stopping Django server...")
        django_process.terminate()

    print("✅ All processes closed.")

if __name__ == "__main__":
    main()
