import subprocess, sys, os
from pathlib import Path

os.chdir(Path(__file__).parent)

print("\nSynthMed - Launch options")
print("1. User site  - http://localhost:8501")
print("2. Admin site - http://localhost:8502")
choice = input("\nEnter 1 or 2: ").strip()

if choice == "1":
    subprocess.run([sys.executable, "-m", "streamlit", "run",
                    "user_app.py", "--server.port", "8501"])
elif choice == "2":
    subprocess.run([sys.executable, "-m", "streamlit", "run",
                    "admin_app.py", "--server.port", "8502"])
else:
    print("Invalid choice.")