import os, sys, subprocess

# Run commands and identify errors
def run_command(command):
    try:
        subprocess.run(command, check=True, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Error encountered: {e}")
        sys.exit(1)

def setup():
    print("** Starting environment setup **")

    # Create the virtual environment
    if not os.path.exists(".venv"):
        print("** Creating virtual environment **")
        run_command(f"{sys.executable} -m venv .venv")
    else:
        print("** Virtual environment exists, skipping creation **")

    # Determine path based on os
    if os.name == 'nt':  # windows
        pip_path = os.path.join(".venv", "Scripts", "pip.exe")
        python_path = os.path.join(".venv", "Scripts", "python.exe")
    else:  # macOS / linux
        pip_path = os.path.join(".venv", "bin", "pip")
        python_path = os.path.join(".venv", "bin", "python")

    # Upgrade pip
    print("** Updating pip **")
    run_command(f"{python_path} -m pip install --upgrade pip --quiet")

    # Install requirements.txt
    if os.path.exists("requirements.txt"):
        print("** Installing dependencies from requirements.txt **")
        run_command(f"{pip_path} install -r requirements.txt --quiet")
        print("** Dependencies installed successfully **")
    else:
        print("** No requirements.txt found **")

    print("\n** Environment set up complete **")
    activate = ""
    if os.name == 'nt':
        activate = ".\\.venv\\Scripts\\activate"
    else:
        activate = "source .venv/bin/activate"

    print(f"To activate your environment, use: {activate}\n")

if __name__ == "__main__":
    setup()