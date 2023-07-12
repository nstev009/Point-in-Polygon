# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 13:54:58 2023

@author: wjeanph
"""
import subprocess
import os
import sys


def create_python_environment(env_name, env_path, requirements_file):
    """
    Creates a Python virtual environment and installs packages from a requirements.txt file.
    
    Usage:
    python create_environment.py <env_name> <env_path> <requirements_file>
    

    Args:
        env_name (str): Name of the environment.
        env_path (str): Path where the environment will be created.
        requirements_file (str): Path to the requirements.txt file.
        
    Note:
        use \\\ to escape \ in the path or use r'path'
    """
    # Step 1: Create a new directory for the environment
    os.makedirs(env_path, exist_ok=True)

    # Step 2: Create a new virtual environment
    subprocess.call(['python', '-m', 'venv', os.path.join(env_path, env_name)])

    # Step 3: Activate the virtual environment
    activate_script = os.path.join(env_path, env_name, 'Scripts', 'activate.bat') if os.name == 'nt' else os.path.join(env_path, env_name, 'bin', 'activate')
    subprocess.call(activate_script, shell=True)

    # Step 4: Install packages from the requirements.txt file
    subprocess.call(['pip', 'install', '-r', requirements_file])

    print("Python environment created and packages installed.")


if __name__ == '__main__':
    # Check if help flag is provided
    if len(sys.argv) == 2 and sys.argv[1] in ['-h', '--help']:
        print(create_python_environment.__doc__)
        sys.exit(0)

    # Check if all command line arguments are provided
    if len(sys.argv) != 4:
        print("Error: Invalid number of arguments. Use '-h' or '--help' for help.")
        sys.exit(1)

    # Extract command line arguments
    env_name = sys.argv[1]
    env_path = sys.argv[2]
    requirements_file = sys.argv[3]

    # Call the function to create the Python environment
    create_python_environment(env_name, env_path, requirements_file)
