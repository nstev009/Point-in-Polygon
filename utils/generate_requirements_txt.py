import subprocess
import os

# Step 1: Activate your Python environment if necessary

# Step 2: Generate the initial requirements.txt file
subprocess.call(['pip', 'freeze', '>', 'temp_requirements.txt'], shell=True)

# Step 3: Read the requirements.txt file
with open('temp_requirements.txt', 'r') as file:
    lines = file.readlines()

# Step 4: Specify the packages to exclude
packages_to_exclude = ['spyder', 'spyder-terminal', 'spyder-kernels']

# Step 5: Remove lines corresponding to excluded packages
filtered_lines = [line for line in lines if not any(package in line for package in packages_to_exclude)]

# Step 6: Write the modified requirements.txt file in the current directory
requirements_file = os.path.join(os.getcwd(), 'requirements.txt')
with open(requirements_file, 'w') as file:
    file.writelines(filtered_lines)

# Step 7: Remove the temporary file
os.remove('temp_requirements.txt')

print(f"requirements.txt file has been created with excluded packages.\nPath to file: {requirements_file}")
