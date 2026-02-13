🖐️ GestureMatching Project
This repository contains tools for training and evaluating solutions for gesture recognition, with support for FC-net architectures.

🚀 Setup Instructions
1. Prerequisites
Python: >= 3.12 (Required for modern ML dependencies like JAX/TensorFlow)

Poetry: Latest version installed on your Windows PC.

2. Environment Initialization
To set up the deterministic environment and avoid version conflicts, run:

PowerShell

# Initialize the project and install all dependencies
poetry install
3. Activating the Environment
Poetry 2.0+ provides a clean way to enter your virtual environment in PowerShell:

PowerShell

# Activate the environment in your current session
Invoke-Expression (poetry env activate)
To leave the environment later, simply type deactivate.

🛠️ Usage
You can run the project scripts either by activating the environment (above) or by using the poetry run prefix.

Run Inference
To run the live gesture matching inference:

PowerShell

poetry run python inference.py
Run Evaluation
To evaluate the model performance and solution accuracy:

PowerShell

poetry run python evaluation.py
📦 Dependency Management
If you need to manage packages in the project:

Add a package: poetry add <package-name>

Update all packages: poetry update

Check for conflicts: poetry check

Generated for the GestureMatching Environment.