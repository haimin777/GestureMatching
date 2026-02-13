# 🖐️ GestureMatching Project

This repository contains tools for training and evaluating solutions for
**gesture recognition**, with support for **FC-net architectures**.

------------------------------------------------------------------------

## 🚀 Setup Instructions

### **Prerequisites**

-   **Python**: \>= 3.12\
    *(Required for modern ML dependencies like JAX / TensorFlow)*

-   **Poetry**: Latest version installed on your Windows PC

------------------------------------------------------------------------

## ⚙️ Environment Initialization

To set up a deterministic environment and avoid dependency conflicts,
run:

``` powershell
# Initialize the project and install all dependencies
poetry install
```

------------------------------------------------------------------------

## ▶️ Activating the Environment

Poetry 2.0+ provides a clean way to activate your virtual environment in
PowerShell:

``` powershell
# Activate the environment in your current session
Invoke-Expression (poetry env activate)
```

To leave the environment later:

``` powershell
deactivate
```

------------------------------------------------------------------------

## 🛠️ Usage

You can run project scripts either by activating the environment or by
using the `poetry run` prefix.

### **Run Inference**

To run live gesture matching inference:

``` powershell
poetry run python inference.py
```

------------------------------------------------------------------------

### **Run Evaluation**

To evaluate model performance and solution accuracy:

``` powershell
poetry run python evaluation.py
```

------------------------------------------------------------------------

## 📦 Dependency Management

If you need to manage packages:

-   **Add a package**

``` powershell
poetry add <package_name>
```

-   **Update all packages**

``` powershell
poetry update
```

-   **Check for conflicts**

``` powershell
poetry check
```

------------------------------------------------------------------------

✅ Generated for the **GestureMatching Environment**
