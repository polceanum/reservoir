# reservoir: sparsity and scaling experiments

A lightweight library for exploring the sparsity and scaling behaviors of reservoir computing.

## Installation Instructions

### Prerequisites
This library was tested with Python 3.10 (3.9+ should work) and [Miniforge](https://github.com/conda-forge/miniforge) to manage dependencies.

### Installing Miniforge

#### macOS
1. Install Miniforge using Homebrew:
    ```bash
    brew install miniforge
    ```

2. Initialize Conda for your shell (e.g., zsh):
    ```bash
    conda init zsh
    ```

3. Restart your terminal or run:
   ```bash
   source ~/.zshrc
   ```

#### Linux (Ubuntu)
1. Download the Miniforge installer:
   ```bash
   wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
   ```
2. Run the installer:
   ```bash
   bash Miniforge3-Linux-x86_64.sh
   ```
3. Follow the prompts to complete the installation.
4. Initialize Conda for your shell (e.g., `bash`):
   ```bash
   conda init bash
   ```
5. Restart your terminal or run:
   ```bash
   source ~/.bashrc
   ```

### Setting Up the Environment
1. Create a new Conda environment with Python 3.10:
   ```bash
   conda create -n reservoir-env python=3.10 -y
   ```
2. Activate the environment:
   ```bash
   conda activate reservoir-env
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
