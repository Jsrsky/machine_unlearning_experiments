import os
import sys

def setup_notebook():
    """
    Perform common notebook setup tasks like adding project root to sys.path
    and setting a random seed, while handling Kaggle, Colab, and local environments.

    Args:
        seed (int): Seed for random number generators.
    """
    # Determine the environment and set the project root path accordingly
    if 'kaggle' in os.getcwd():
        # Kaggle environment
        project_root = '/kaggle/working/machine_unlearning_experiments'
    elif 'content' in os.getcwd() and 'colab' in os.getcwd():
        # Google Colab environment
        from google.colab import drive
        drive.mount('/content/drive')  # Mount Google Drive (if not already mounted)
        project_root = '/content/drive/MyDrive/machine_unlearning_experiments'
    else:
        # Local environment
        project_root = os.path.abspath(os.path.dirname(__file__))

    # Add the project root to sys.path if not already present
    if project_root not in sys.path:
        sys.path.append(project_root)

    print(f"Notebook setup completed. Project root added to sys.path: {project_root}")