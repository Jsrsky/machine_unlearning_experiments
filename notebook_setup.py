from pathlib import Path
import sys

def setup_notebook():
    """
    Perform common notebook setup tasks like adding project root to sys.path
    and setting a random seed, while handling Kaggle, Colab, and local environments.
    """
    # Determine the environment and set the project root path accordingly
    current_path = Path.cwd()
    if 'kaggle' in str(current_path):
        # Kaggle environment
        project_root = Path('/kaggle/working/machine_unlearning_experiments')
    elif 'content' in str(current_path) and 'colab' in str(current_path):
        # Google Colab environment
        from google.colab import drive
        drive.mount('/content/drive')  # Mount Google Drive (if not already mounted)
        project_root = Path('/content/drive/MyDrive/machine_unlearning_experiments')
    else:
        # Local environment
        project_root = Path(__file__).resolve().parent

    # Add the project root to sys.path if not already present
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

    print(f"Notebook setup completed. Project root added to sys.path: {project_root}")