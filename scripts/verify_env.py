"""Task 1.1: record exact library versions and check for a GPU."""
import importlib
import importlib.metadata
import platform
import sys

print("=" * 60)
print("VisionGuard environment & library verification")
print("=" * 60)
print(f"OS platform      : {platform.platform()}")
print(f"Python version   : {sys.version.split()[0]}")
print(f"Python executable: {sys.executable}")
print("-" * 60)

for label, module, dist in [("TensorFlow", "tensorflow", "tensorflow"), ("Keras", "keras", "keras"),
                            ("Keras Tuner", "keras_tuner", "keras-tuner"), ("MLflow", "mlflow", "mlflow"),
                            ("DagsHub", "dagshub", "dagshub"), ("DVC", "dvc", "dvc"),
                            ("Scikit-Learn", "sklearn", "scikit-learn"), ("Pandas", "pandas", "pandas"),
                            ("NumPy", "numpy", "numpy"), ("Pillow", "PIL", "pillow"),
                            ("Matplotlib", "matplotlib", "matplotlib"), ("Seaborn", "seaborn", "seaborn"),
                            ("Flask", "flask", "flask"), ("SQLAlchemy", "sqlalchemy", "sqlalchemy"),
                            ("Requests", "requests", "requests")]:
    try:
        importlib.import_module(module)
        print(f"{label:<14}: {importlib.metadata.version(dist)}")
    except ImportError:
        print(f"{label:<14}: NOT INSTALLED")

try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    print("-" * 60)
    print(f"Available GPUs   : {gpus if gpus else 'None detected (running in CPU mode)'}")
except ImportError:
    pass
