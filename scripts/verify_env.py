"""Task 1.1: record exact library versions and check for a GPU."""
import importlib
import platform
import sys

print("=" * 60)
print("VisionGuard environment & library verification")
print("=" * 60)
print(f"OS platform      : {platform.platform()}")
print(f"Python version   : {sys.version.split()[0]}")
print(f"Python executable: {sys.executable}")
print("-" * 60)

for label, module in [("TensorFlow", "tensorflow"), ("Keras Tuner", "keras_tuner"), ("MLflow", "mlflow"),
                      ("DagsHub", "dagshub"), ("DVC", "dvc"), ("Scikit-Learn", "sklearn"),
                      ("Pandas", "pandas"), ("NumPy", "numpy"), ("Pillow", "PIL"),
                      ("Matplotlib", "matplotlib"), ("Seaborn", "seaborn"), ("Flask", "flask"),
                      ("SQLAlchemy", "sqlalchemy"), ("Requests", "requests")]:
    try:
        mod = importlib.import_module(module)
        print(f"{label:<14}: {getattr(mod, '__version__', 'installed (no __version__)')}")
    except ImportError:
        print(f"{label:<14}: NOT INSTALLED")

try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    print("-" * 60)
    print(f"Available GPUs   : {gpus if gpus else 'None detected (running in CPU mode)'}")
except ImportError:
    pass
