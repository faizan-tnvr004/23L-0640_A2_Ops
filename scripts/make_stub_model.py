"""Create a tiny placeholder model (used by CI where the DVC-tracked model is absent).

Usage: python scripts/make_stub_model.py models/champion_model.keras
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.stub_model import build_stub_model  # noqa: E402

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "models/champion_model.keras"
    os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
    print("wrote", build_stub_model(target))
