import sys
import importlib

PACKAGES = [
    "numpy", "pandas", "sklearn", "scipy", "matplotlib", "seaborn",
    "torch", "torchvision", "torchinfo", "thop", "onnx", "onnxruntime",
    "mlflow", "memory_profiler", "psutil", "codecarbon", "fastapi",
    "uvicorn", "pytest", "httpx", "locust", "requests", "pyarrow",
    "joblib", "tqdm",
]

def main():
    print(f"Python: {sys.version}")
    print(f"Executable: {sys.executable}")
    print("-" * 40)
    for name in PACKAGES:
        try:
            mod = importlib.import_module(name)
            version = getattr(mod, "__version__", "unknown")
            print(f"{name}: {version}")
        except Exception as e:
            print(f"{name}: NOT INSTALLED ({e})")

if __name__ == "__main__":
    main()
