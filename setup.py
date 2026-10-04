from setuptools import setup, find_packages

setup(
    name="tpn-model",
    version="0.2.0",
    description="Temporal Packet Network Inference Engine",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        # Required for the vectorized fast path. The pure-Python backend
        # still works without it, but is ~40-50x slower.
        "numpy>=1.20",
    ],
    extras_require={
        # Optional: only needed to load GGUF model files.
        "gguf": ["gguf>=0.6"],
        "test": ["pytest>=7.0", "numpy>=1.20"],
    },
)
