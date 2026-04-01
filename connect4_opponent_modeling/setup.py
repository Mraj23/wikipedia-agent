from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8") if (this_directory / "README.md").exists() else ""

setup(
    name="connect4_opponent_modeling",
    version="0.1.0",
    description="Does Explicit Opponent Modeling During RL Training Develop Transferable Adversarial Reasoning? A Connect Four Study.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "open_spiel",
        "torch>=2.0.0",
        "transformers>=4.51.0",
        "datasets",
        "numpy",
        "tqdm",
        "matplotlib",
        "pytest",
        "accelerate",
        "peft",
        "pandas",
        "scipy",
    ],
)
