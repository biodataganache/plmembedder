# setup.py
from setuptools import setup, find_packages

setup(
    name="plmembedder",
    version="0.1.0",
    description="Protein Language Model Embedding Library",
    author="Jason McDermott",
    author_email="Jason.McDermott@pnnl.gov",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "torch>=1.9.0",
        "tqdm>=4.62.0",
    ],
    extras_require={
        "esm": ["fair-esm>=2.0.0"],
        "protbert": ["transformers>=4.20.0", "sentencepiece"],
        "prott5": ["transformers>=4.20.0", "sentencepiece"],
        "all": [
            "fair-esm>=2.0.0",
            "transformers>=4.20.0",
            "sentencepiece",
        ],
    },
    entry_points={
        "console_scripts": [
            "plmembedder=plmembedder.__main__:main",
            "plmembed=plmembedder.__main__:main",  # Short alias
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
