from setuptools import find_packages, setup

setup(
    name="audio-data-pytorch",
    packages=find_packages(exclude=[]),
    version="0.0.8",
    license="MIT",
    description="Audio Data - PyTorch",
    long_description_content_type="text/markdown",
    author="Flavio Schneider",
    author_email="archinetai@protonmail.com",
    url="https://github.com/archinetai/audio-data-pytorch",
    keywords=["artificial intelligence", "deep learning", "audio dataset"],
    install_requires=[
        "torch>=1.6",
        "torchaudio>=0.10.0",
        "data-science-types>=0.2",
        "requests",
        "tqdm",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
    ],
)
