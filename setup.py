## need to update this 
## not really important

from setuptools import setup, find_packages

setup(
    name="dcgan-pytorch-mnist",
    version="0.1.0",
    description="A DCGAN implementation on FashionMNIST using PyTorch.",
    author="[Your Name]",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "matplotlib",
        "tqdm"
    ],
)
