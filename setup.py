from setuptools import setup, find_packages

setup(
    name="cbo",
    version="0.1.0",
    packages=find_packages(exclude=["in_domain", "out_of_domain", "optimizers", "common"]),
    python_requires=">=3.10",
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "tqdm",
        "wandb",
        "einops",
    ],
    extras_require={
        "bayesian": ["bayesian-torch", "ivon-opt"],
        "laplace": ["laplace-torch"],
        "imagenet": ["ffcv"],
    },
)
