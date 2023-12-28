from setuptools import setup, find_packages
from codecs import open
from os import path

project_name = "pytorch_model_wrapper"
HERE = path.abspath(path.dirname(__file__))
with open(path.join(HERE, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()
    
setup(
    name=project_name,
    version='0.0.3',
    description="Pytorch model wrapper",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Mikhail Gubarenko",
    author_email="dmdgik@gmail.com",
    license="Apache-2.0",
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent"
    ],
    package_dir={"" : "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    install_requires=[
        "tqdm",
        "torch",
        "accelerate",
        "mlflow",
        "loguru",
        "pyyaml",
        "boto3",
        "tensorboard",
    ]
)