"""
Setup file
"""

from setuptools import setup, find_packages

setup(
    name="mlops",
    version="0.1.0",
    packages=find_packages(),
    author="Stepan Denisov",
    author_email="sd.denisoff@gmail.com",
    description="MLOps end-to-end solution",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/sd-denisoff/mlops_hse_2024",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
