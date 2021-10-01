from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ["numpy >= 1.21", "scipy >= 1.2.1"]

setup(
    name="RECLAC",
    version="1.0.0",
    author="Tobias Braun",
    author_email="tobraun@pik-potsdam.de",
    description="A package to compute recurrence lacunarity and other box-counting-based recurrence quantifiers",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/ToBraun/RECLAC",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
    ],
)
