from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='lrgwd',
    version='0.0.1',
    description="Learning gravity wave drag with MiMA",
    long_description=long_description,
    long_description_context_type='test/markdown',
    url='https://github.com/zacespinosa/Learning-GWD-with-MIMA',
    author='Zac Espinosa',
    author_email='zespinos@stanford.edu',
    python_requires='>=3.6.9, <4',
    install_requires=[
        'mlflow',
        'scipy',
        'numpy',
        'pandas',
        'matplotlib',
        'tqdm',
        'sklearn',
    ],
)
