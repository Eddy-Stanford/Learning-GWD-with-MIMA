from os import path

from setuptools import find_packages, setup

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

print("PACKAGES: ", find_packages(where="lrgwd"))

setup(
    name='lrgwd',
    version='0.0.1',
    description="Learning gravity wave drag with MiMA",
    long_description=long_description,
    long_description_context_type="test/markdown",
    url='https://github.com/zacespinosa/Learning-GWD-with-MIMA',
    author='Zac Espinosa',
    author_email='zespinos@stanford.edu',
    python_requires='>=3.6.9, <4',
    packages=[
        'lrgwd',
        'lrgwd.ingestor',
        'lrgwd.extractor',
        'lrgwd.split',
        'lrgwd.train',
        'lrgwd.performance.evaluate',
        'lrgwd.performance.compare',
        'lrgwd.utils',
    ],
    entry_points="""
        [console_scripts]
        lrgwd=lrgwd.__main__.py:cli
    """,
)

"""
install_requires=[Vjj
        'mlflow',
        'scipy',
        'numpy',
        'pandas',
        'matplotlib',
        'tqdm',
        'sklearn',
        'coloredlogs',
    ],
"""
