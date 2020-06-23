Learning-GWD-with-MIMA
======================

This repository creates an ML pipeline to learn gravity wave drag from MiMA data and couples the deployed model with MiMA simulations. 

Getting Started
---------------

1. Create virtual environment
```sh
$ python -m venv env
```
2. Clone repo into env
```sh
$ git clone https://github.com/zacespinosa/Learning-GWD-with-MIMA.git
```
3. Install `lrgwd`
```sh
$ pip install lrgwd 
```
4. Install depenencies as needed (I haven't got around to updating `setup.py` with all requirements yet)

## Pipeline Overview
This outline does not list all flag for each command. To see all flags use `--help`.
```sh
$ python lrgwd <command> --help
$ python lrgwd --help
```

`ingestor`:

The ingestor consumes raw CDF data from MiMA and converts it to a compressed npz file.
If the visualize flag is set to true `ingestor` also creates histograms of each feature and the gwd at each pressure.

```sh
$ python lrgwd ingestor \
    --source-path <File Path to raw CDF data> \
    --save-path <File Path to save NPZ data> \
    --convert <Bool to convert from CDF to NPZ> \
    --visualize <Bool to create visualizations>
```

`extractor`:

The extractor consumes npz files and creates feature tensors and labels for the full
dataset and saves them in csv files.

```sh
$ python lrgwd extractor \
    --source-path <File path to raw dataset as npz> \
    --save-path <File path to save extracted dataset> \
    --plevels-included <Number of top plevels to use in feature tensors> \
    --num-samples <Number of samples to extract from the source-path>
```

`split`:

Splits raw csv data into train, validation, and test datasets and creates StandardScalers
to use when training the data

```sh
$ lrgwd split \
    ---source-path <File Path to extracted tensors> \
    --save-path <File Path to save splits> \
    --val-split <Float determining how much to allocate to train> \
    --test-split <Float determining how much to allocate to train>
```

`train`:

Train pulls from the `models` folder. In this step the given model is trained and hyperparameter tuning is done using the validation dataset.

```sh
$ lrgwd train \
    --save-path <File Path to save trained data> \
    --source-path  <File Path to train and validate datasets> \
    --model <Name of model to train>
```

`evaluate`:

Evaluates trained model and produces a performance report
```sh
$ lrgwd evaluate \
    --save-path <File Path to save performance report> \
    --source-path <File Path to test and labels datasets> \
    --model-path <File Path to trained model>
```

## Fortran Adapter 
This section outlines the api that Fortran will use to interact with this python model.