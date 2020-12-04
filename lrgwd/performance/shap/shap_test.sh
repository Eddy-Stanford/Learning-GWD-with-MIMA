#!/bin/bash
#SBATCH --job-name utempeval
#SBATCH --ntasks=8
#SBATCH --partition=twohour
#SBATCH --mem-per-cpu=8GB
#SBATCH --time=02:00:00
#SBATCH --error=/scratch/zespinos/scripts/feature_experiments/utemp_eval.err
#SBATCH --output=/scratch/zespinos/scripts/feature_experiments/utemp_eval.out

CODE_DIR=/scratch/zespinos/Learning-GWD-with-MIMA
DATA_DIR=/data/cees/zespinos/runs/feature_experiments/40_levels

MODEL_ONE=baseline.120.hdf5

EXPERIMENT=full_features_cont #only_u
TRAIN_YEAR=year_one
TEST_YEAR=year_four
TARGET=gwfv

cd $CODE_DIR

python lrgwd shap --no-tracking --model-path $DATA_DIR/$TRAIN_YEAR/train/$TARGET/$EXPERIMENT/$MODEL_ONE --save-path $DATA_DIR/$TEST_YEAR/shap/$TARGET/$EXPERIMENT --scaler-path $DATA_DIR/year_one/split/full_features --source-path $DATA_DIR/$TEST_YEAR/extractor/ --target $TARGET --num-test-samples 10000



