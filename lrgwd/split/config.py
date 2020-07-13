DEFAULTS = {
    "save_path": "./runs/dev/split",
    "source_path": "./runs/dev/extracted",
    "test_split": .10,
    "val_split": .05,
    "batch_size": 100000,
}

TRAIN_TENSORS_FN = "train_tensors.csv"
VAL_TENSORS_FN = "val_tensors.csv"
TEST_TENSORS_FN = "test_tensors.csv"

TRAIN_LABELS_FN = "train_labels.csv"
VAL_LABELS_FN = "val_labels.csv"
TEST_LABELS_FN = "test_labels.csv"

TRAIN_GWFV_FN = "train_gwfv.csv"
TRAIN_GWFU_FN = "train_gwfu.csv"

VAL_GWFV_FN = "val_gwfv.csv"
VAL_GWFU_FN = "val_gwfu.csv"


TEST_GWFV_FN = "test_gwfv.csv"
TEST_GWFU_FN = "test_gwfu.csv"

GWFU_SCALER_FN = "gwfu_scaler.pkl"
GWFV_SCALER_FN = "gwfv_scaler.pkl"
TENSORS_SCALER_FN = "tensors_scaler.pkl"
