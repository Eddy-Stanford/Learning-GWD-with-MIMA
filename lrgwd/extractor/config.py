DEFAULTS = {
    "save_path": "./runs/dev/extracted",
    "source_path": "./runs/dev/ingested/raw_data.npz",
    "batch_size": 10000, 
    "CNN_features": {
        "step_size": 1,
        "start_time": 1,
        "num_steps": 64,
    },
    "num_samples": None,
    "plevels_included": 18
}

TENSORS_FN = "tensors.csv"
GWFU_FN = "gwfu.csv"
GWFV_FN = "gwfv.csv"
LABELS_FN = "labels.csv"