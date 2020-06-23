DEFAULTS = {
    "save_path": "./runs/dev/train",
    "source_path": "./runs/dev/split",
    "batch_size": 256,
    "chunk_size": 500,
    "model": "baseline",
    "target": "gwfu",
    "epochs": 10,
    "num_workers": 8,
}

MONITOR_METRIC = "output_9_loss"
