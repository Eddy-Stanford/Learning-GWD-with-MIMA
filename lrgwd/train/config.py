DEFAULTS = {
    "save_path": "./runs/dev/train",
    "source_path": "./runs/dev/split",
    "batch_size": 256,
    "chunk_size": 500000,
    "model": "baseline",
    "target": "gwfu",
    "epochs": 10,
    "num_workers": 8,
    "steps_per_epoch": 1000,
    "validation_steps": 10,
    "learning_rate": .0001,
}

MONITOR_METRIC = "loss"
