DEFAULTS = {
    "model_path": "./runs/dev/train/baseline.01.hdf5",
    "compare": {
        "source_path": ["./runs/dev/performance/evaluate"],
        "save_path": "./runs/dev/performance/compare",
    },
    "evaluate": {
        "save_path": "./runs/dev/performance/evaluate",
        "source_path": "./runs/dev/split",
    },
    "target": "gwfu",
    "model": "baseline"
}
