import tensorflow as tf

BLOCK_PARAMS = {
    "layers_in_block": 2,
    "activation": "relu",
    "units": [256, 256],
    "units_in_plevel_hidden_layers": [256, 128, 64, 32],
}

WAVENET_PARAMS = {
    "units": [500, 500, 256, 256, 256, 256, 128, 128]
}

VALID_MODELS = ["baseline", "wavenet_2"]

LOSS_ARRAY = [
    tf.keras.losses.MeanSquaredError(), # 0
    tf.keras.losses.MeanSquaredError(), # 1
    tf.keras.losses.MeanSquaredError(), # 2
    tf.keras.losses.MeanSquaredError(), # 3
    tf.keras.losses.MeanSquaredError(), # 4
    tf.keras.losses.MeanSquaredError(), # 5
    tf.keras.losses.MeanSquaredError(), # 6
    tf.keras.losses.MeanSquaredError(), # 7
    tf.keras.losses.MeanSquaredError(), # 8
    tf.keras.losses.MeanSquaredError(), # 9
    tf.keras.losses.MeanSquaredError(), # 10
    tf.keras.losses.MeanSquaredError(), # 11
    tf.keras.losses.MeanSquaredError(), # 12
    tf.keras.losses.MeanAbsoluteError(), # 13
    tf.keras.losses.MeanAbsoluteError(), # 14
    tf.keras.losses.MeanAbsoluteError(), # 15
    tf.keras.losses.MeanAbsoluteError(), # 16
    tf.keras.losses.MeanAbsoluteError(), # 17
]
