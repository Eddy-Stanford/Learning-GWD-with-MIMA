from typing import List, Tuple

import numpy as np
import tensorflow as tf
from lrgwd.config import NON_ZERO_GWD_PLEVELS
from lrgwd.models.config import WAVENET_PARAMS
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, BatchNormalization


class Wavenet_2():
    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def compile_model(self, model, learning_rate):
        # Optimizer
        adam_optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8, amsgrad=False
        )
        model.compile(
            # Adam combines AdaGrad (exponentially weighted derivates- hyperparams B1 and B2)
            # RMSProp (reduces variation in steps)
            optimizer=adam_optimizer,
            loss=tf.keras.losses.LogCosh(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE, name="log_cosh"),
            metrics=[
                # Fits to Median: robust to unwanted outliers
                tf.keras.metrics.MeanAbsoluteError(name="mean_absolute_error", dtype=None),
                # # Fits to Mean: robust to wanted outliers
                tf.keras.metrics.MeanSquaredError(name="mean_squared_error", dtype=None),
                # # Twice diferentiable, combination of MSE and MAE
                tf.keras.metrics.LogCoshError(name="logcosh", dtype=None),
                # # STD of residuals
                tf.keras.metrics.RootMeanSquaredError(
                    name="root_mean_squared_error", dtype=None
                )
            ]
        )
        return model

    def build(self,
              input_shape: Tuple[int] = (122, ), # 40*3 + 2
              output_shape: Tuple[int] = (66,), # 33 gwfu and 33 gwfv
              learning_rate: float = .001
        ):
        # Generate Layers
        inputs = self.add_input_layer(input_shape)
        hidden_layers = self.add_blocks(inputs)
        outputs = self.add_output_layer(output_shape, hidden_layers)

        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)

        # Optimizer
        adam_optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8, amsgrad=False
        )
        # Compile
        self.model.compile(
            # Adam combines AdaGrad (exponentially weighted derivates- hyperparams B1 and B2)
            # RMSProp (reduces variation in steps)
            optimizer=adam_optimizer,
            loss=tf.keras.losses.LogCosh(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE, name="log_cosh"),
            metrics=[
                # Fits to Median: robust to unwanted outliers
                tf.keras.metrics.MeanAbsoluteError(name="mean_absolute_error", dtype=None),
                # Fits to Mean: robust to wanted outliers
                tf.keras.metrics.MeanSquaredError(name="mean_squared_error", dtype=None),
                # Twice diferentiable, combination of MSE and MAE
                tf.keras.metrics.LogCoshError(name="logcosh", dtype=None),
                # STD of residuals
                tf.keras.metrics.RootMeanSquaredError(
                    name="root_mean_squared_error", dtype=None
                )
            ]
        )
        self.model.summary()

        return self.model


    def add_input_layer(self, input_shape) -> tf.keras.Input:
        return tf.keras.Input(shape=input_shape)


    def add_blocks(self, inputs: tf.keras.Input) -> Dense:
        prev_layer = inputs
        for units in WAVENET_PARAMS["units"]:
            prev_layer = Dense(
                units,
                activation="relu",
                kernel_initializer=tf.keras.initializers.GlorotNormal(), # Xavier
            )(prev_layer)

        return prev_layer

    def add_output_layer(self,
        output_shape: Tuple[int],
        hidden_layers: Dense
    ) -> List[Dense]:
        output_layer = Dense(
            units=output_shape[0],
        )(hidden_layers)

        return output_layer

def model_test():
    wavenet = Wavenet_2()
    model = wavenet.build()
    model.summary()


