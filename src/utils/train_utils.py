import tensorflow as tf
import keras
import numpy as np
import keras.backend as K
import sys
NUMBER_TYPE = tf.float64  # or tf.float32
import tensorflow_probability as tfp
tfd = tfp.distributions

class LossAndErrorPrintingCallback(keras.callbacks.Callback):

    @staticmethod
    def __log(message, flush=True):
        sys.stdout.write(message)
        if flush:
            sys.stdout.flush()

    def on_train_batch_end(self, batch, logs=None):
        pass

    def on_test_batch_end(self, batch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        LossAndErrorPrintingCallback.__log(
            "\rThe average loss for epoch {} is {:7.10f} ".format(
                epoch, logs["loss"]
            )
        )





class SDEIdentification:
    """
    Wrapper class that can be used for SDE identification.
    Needs a "tf.keras.Model" like the SDEApproximationNetwork or VAEModel to work.
    """

    def __init__(self, model):
        self.model = model

    def train_model(self, x_n, x_np1, validation_split=0.1, n_epochs=100, batch_size=1000, step_size=None,
                    callbacks=[]):
        print(f"training for {n_epochs} epochs with {int(x_n.shape[0] * (1 - validation_split))} data points"
              f", validating with {int(x_n.shape[0] * validation_split)}")

        if not (step_size is None): # executed
            x_n = np.column_stack([step_size, x_n])
        y_full = np.column_stack([x_n, x_np1])  
        

        if len(callbacks) == 0:
            callbacks.append(LossAndErrorPrintingCallback())

        hist = self.model.fit(x=y_full,
                              epochs=n_epochs,
                              batch_size=batch_size,
                              verbose=0,
                              validation_split=validation_split,
                              callbacks=callbacks)
        return hist

    def drift_diffusivity(self, x):
        drift, std = self.model.call_xn(x)
        return K.eval(drift), K.eval(std)

    def sample_path(self, x0, step_size, NT, N_iterates, map_every_iteration=None):
        """
        Use the neural network to sample a path with the Euler Maruyama scheme.
        """
        step_size = tf.cast(np.array(step_size), dtype=NUMBER_TYPE)
        paths = [np.ones((N_iterates, 1)) @ np.array(x0).reshape(1, -1)]
        for it in range(NT):
            x_n = paths[-1]
            apx_mean, apx_scale = self.model.call_xn(x_n)
            x_np1 = tfd.MultivariateNormalDiag(
                loc=x_n + step_size * apx_mean,
                scale_diag=tf.math.sqrt(step_size) * apx_scale
            ).sample()

            x_i = keras.backend.eval(x_np1)
            if not (map_every_iteration is None):
                x_i = map_every_iteration(x_i)
            paths.append(x_i)
        return [
            np.row_stack([paths[k][i] for k in range(len(paths))])
            for i in range(N_iterates)
        ]
