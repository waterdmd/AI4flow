# losses.py

import tensorflow as tf
from keras import backend as K
from keras.losses import Loss

class KgeLoss(Loss):
    def __init__(self, name='kge_loss'):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        mean_true = tf.reduce_mean(y_true)
        mean_pred = tf.reduce_mean(y_pred)
        std_true = tf.math.reduce_std(y_true)
        std_pred = tf.math.reduce_std(y_pred)
        covariance = tf.reduce_mean((y_true - mean_true) * (y_pred - mean_pred))
        correlation = covariance / (std_true * std_pred + K.epsilon())
        std_ratio = std_pred / (std_true + K.epsilon())
        bias_ratio = mean_pred / (mean_true + K.epsilon())
        kge = 1 - tf.sqrt(
            tf.square(correlation - 1) + tf.square(std_ratio - 1) + tf.square(bias_ratio - 1)
        )
        return 1 - kge

    def get_config(self):
        base_config = super().get_config()
        return {**base_config}

class NseLoss(Loss):
    def __init__(self, name='nse_loss'):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        mean_true = tf.reduce_mean(y_true)
        numerator = tf.reduce_sum(tf.square(y_true - y_pred))
        denominator = tf.reduce_sum(tf.square(y_true - mean_true))
        nse = 1 - numerator / (denominator + K.epsilon())
        return 1 - nse

    def get_config(self):
        base_config = super().get_config()
        return {**base_config}

class ExpectileLoss(Loss):
    def __init__(self, expectile=0.5, name='expectile_loss'):
        super().__init__(name=name)
        self.expectile = expectile

    def call(self, y_true, y_pred):
        e = y_true - y_pred
        loss = tf.reduce_mean(
            tf.where(
                e >= 0,
                self.expectile * tf.square(e),
                (1 - self.expectile) * tf.square(e)
            )
        )
        return loss

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "expectile": self.expectile}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class QuantileLoss(Loss):
    def __init__(self, quantile=0.5, name='quantile_loss'):
        super().__init__(name=name)
        self.quantile = quantile

    def call(self, y_true, y_pred):
        e = y_true - y_pred
        loss = tf.reduce_mean(
            tf.maximum(self.quantile * e, (self.quantile - 1) * e)
        )
        return loss

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "quantile": self.quantile}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class HuberLoss(Loss):
    def __init__(self, delta=1.0, name='huber_loss'):
        super().__init__(name=name)
        self.delta = delta

    def call(self, y_true, y_pred):
        return tf.reduce_mean(tf.keras.losses.huber(y_true, y_pred, delta=self.delta))

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "delta": self.delta}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class WeightedMSELoss(Loss):
    def __init__(self, weights=None, name='weighted_mse_loss'):
        super().__init__(name=name)
        self.weights = weights

    def call(self, y_true, y_pred):
        if self.weights is None:
            weights = tf.ones_like(y_true)
        else:
            weights = tf.cast(self.weights, dtype=tf.float32)
        return tf.reduce_mean(weights * tf.square(y_true - y_pred))

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "weights": self.weights}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class MaeLoss(Loss):
    def __init__(self, name='mae_loss'):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        return tf.reduce_mean(tf.abs(y_true - y_pred))

    def get_config(self):
        base_config = super().get_config()
        return {**base_config}

class LogNseLoss(Loss):
    def __init__(self, epsilon=1e-6, name='lognse_loss'):
        super().__init__(name=name)
        self.epsilon = epsilon  # Small constant to avoid log(0)

    def call(self, y_true, y_pred):
        # Ensure y_true and y_pred are non-negative
        y_true = tf.maximum(y_true, 0.0)
        y_pred = tf.maximum(y_pred, 0.0)

        # Add epsilon to avoid log(0)
        y_true_log = tf.math.log(y_true + self.epsilon)
        y_pred_log = tf.math.log(y_pred + self.epsilon)

        # Handle cases where y_true_log has zero variance
        mean_true_log = tf.reduce_mean(y_true_log)
        denominator = tf.reduce_sum(tf.square(y_true_log - mean_true_log))
        # Replace zero denominators with a small constant
        denominator = tf.where(tf.equal(denominator, 0.0), K.epsilon(), denominator)

        numerator = tf.reduce_sum(tf.square(y_true_log - y_pred_log))

        nse = 1 - numerator / denominator
        # Ensure nse is within a valid range [-1, 1]
        nse = tf.clip_by_value(nse, -1.0, 1.0)

        # Return the loss
        loss = 1 - nse  # Subtract from 1 to convert to a loss
        return loss

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "epsilon": self.epsilon}


def kge_metric(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    mean_true = tf.reduce_mean(y_true)
    mean_pred = tf.reduce_mean(y_pred)
    std_true = tf.math.reduce_std(y_true)
    std_pred = tf.math.reduce_std(y_pred)
    covariance = tf.reduce_mean((y_true - mean_true) * (y_pred - mean_pred))
    correlation = covariance / (std_true * std_pred + K.epsilon())
    std_ratio = std_pred / (std_true + K.epsilon())
    bias_ratio = mean_pred / (mean_true + K.epsilon())
    kge = 1 - tf.sqrt(
        tf.square(correlation - 1) + tf.square(std_ratio - 1) + tf.square(bias_ratio - 1)
    )
    return kge

def nse_metric(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    mean_true = tf.reduce_mean(y_true)
    numerator = tf.reduce_sum(tf.square(y_true - y_pred))
    denominator = tf.reduce_sum(tf.square(y_true - mean_true))
    nse = 1 - numerator / (denominator + K.epsilon())
    return nse


def get_loss_function(loss_name, loss_params=None):
    available_loss_functions = {
        'nse_loss': NseLoss,
        'kge_loss': KgeLoss,
        'expectile_loss': ExpectileLoss,
        'quantile_loss': QuantileLoss,
        'huber_loss': HuberLoss,
        'weighted_mse_loss': WeightedMSELoss,
        'mae_loss': MaeLoss,
        'log_nse':LogNseLoss
    }

    if loss_name not in available_loss_functions:
        raise ValueError(f"Invalid loss function name: {loss_name}")

    loss_cls = available_loss_functions[loss_name]

    if loss_params:
        return loss_cls(**loss_params)
    else:
        return loss_cls()

def get_metric_function(metric_name):
    available_metrics = {
        'nse_metric': nse_metric,
        'kge_metric': kge_metric,
        'mae': tf.keras.metrics.MeanAbsoluteError(),
        'mse': tf.keras.metrics.MeanSquaredError(),
        # Add more metrics if needed
    }

    if metric_name not in available_metrics:
        raise ValueError(f"Invalid metric name: {metric_name}")

    return available_metrics[metric_name]
