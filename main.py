#main.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING logs from TensorFlow

import warnings
warnings.filterwarnings('ignore')  # Suppress all Python warnings

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)  # Suppress INFO and WARNING logs from TensorFlow

import yaml
import argparse
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.callbacks import EarlyStopping, TensorBoard, LearningRateScheduler
import datetime
import gc
from keras import backend as K
from keras import mixed_precision
import xarray as xr

import losses
import models
from metrics import calculate_metrics, get_available_metrics

# Enable mixed precision for memory efficiency
mixed_precision.set_global_policy('mixed_float16')

# Enable GPU memory growth to avoid pre-allocating all GPU memory at once
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Learning rate schedule
def lr_schedule(epoch, lr):
    if epoch < 20:
        return 1e-3
    elif 20 <= epoch < 25:
        return 5e-4
    else:
        return 1e-4

# Create sequences from the data
def create_sequences(data, n_steps_in, n_steps_out):
    X, y = [], []
    for i in range(len(data)):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        if out_end_ix > len(data):
            break
        seq_x, seq_y = data[i:end_ix], data[end_ix:out_end_ix, -1]  # Assuming the last column is the target
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def create_time_sequences(data, n_steps_in, n_steps_out):
    X = []
    for i in range(len(data)):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        if out_end_ix > len(data):
            break
        seq_x = data[i:end_ix]
        X.append(seq_x)
    return np.array(X)

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Run forecast with a specific config file.')
parser.add_argument('--config', type=str, default='config.yml', help='Path to the config file.')
args = parser.parse_args()

# Load the specified configuration file
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

# Create output directory based on the output_prefix
output_dir = config['output_prefix']
os.makedirs(output_dir, exist_ok=True)

# Prepare directory for saving evaluation metrics
evaluation_folder = os.path.join(output_dir, config.get('evaluation_folder', 'evaluation_metrics'))
os.makedirs(evaluation_folder, exist_ok=True)

# Helper function to load and process datasets
def load_and_process_data(file_paths, for_transfer_learning=False):
    data_list = []
    for path in file_paths:
        if os.path.isdir(path):
            csv_files = glob.glob(os.path.join(path, "*.csv"))
        else:
            csv_files = [path]

        for csv_file in csv_files:
            dataset = pd.read_csv(csv_file, header=0, index_col=0)
            dataset.index = pd.to_datetime(dataset.index)
            dataset.replace('Min', np.nan, inplace=True)
            dataset = dataset.asfreq('D')
            dataset = dataset.apply(pd.to_numeric, errors='coerce')
            dataset = dataset.loc[dataset['streamflow'].first_valid_index():]
            dataset = dataset.loc[:dataset['streamflow'].last_valid_index()]
            dataset = dataset.interpolate(method='linear').ffill().bfill()
            if for_transfer_learning:
                data_list.append(dataset)
            else:
                data_list.append((csv_file, dataset))
    return data_list

# Load main datasets
print("Loading main datasets...")
main_datasets = load_and_process_data(config['data_paths'])

# Load transfer learning datasets (if applicable)
if 'transfer_learning_paths' in config:
    print("Loading transfer learning datasets...")
    transfer_datasets = load_and_process_data(config['transfer_learning_paths'], for_transfer_learning=True)
else:
    transfer_datasets = []

# Identify all feature names (excluding 'streamflow') across all main datasets
print("Identifying all feature names...")
all_features = set()
for _, dataset in main_datasets:
    all_features.update([col for col in dataset.columns if col != 'streamflow'])
all_features = sorted(all_features)  # Ensure consistent ordering
number_of_features = len(all_features) + 1  # +1 for 'streamflow' column

# Compute global min and max for each column across all main and transfer learning datasets
print("Computing global min and max for each column...")
min_max_dict = {}
# Include both main and transfer datasets for scaling
for dataset in main_datasets:
    _, ds = dataset
    for col in ds.columns:
        col_min = ds[col].min()
        col_max = ds[col].max()
        if col not in min_max_dict:
            min_max_dict[col] = {'min': col_min, 'max': col_max}
        else:
            if col_min < min_max_dict[col]['min']:
                min_max_dict[col]['min'] = col_min
            if col_max > min_max_dict[col]['max']:
                min_max_dict[col]['max'] = col_max

for ds in transfer_datasets:
    for col in ds.columns:
        col_min = ds[col].min()
        col_max = ds[col].max()
        if col not in min_max_dict:
            min_max_dict[col] = {'min': col_min, 'max': col_max}
        else:
            if col_min < min_max_dict[col]['min']:
                min_max_dict[col]['min'] = col_min
            if col_max > min_max_dict[col]['max']:
                min_max_dict[col]['max'] = col_max

# Convert min_max_dict to a DataFrame and save as CSV
min_max_df = pd.DataFrame(min_max_dict).T  # Transpose for easier CSV format
min_max_csv_path = os.path.join(output_dir, 'min_max.csv')
min_max_df.to_csv(min_max_csv_path)
print(f"Global min and max saved to {min_max_csv_path}")

# Function to scale data using global min and max
def scale_data(df, min_max_dict):
    scaled_df = pd.DataFrame(index=df.index)
    for col in df.columns:
        min_val = min_max_dict[col]['min']
        max_val = min_max_dict[col]['max']
        if max_val > min_val:
            scaled_df[col] = (df[col] - min_val) / (max_val - min_val)
        else:
            scaled_df[col] = 0.0  # Handle case where min == max to avoid division by zero
    return scaled_df

# Data generator for tf.data.Dataset
def data_generator(datasets, n_steps_in, n_steps_out, min_max_dict, features, split='train'):
    for dataset_info in datasets:
        csv_file, dataset = dataset_info
        station_name = os.path.splitext(os.path.basename(csv_file))[0] if csv_file else "transfer_learning"
        print(f"Processing station: {station_name} for split: {split}")

        # Ensure the dataset has all required features
        missing_features = set(features) - set(dataset.columns)
        if missing_features:
            print(f"Warning: Missing features {missing_features} in station {station_name}. Filling with NaN.")
            for mf in missing_features:
                dataset[mf] = np.nan

        # Reorder the dataset columns to match all_features + 'streamflow'
        dataset = dataset[features + ['streamflow']]

        # Extract time features
        dataset['year'] = dataset.index.year
        dataset['month'] = dataset.index.month
        dataset['day'] = dataset.index.day
        dataset['day_of_week'] = dataset.index.dayofweek + 1  # Monday=1, Sunday=7
        dataset['day_of_year'] = dataset.index.dayofyear

        time_features = ['year', 'month', 'day', 'day_of_week', 'day_of_year']

        # Scale all features using global min and max
        scaled_features = scale_data(dataset[features], min_max_dict)
        # Scale the target variable (streamflow) using global min and max
        scaled_target = scale_data(dataset[['streamflow']], min_max_dict)

        # Combine scaled features and target
        scaled = np.hstack((scaled_features.values, scaled_target.values))  # 'streamflow' is the last column

        # Prepare time features (without scaling, as temporal encoding will handle it)
        time_data = dataset[time_features].values  # Shape: (num_samples, num_time_features)

        # Create sequences
        X, y = create_sequences(scaled, n_steps_in, n_steps_out)
        time_sequences = create_time_sequences(time_data, n_steps_in, n_steps_out)  # Updated call

        # Determine split indices
        total_sequences = len(X)
        train_end = int(total_sequences * 0.7)
        val_end = int(total_sequences * 0.8)

        if split == 'train':
            sequences_X = X[:train_end]
            sequences_y = y[:train_end]
            sequences_time = time_sequences[:train_end]
        elif split == 'val':
            sequences_X = X[train_end:val_end]
            sequences_y = y[train_end:val_end]
            sequences_time = time_sequences[train_end:val_end]
        else:
            continue  # Skip processing for 'test' split

        # Yield data for the specified split
        for i in range(len(sequences_X)):
            yield (
                (sequences_X[i].astype(np.float32), sequences_time[i].astype(np.float32)),
                sequences_y[i].astype(np.float32)
            )  # Ensure float32 dtype for TensorFlow

# Create separate generators for training and validation
train_generator = lambda: data_generator(
    [(None, ds) for ds in transfer_datasets] + main_datasets,  # Include transfer datasets in train split
    config['n_steps_in'],
    config['n_steps_out'],
    min_max_dict,
    all_features,
    split='train'
)

val_generator = lambda: data_generator(
    [(None, ds) for ds in transfer_datasets] + main_datasets,  # Include transfer datasets in val split
    config['n_steps_in'],
    config['n_steps_out'],
    min_max_dict,
    all_features,
    split='val'
)

# Create dataset from generator for training
train_dataset = tf.data.Dataset.from_generator(
    train_generator,
    output_signature=(
        (
            tf.TensorSpec(shape=(config['n_steps_in'], number_of_features), dtype=tf.float32),  # Input features
            tf.TensorSpec(shape=(config['n_steps_in'], 5), dtype=tf.float32)  # Time features
        ),
        tf.TensorSpec(shape=(config['n_steps_out'],), dtype=tf.float32)  # Target
    )
)

# Create dataset from generator for validation
val_dataset = tf.data.Dataset.from_generator(
    val_generator,
    output_signature=(
        (
            tf.TensorSpec(shape=(config['n_steps_in'], number_of_features), dtype=tf.float32),  # Input features
            tf.TensorSpec(shape=(config['n_steps_in'], 5), dtype=tf.float32)  # Time features
        ),
        tf.TensorSpec(shape=(config['n_steps_out'],), dtype=tf.float32)  # Target
    )
)

# Shuffle, batch, and prefetch the training dataset
train_dataset = train_dataset.shuffle(buffer_size=10000) \
                             .batch(config['batch_size']) \
                             .prefetch(tf.data.AUTOTUNE)

# Batch and prefetch the validation dataset
val_dataset = val_dataset.batch(config['batch_size']) \
                         .prefetch(tf.data.AUTOTUNE)

# Build the model using models.py
output_prefix = config['output_prefix']  # e.g., "run_1"
log_dir = os.path.join(
    "logs",
    "fit",
    f"{output_prefix}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
)

# Callbacks
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_scheduler = LearningRateScheduler(lr_schedule)  # Uncomment if using learning rate scheduler

# Build the model using models.py
model = models.get_model(config, number_of_features)

# Get loss function and metrics
loss_function = losses.get_loss_function(
    config['loss_function'], config.get('loss_params', {})
)
metrics = [
    losses.get_metric_function(metric_name)
    for metric_name in config.get('metrics', [])
]

# Compile the model
learning_rate = config.get('learning_rate', 0.001)
optimizer_name = config.get('optimizer', 'Adam').lower()
clipvalue = config.get('clipvalue', 1.0)

if optimizer_name == 'adam':
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipvalue=clipvalue)
elif optimizer_name == 'sgd':
    opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)
else:
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")

model.compile(optimizer=opt, loss=loss_function, metrics=metrics)

model.summary()

# Train the model using the separate training and validation datasets
history = model.fit(
    train_dataset,
    epochs=config['epochs'],
    validation_data=val_dataset,
    verbose=1,
    callbacks=[early_stopping, tensorboard_callback, lr_scheduler]  # Add lr_scheduler if needed
)

# Save the model if required
if config.get('export_model', False):
    model_path = os.path.join(output_dir, "combined_model.keras")
    model.save(model_path)
    print(f"Model saved at {model_path}")

# Clear GPU memory
gc.collect()
K.clear_session()

# ========================= EVALUATION ================================
# Load test data
test_data_folder = config.get('test_data_folder', 'test_data')
test_csv_files = glob.glob(os.path.join(test_data_folder, "*_test_data.csv"))

for test_csv_file in test_csv_files:
    station_name = os.path.splitext(os.path.basename(test_csv_file))[0].replace('_test_data', '')
    print(f"Testing on station: {station_name}")

    test_dataset = pd.read_csv(test_csv_file, header=0, index_col=0)
    test_dataset.index = pd.to_datetime(test_dataset.index)

    # Ensure the dataset has all required features
    missing_features = set(all_features + ['streamflow']) - set(test_dataset.columns)
    if missing_features:
        print(f"Warning: Missing features {missing_features} in station {station_name}. Filling with NaN.")
        for mf in missing_features:
            test_dataset[mf] = np.nan

    # Reorder the dataset columns to match all_features + 'streamflow'
    test_dataset = test_dataset[all_features + ['streamflow']]

    # Extract time features
    test_dataset['year'] = test_dataset.index.year
    test_dataset['month'] = test_dataset.index.month
    test_dataset['day'] = test_dataset.index.day
    test_dataset['day_of_week'] = test_dataset.index.dayofweek + 1
    test_dataset['day_of_year'] = test_dataset.index.dayofyear
    time_features = ['year', 'month', 'day', 'day_of_week', 'day_of_year']
    time_data = test_dataset[time_features].values

    # Scale all features using global min and max
    scaled_features = scale_data(test_dataset[all_features], min_max_dict)
    # Scale the target variable (streamflow) using global min and max
    scaled_target = scale_data(test_dataset[['streamflow']], min_max_dict)

    # Combine scaled features and target
    scaled = np.hstack((scaled_features.values, scaled_target.values))  # 'streamflow' is the last column

    # Create sequences for LSTM
    test_X, test_y = create_sequences(scaled, config['n_steps_in'], config['n_steps_out'])
    test_time_sequences = create_time_sequences(time_data, config['n_steps_in'], config['n_steps_out'])

    # Create a tf.data.Dataset for testing
    test_dataset_tf = tf.data.Dataset.from_tensor_slices(
        ((test_X.astype(np.float32), test_time_sequences.astype(np.float32)), test_y.astype(np.float32))
    )
    test_dataset_tf = test_dataset_tf.batch(config['batch_size']).prefetch(tf.data.AUTOTUNE)

    # Predict on the test dataset
    y_pred = model.predict(test_dataset_tf)

    # Inverse transform the predictions and actual values using global min and max
    # Assuming 'streamflow' is the last column in min_max_dict
    streamflow_min = min_max_dict['streamflow']['min']
    streamflow_max = min_max_dict['streamflow']['max']
    y_pred_inv = y_pred * (streamflow_max - streamflow_min) + streamflow_min
    test_y_inv = test_y * (streamflow_max - streamflow_min) + streamflow_min

    # Ensure that the lengths of y_pred_inv and test_y_inv match before constructing the DataFrame
    min_length = min(len(y_pred_inv), len(test_y_inv))
    y_pred_inv = y_pred_inv[:min_length]  # Truncate to the minimum length
    test_y_inv = test_y_inv[:min_length]  # Truncate to the minimum length

    # Create the date range for the test data
    test_start_date = test_dataset.index[config['n_steps_in'] + config['n_steps_out'] - 1]
    test_dates = pd.date_range(start=test_start_date, periods=min_length)

    # Repeat each date for each forecast horizon and shift by the horizon
    forecast_horizons = np.arange(1, config['n_steps_out'] + 1)  # e.g., [1, 2, 3] for n_steps_out=3
    repeated_dates = np.repeat(test_dates, config['n_steps_out'])
    shifted_dates = repeated_dates + pd.to_timedelta(np.tile(forecast_horizons, min_length), unit='D')

    # Flatten the observed and forecasted values
    forecasted_values = y_pred_inv.flatten()
    observed_values = test_y_inv.flatten()

    # Construct DataFrame with correctly aligned dates
    df_result = pd.DataFrame({
        'Date': shifted_dates,
        'Forecast_Horizon': np.tile(forecast_horizons, min_length),
        'Observed': observed_values,
        'Forecasted': forecasted_values
    })

    # Save the forecast vs observed results to a CSV file
    result_csv_path = os.path.join(output_dir, f"{station_name}_forecast_vs_observed.csv")
    df_result.to_csv(result_csv_path, index=False)
    print(f"Results saved for {station_name} at {result_csv_path}")

    # ======================= METRICS CALCULATION ===========================
    print(f"Calculating metrics for station: {station_name}")

    # Convert to xarray DataArrays for metric calculations
    y_pred_da = xr.DataArray(y_pred_inv)
    test_y_da = xr.DataArray(test_y_inv)

    # Define the metrics list, excluding Peak-Timing and Missed-Peaks
    metrics_list = get_available_metrics()
    metrics_list = [m for m in metrics_list if m not in ["Peak-Timing", "Missed-Peaks"]]

    # Dictionary to store metrics per forecast day
    metrics_per_day = {}

    # Calculate metrics for each forecast day
    for day in range(config['n_steps_out']):
        # Extract the observations and simulations for the current forecast horizon
        obs = test_y_da[:, day] if test_y_da.ndim > 1 else test_y_da
        sim = y_pred_da[:, day] if y_pred_da.ndim > 1 else y_pred_da

        # Calculate metrics
        day_metrics = calculate_metrics(obs, sim, metrics=metrics_list)
        metrics_per_day[day + 1] = day_metrics  # Store metrics for each forecast day

    # Convert metrics_per_day into a DataFrame and save as CSV
    metrics_df = pd.DataFrame(metrics_per_day).T  # Transpose for better readability
    metrics_csv_path = os.path.join(evaluation_folder, f"{station_name}_daywise_metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=True)
    print(f"Metrics saved for {station_name} at {metrics_csv_path}")
