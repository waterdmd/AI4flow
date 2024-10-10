import os
import yaml
import argparse
import numpy as np
import pandas as pd
import glob

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Export test data for evaluation.')
parser.add_argument('--config', type=str, default='config.yml', help='Path to the config file.')
args = parser.parse_args()

# Load the specified configuration file
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

# Create output directory for test data
test_data_folder = config.get('test_data_folder', 'test_data')
os.makedirs(test_data_folder, exist_ok=True)

# Helper function to load and process datasets
def load_and_process_data(file_paths):
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
            data_list.append((csv_file, dataset))
    return data_list

# Load main datasets
print("Loading main datasets...")
main_datasets = load_and_process_data(config['data_paths'])

# Export test data
for csv_file, dataset in main_datasets:
    station_name = os.path.splitext(os.path.basename(csv_file))[0]
    # Split the dataset into train, validation, and test
    total_records = len(dataset)
    train_end = int(total_records * 0.7)
    val_end = int(total_records * 0.8)
    test_data = dataset.iloc[val_end:]
    # Save the test data to CSV
    test_data_path = os.path.join(test_data_folder, f"{station_name}_test_data.csv")
    test_data.to_csv(test_data_path)
    print(f"Test data saved for {station_name} at {test_data_path}")
