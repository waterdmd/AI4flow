import os
import glob
import pandas as pd
import numpy as np
import xarray as xr
from metrics import calculate_metrics, get_available_metrics  # Import metrics functions

# Base directory containing the model output directories
base_dir = '/scratch/kdahal3/camels_losses'

# Define the forecast horizons you want to process
forecast_horizons = [1, 7, 30]

# Define the metrics list, excluding Peak-Timing and Missed-Peaks
metrics_list = get_available_metrics()
metrics_list = [m for m in metrics_list if m not in ["Peak-Timing", "Missed-Peaks"]]

# Function to process data for a given forecast horizon
def process_forecast_horizon(forecast_horizon):
    print(f"\nProcessing Forecast Horizon: {forecast_horizon} day(s)")

    # Find all model directories ending with f'_{forecast_horizon}day'
    model_dirs = glob.glob(os.path.join(base_dir, f'*_{forecast_horizon}day'))

    if not model_dirs:
        print(f"No model directories found for forecast horizon {forecast_horizon} day(s).")
        return

    # Get a list of all station IDs from the CSV filenames in the first model directory
    first_model_dir = model_dirs[0]
    csv_pattern = 'camels_*_combined_forecast_vs_observed.csv'
    csv_files = glob.glob(os.path.join(first_model_dir, csv_pattern))

    if not csv_files:
        print(f"No CSV files found in the first model directory: {first_model_dir}")
        return

    station_ids = [os.path.basename(f).split('_')[1] for f in csv_files]

    # Prepare a DataFrame to store results
    bma_results = []

    # Keep track of all possible model names for consistent columns
    all_model_names = set()

    # Loop over each station
    for station_id in station_ids:
        print(f"\nProcessing Station ID: {station_id}")
        station_data = {}
        station_data['Station_ID'] = station_id
        model_forecasts = {}
        observed_data = None
        models_with_data = []

        # Loop over each model directory to collect forecasts for the station
        for model_dir in model_dirs:
            model_name = os.path.basename(model_dir)
            csv_filename = f'camels_{station_id}_combined_forecast_vs_observed.csv'
            csv_file_path = os.path.join(model_dir, csv_filename)

            if not os.path.isfile(csv_file_path):
                # Skip if the station data is not available in this model
                print(f"Model {model_name}: No data for station {station_id}")
                continue

            print(f"Reading data for model {model_name}, station {station_id}")
            # Read the CSV file
            df = pd.read_csv(csv_file_path)

            # For forecast horizons >1, select only the desired Forecast_Horizon
            if forecast_horizon > 1:
                df = df[df['Forecast_Horizon'] == forecast_horizon]
                if df.empty:
                    print(f"Warning: No data for Forecast_Horizon={forecast_horizon} in station {station_id}, model {model_name}")
                    continue

            # Convert 'Date' column to datetime
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date').reset_index(drop=True)

            # Store the observed data (assuming it's the same across models)
            if observed_data is None:
                observed_data = df[['Date', 'Observed']].copy()
            else:
                # Check if observed data matches
                if not observed_data['Observed'].equals(df['Observed']):
                    print(f"Warning: Observed data mismatch for station {station_id} in model {model_name}")

            # Store the model's forecast
            model_forecasts[model_name] = df['Forecasted'].values
            models_with_data.append(model_name)
            all_model_names.add(model_name)  # Add to the set of all model names

        if not models_with_data:
            # No models have data for this station
            print(f"No models have data for station {station_id} at Forecast_Horizon={forecast_horizon}")
            continue

        # Create a DataFrame with observed and forecasts
        combined_df = observed_data.copy()
        for model_name in models_with_data:
            combined_df[model_name] = model_forecasts[model_name]

        # Remove any rows with NaN values (if any models are missing forecasts for certain dates)
        combined_df = combined_df.dropna()

        # Extract observed and forecasts
        observed = combined_df['Observed'].values
        forecasts = combined_df[models_with_data].values  # Shape: (n_samples, n_models)

        print(f"Number of samples: {len(observed)}")
        print(f"Models with data: {models_with_data}")

        n_models = forecasts.shape[1]
        n_samples = forecasts.shape[0]

        # Initialize a list to store NSE values for weighting
        nse_values = []
        model_metrics = {}

        # Calculate metrics for individual models
        for idx, model_name in enumerate(models_with_data):
            model_forecast = forecasts[:, idx]

            # Convert to xarray DataArrays for metric calculations
            obs_da = xr.DataArray(observed)
            sim_da = xr.DataArray(model_forecast)

            # Calculate metrics
            print(f"Calculating metrics for model {model_name}")
            metrics = calculate_metrics(obs_da, sim_da, metrics=metrics_list)

            # Store metrics
            for metric_name, metric_value in metrics.items():
                station_data[f'{metric_name}_{model_name}'] = metric_value

            # Store NSE value for weighting
            nse_model = metrics['NSE']
            nse_values.append(nse_model)
            model_metrics[model_name] = metrics

        # Convert NSE values to a numpy array for calculations
        nse_values = np.array(nse_values)

        # Implement proper Bayesian Model Averaging

        # Calculate prior probabilities (uniform priors)
        prior_probabilities = np.full(n_models, 1 / n_models)

        # Compute log-likelihoods for each model
        log_likelihoods = []
        for idx in range(n_models):
            residuals = observed - forecasts[:, idx]
            sigma_k_squared = np.var(residuals, ddof=1)
            n = len(observed)
            # Handle case where sigma_k_squared is zero
            if sigma_k_squared == 0:
                # Assign a very small variance
                sigma_k_squared = 1e-6
            log_likelihood = -0.5 * n * np.log(2 * np.pi * sigma_k_squared) - (np.sum(residuals ** 2) / (2 * sigma_k_squared))
            log_likelihoods.append(log_likelihood)

        log_likelihoods = np.array(log_likelihoods)

        # Compute log posterior probabilities (since priors are equal, they cancel out in log space)
        log_posteriors = log_likelihoods

        # To prevent numerical underflow, subtract the maximum log posterior
        max_log_posterior = np.max(log_posteriors)
        log_posteriors -= max_log_posterior

        # Convert to posterior probabilities
        posteriors = np.exp(log_posteriors)
        posterior_probabilities = posteriors / np.sum(posteriors)

        print(f"Posterior probabilities for models: {dict(zip(models_with_data, posterior_probabilities))}")

        # Compute the BMA forecast
        bma_forecast = np.dot(forecasts, posterior_probabilities)

        # Convert to xarray DataArrays for metric calculations
        bma_forecast_da = xr.DataArray(bma_forecast)
        obs_da = xr.DataArray(observed)

        # Calculate metrics for BMA forecast
        print("Calculating metrics for BMA forecast")
        metrics_bma = calculate_metrics(obs_da, bma_forecast_da, metrics=metrics_list)
        for metric_name, metric_value in metrics_bma.items():
            station_data[f'{metric_name}_BMA'] = metric_value

        # Store the posterior probabilities as weights
        for idx, model_name in enumerate(models_with_data):
            station_data[f'Weight_{model_name}'] = posterior_probabilities[idx]

        # Ensure that weights for all possible models are included, even if zero
        for model_name in all_model_names:
            if f'Weight_{model_name}' not in station_data:
                station_data[f'Weight_{model_name}'] = 0.0

        # Calculate Mean Forecast
        mean_forecast = forecasts.mean(axis=1)

        # Convert to xarray DataArrays for metric calculations
        mean_forecast_da = xr.DataArray(mean_forecast)
        obs_da = xr.DataArray(observed)

        # Calculate metrics for Mean Forecast
        print("Calculating metrics for Mean Forecast")
        metrics_mean = calculate_metrics(obs_da, mean_forecast_da, metrics=metrics_list)
        for metric_name, metric_value in metrics_mean.items():
            station_data[f'{metric_name}_Mean_Forecast'] = metric_value

        # Calculate actual coverage based on observed data and forecast range (min-max bounds from all models)
        lower_bound_bma = forecasts.min(axis=1)
        upper_bound_bma = forecasts.max(axis=1)
        actual_coverage = np.sum((observed >= lower_bound_bma) & (observed <= upper_bound_bma)) / n_samples
        station_data['Actual_Coverage_BMA'] = actual_coverage

        # Append the results
        bma_results.append(station_data)

    if not bma_results:
        print(f"No results to display for Forecast_Horizon={forecast_horizon} day(s).")
        return

    # Create a DataFrame from the results, ensuring all columns are included
    bma_results_df = pd.DataFrame(bma_results)

    # Display the results
    print(bma_results_df)

    # Optionally, save the results to a CSV file
    output_filename = f'bma_results_{forecast_horizon}day_with_metrics.csv'
    bma_results_df.to_csv(output_filename, index=False)
    print(f"Results saved to {output_filename}")

# Process each forecast horizon
for fh in forecast_horizons:
    process_forecast_horizon(fh)
