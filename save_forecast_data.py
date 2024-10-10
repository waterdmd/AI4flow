import os
import glob
import pandas as pd
import numpy as np

# -----------------------------
# Configuration and Setup
# -----------------------------

# Base directory containing the model output directories
base_dir = '/scratch/kdahal3/camels_losses'

# Define forecast horizons
forecast_horizons = [1, 7, 30]
forecast_horizon_labels = {1: '1day', 7: '7day', 30: '30day'}

# Pattern to locate BMA weights files
weights_files_pattern = {
    fh: os.path.join(base_dir, f'bma_results_{label}_with_metrics.csv')
    for fh, label in forecast_horizon_labels.items()
}

# Define model directories pattern per forecast horizon
model_dirs_pattern = {
    fh: os.path.join(base_dir, f'*_{label}')
    for fh, label in forecast_horizon_labels.items()
}

# Path to the station coordinates CSV file
coords_file = '/scratch/kdahal3/Caravan/attributes/camels/attributes_other_camels.csv'

# Output directory for BMA CSV results
output_dir = os.path.join(base_dir, 'all_stations_BMA_outputs')
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# Read Station Coordinates
# -----------------------------

try:
    coords_df = pd.read_csv(coords_file)
    # Extract Station_ID from 'gauge_id' by splitting and padding
    coords_df['Station_ID'] = coords_df['gauge_id'].apply(lambda x: x.split('_')[1])
    coords_df['Station_ID'] = coords_df['Station_ID'].astype(str).str.strip().str.zfill(8)  # Ensure 8 characters with leading zeros
    coords_df = coords_df[['Station_ID', 'gauge_lat', 'gauge_lon']]  # Keep only necessary columns
    print(f"Successfully read station coordinates. Total stations: {len(coords_df)}")
except Exception as e:
    print(f"Error reading coordinates file: {e}")
    exit(1)

# -----------------------------
# Function Definitions
# -----------------------------

def extract_model_name(weight_column):
    """
    Extracts the model name from the weight column name by removing the 'Weight_' prefix.
    Retains the forecast horizon suffix to ensure consistency with forecast data.
    
    Example:
        'Weight_camels_LSTM_nse_loss_1day' -> 'camels_LSTM_nse_loss_1day'
    """
    # Remove 'Weight_' prefix
    model_name = weight_column.replace('Weight_', '')
    return model_name

def apply_bma_forecast(forecast_df, weights_df):
    """
    Applies BMA to the forecasted values using the provided weights.
    
    Parameters:
        forecast_df (pd.DataFrame): Pivoted DataFrame with models as columns.
        weights_df (pd.DataFrame): DataFrame with Station_ID, Model, and Weight.
    
    Returns:
        pd.Series: BMA Forecast values.
    """
    # Pivot weights to wide format: rows=Station_ID, columns=Model, values=Weight
    weights_wide = weights_df.pivot(index='Station_ID', columns='Model', values='Weight')

    # Ensure that model columns in forecast_df match weights_wide
    common_models = forecast_df.columns.intersection(weights_wide.columns)
    print(f"Number of common models: {len(common_models)}")
    print(f"Common models (sample): {common_models.tolist()[:5]}...")  # Display first 5 common models

    if len(common_models) == 0:
        print("No common models found between forecast data and weights.")
        return pd.Series(dtype='float64')

    # Align forecast_df and weights_wide to common_models
    forecast_aligned = forecast_df[common_models]
    weights_aligned = weights_wide[common_models]

    # Normalize weights per Station_ID to ensure they sum to 1
    weights_normalized = weights_aligned.div(weights_aligned.sum(axis=1), axis=0)

    # Handle stations where the sum of weights is zero to avoid division by zero
    weights_normalized = weights_normalized.fillna(0)

    # Multiply forecasts by weights
    weighted_forecasts = forecast_aligned.multiply(weights_normalized)

    # Sum across models to get BMA forecast
    bma_forecast = weighted_forecasts.sum(axis=1)

    return bma_forecast

# -----------------------------
# Processing Forecast Horizons
# -----------------------------

# Define specific stations to inspect (for debugging)
stations_to_inspect = ['01022500', '04040500', '05399500']  # Add or modify as needed

for fh in forecast_horizons:
    fh_label = forecast_horizon_labels[fh]
    print(f"\nProcessing {fh}-Day Forecast Horizon...")
    
    # Path to the BMA weights file for the current forecast horizon
    weights_file = weights_files_pattern[fh]
    if not os.path.isfile(weights_file):
        print(f"Weights file for {fh}-day forecast not found at {weights_file}. Skipping.")
        continue
    
    # Read the weights file
    try:
        weights_df = pd.read_csv(weights_file)
        print(f"Successfully read weights file for {fh}-day forecast: {weights_file}")
        print(f"Weights file columns: {weights_df.columns.tolist()}")
    except Exception as e:
        print(f"Error reading weights file '{weights_file}': {e}. Skipping this forecast horizon.")
        continue
    
    # Extract model names and weights by melting the DataFrame
    weight_columns = [col for col in weights_df.columns if col.startswith('Weight_')]
    if not weight_columns:
        print(f"No weight columns found in weights file '{weights_file}'. Skipping.")
        continue
    
    # Melt the weights DataFrame to long format
    weights_melted = weights_df.melt(
        id_vars='Station_ID',
        value_vars=weight_columns,
        var_name='Weight_Model',
        value_name='Weight'
    )
    
    # Extract model names from 'Weight_Model' column
    weights_melted['Model'] = weights_melted['Weight_Model'].apply(lambda x: extract_model_name(x))
    
    # Keep only relevant columns
    weights_melted = weights_melted[['Station_ID', 'Model', 'Weight']]
    
    # Clean and format Station_IDs
    weights_melted['Station_ID'] = weights_melted['Station_ID'].astype(str).str.strip().str.zfill(8)
    
    # Normalize weights per Station_ID to ensure they sum to 1
    weights_melted['Weight'] = weights_melted.groupby('Station_ID')['Weight'].transform(lambda x: x / x.sum() if x.sum() != 0 else 0)
    
    # Remove any weights that are zero after normalization
    weights_melted = weights_melted[weights_melted['Weight'] > 0]
    
    print(f"Extracted weights for {fh}-day forecast. Total weight entries: {len(weights_melted)}")
    
    # Find all model directories matching the current forecast horizon
    model_dirs = glob.glob(model_dirs_pattern[fh])
    if not model_dirs:
        print(f"No model directories found for {fh}-day forecast with pattern '{model_dirs_pattern[fh]}'. Skipping.")
        continue
    print(f"Found {len(model_dirs)} model directories for {fh}-day forecast.")
    
    # Initialize a list to store data from all models
    all_models_data = []
    
    for model_dir in model_dirs:
        # Extract model name from directory name
        model_name = os.path.basename(model_dir)
        
        # Find all station CSV files in the current model directory
        station_csv_pattern = os.path.join(model_dir, 'camels_*_combined_forecast_vs_observed.csv')
        station_csv_files = glob.glob(station_csv_pattern)
        
        if not station_csv_files:
            print(f"No station CSV files found in '{model_dir}'. Skipping this model.")
            continue
        
        print(f"Processing {len(station_csv_files)} stations in model '{model_name}'.")
        
        for csv_file in station_csv_files:
            try:
                df = pd.read_csv(csv_file)
            except Exception as e:
                print(f"Error reading '{csv_file}': {e}. Skipping this file.")
                continue
            
            # Check for required columns
            required_columns = {'Date', 'Observed', 'Forecasted'}
            if not required_columns.issubset(df.columns):
                print(f"Missing required columns in '{csv_file}'. Skipping this file.")
                continue
            
            # Extract Station_ID from filename
            filename = os.path.basename(csv_file)
            try:
                station_id = filename.split('_')[1]
                station_id = station_id.strip().zfill(8)  # Ensure 8-digit string
            except IndexError:
                print(f"Unexpected filename format: '{filename}'. Skipping this file.")
                continue
            
            # Add 'Model' and 'Station_ID' columns
            df['Model'] = model_name
            df['Station_ID'] = station_id
            all_models_data.append(df)
    
    if not all_models_data:
        print(f"No forecast data collected for {fh}-day forecast horizon. Skipping.")
        continue
    
    # Combine all model data into a single DataFrame
    forecast_combined = pd.concat(all_models_data, ignore_index=True)
    print(f"Combined forecast data shape before processing: {forecast_combined.shape}")
    
    # Convert 'Date' column to datetime
    try:
        forecast_combined['Date'] = pd.to_datetime(forecast_combined['Date'])
    except Exception as e:
        print(f"Error converting 'Date' column to datetime: {e}. Skipping {fh}-day forecast.")
        continue
    
    # Check for consistency in 'Observed' values across models for the same station and date
    observed_consistency = forecast_combined.groupby(['Station_ID', 'Date'])['Observed'].nunique()
    inconsistent_observed = observed_consistency[observed_consistency > 1]
    if not inconsistent_observed.empty:
        print(f"Warning: {len(inconsistent_observed)} station-date combinations have inconsistent 'Observed' values.")
        # For simplicity, take the first observed value in such cases
        forecast_combined = forecast_combined.sort_values('Model').drop_duplicates(subset=['Station_ID', 'Date'], keep='first')
        print(f"Resolved inconsistent 'Observed' values by retaining the first occurrence.")
    
    # Pivot the DataFrame to have one column per model's forecast
    forecast_pivot = forecast_combined.pivot_table(
        index=['Station_ID', 'Date'],
        columns='Model',
        values='Forecasted'
    )
    
    # Get unique observed values
    observed_df = forecast_combined[['Station_ID', 'Date', 'Observed']].drop_duplicates().set_index(['Station_ID', 'Date'])
    
    # Combine observed values with pivoted forecasts
    combined_pivot = forecast_pivot.join(observed_df)
    
    # Remove rows with NaN values (if some models are missing data)
    combined_pivot = combined_pivot.dropna()
    print(f"Combined pivot shape after dropping NaNs: {combined_pivot.shape}")
    
    if combined_pivot.empty:
        print(f"All data is NaN after merging for {fh}-day forecast. Skipping.")
        continue
    
    # Identify model columns
    model_columns = [col for col in combined_pivot.columns if col not in ['Observed']]
    if not model_columns:
        print(f"No model forecast columns found for {fh}-day forecast. Skipping.")
        continue
    
    # Apply BMA to calculate the averaged forecast
    bma_forecast_series = apply_bma_forecast(combined_pivot, weights_melted)
    
    if bma_forecast_series.empty:
        print(f"BMA forecast could not be computed for {fh}-day forecast due to lack of common models. Skipping.")
        continue
    
    # Assign the BMA forecast to the DataFrame
    combined_pivot['BMA_Forecast'] = bma_forecast_series
    
    # Calculate Forecast_Min and Forecast_Max across all models
    combined_pivot['Forecast_Min'] = combined_pivot[model_columns].min(axis=1)
    combined_pivot['Forecast_Max'] = combined_pivot[model_columns].max(axis=1)
    
    # Reset index to turn 'Station_ID' and 'Date' back into columns
    combined_pivot.reset_index(inplace=True)
    
    # Merge with coordinates
    final_df = pd.merge(combined_pivot, coords_df, on='Station_ID', how='left')
    
    # Check for missing coordinates
    missing_coords = final_df['gauge_lat'].isna().sum()
    if missing_coords > 0:
        print(f"Warning: {missing_coords} records have missing coordinates. These will be excluded.")
        final_df = final_df.dropna(subset=['gauge_lat', 'gauge_lon'])
    
    # Reorder and select relevant columns
    output_columns = [
        'Station_ID', 'gauge_lat', 'gauge_lon',
        'Date', 'Observed', 'BMA_Forecast',
        'Forecast_Min', 'Forecast_Max'
    ]
    output_df = final_df[output_columns]
    
    # -----------------------------
    # Debugging: Inspect Specific Stations
    # -----------------------------
    
    print("\n--- Debugging Information for Selected Stations ---")
    for station in stations_to_inspect:
        print(f"\nStation_ID: {station}")
        
        # Extract weights for the station
        station_weights = weights_melted[weights_melted['Station_ID'] == station]
        if station_weights.empty:
            print("  No weights found for this station.")
            continue
        print("  Weights:")
        print(station_weights[['Model', 'Weight']])
        
        # Extract forecasted values for the station
        station_forecast = forecast_combined[forecast_combined['Station_ID'] == station]
        if station_forecast.empty:
            print("  No forecast data found for this station.")
            continue
        print("  Forecasted Values from Models:")
        # To avoid excessive output, show unique forecasted values per model
        station_forecast_unique = station_forecast[['Model', 'Forecasted']].drop_duplicates()
        print(station_forecast_unique)
        
        # Extract BMA forecast for the station
        station_bma = output_df[output_df['Station_ID'] == station]
        if station_bma.empty:
            print("  No BMA forecast found for this station.")
            continue
        print("  BMA Forecast:")
        print(station_bma[['Date', 'Observed', 'BMA_Forecast', 'Forecast_Min', 'Forecast_Max']].head())
    
    print("\n--- End of Debugging Information ---\n")
    
    # -----------------------------
    # Handle Missing Weights
    # -----------------------------
    
    # Identify Station_IDs present in forecast data but missing in weights
    forecast_station_ids = set(forecast_combined['Station_ID'].unique())
    weights_station_ids = set(weights_melted['Station_ID'].unique())
    missing_station_ids = forecast_station_ids - weights_station_ids
    
    if missing_station_ids:
        print(f"Warning: {len(missing_station_ids)} Station_IDs in forecast data do not have corresponding weights.")
        print(f"Sample missing Station_IDs: {list(missing_station_ids)[:5]}")
        # Optionally, remove these Station_IDs from combined_pivot
        combined_pivot = combined_pivot[combined_pivot['Station_ID'].isin(weights_station_ids)]
        print(f"After removing Station_IDs without weights, combined pivot shape: {combined_pivot.shape}")
        
        # Optionally, assign equal weights to missing stations
        # Uncomment the following lines if you wish to assign equal weights
        """
        equal_weight = 1.0 / len(common_models)
        missing_weights = pd.DataFrame({
            'Station_ID': list(missing_station_ids),
            'Model': list(common_models) * len(missing_station_ids),
            'Weight': equal_weight
        })
        weights_melted = pd.concat([weights_melted, missing_weights], ignore_index=True)
        print(f"Assigned equal weights to {len(missing_weights)} Station_IDs without specific weights.")
        """

    # -----------------------------
    # Save to CSV
    # -----------------------------
    
    try:
        # Save the DataFrame to CSV
        output_csv_path = os.path.join(
            output_dir,
            f'BMA_forecasts_{fh}day_forecast.csv'
        )
        output_df.to_csv(output_csv_path, index=False)
        print(f"Successfully saved BMA results for {fh}-day forecast to '{output_csv_path}'")
    except Exception as e:
        print(f"Error saving CSV for {fh}-day forecast: {e}")
        continue

print("\nAll processing completed. BMA results are saved in the output directory.")
