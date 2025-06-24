import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import os
import matplotlib.pyplot as plt
import logging
import time
import gc
from tqdm import tqdm
from datetime import datetime
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("preprocessing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('preprocessing')

class SimpleKilnDataPreprocessor:
    """
    A simplified version of the KilnDataPreprocessor that focuses on essential preprocessing 
    steps and avoids memory-intensive operations.
    
    Key improvements:
    1. Uses minimum necessary lag windows
    2. Reduced rolling statistics calculations
    3. Skips anomaly detection which is memory-intensive
    4. Optimized for speed and memory efficiency
    5. Simplified feature engineering overall
    """
    
    def __init__(self, config=None):
        """
        Initialize preprocessor with simplified configuration
        
        Args:
            config (dict): Configuration parameters
        """
        self.config = config or {
            # Reduced lag windows - only use most relevant ones
            'lag_window_sizes': [1, 6, 24],  # hours: 1 hour, 6 hours, 1 day
            
            # Reduced rolling windows - only most relevant timeframes
            'rolling_window_sizes': [24],  # hours: 1 day
            
            # Keep zone pairs for differential features
            'zone_pairs': [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), 
                          (5, 6), (6, 7), (7, 8), (8, 9), (9, 10)]
        }
        
        self.scalers = {}
        self.imputers = {}
        
    def load_data(self, data_dir):
        """
        Load all kiln datasets from directory
        
        Args:
            data_dir (str): Directory containing the data files
            
        Returns:
            dict: Dictionary of dataframes for each data type
        """
        data = {}
        
        # Load MIS Report (daily)
        mis_path = os.path.join(data_dir, 'mis_report.csv')
        if os.path.exists(mis_path):
            mis_df = pd.read_csv(mis_path, parse_dates=['DATE'])
            mis_df.set_index('DATE', inplace=True)
            
            # Check for duplicate dates
            if mis_df.index.duplicated().any():
                # Handle duplicates by taking the mean
                mis_df = mis_df.groupby(level=0).mean()
                
            data['mis'] = mis_df
        
        # Load Air Calibration data (daily)
        air_path = os.path.join(data_dir, 'air_calibration.csv')
        if os.path.exists(air_path):
            air_df = pd.read_csv(air_path, parse_dates=['DATE'])
            # Handle duplicate dates by pivoting fan values to columns
            # First check if the structure requires pivoting
            if len(air_df) > len(air_df['DATE'].unique()):
                # Need to pivot to handle multiple fans per date
                air_pivoted = air_df.pivot_table(
                    index='DATE', 
                    columns='FAN', 
                    values=['DAMPER', 'VELOCITY', 'AIR_FLOW'], 
                    aggfunc='mean'
                )
                # Flatten the column multi-index
                air_pivoted.columns = [f'{col[0]}_{col[1]}' for col in air_pivoted.columns]
                data['air'] = air_pivoted
            else:
                # No pivoting needed
                air_df.set_index('DATE', inplace=True)
                data['air'] = air_df
        
        # Load QRT Temperature data (2 hourly)
        qrt_path = os.path.join(data_dir, 'qrt_temperature.csv')
        if os.path.exists(qrt_path):
            qrt_df = pd.read_csv(qrt_path, parse_dates=['DATETIME'])
            # Check for duplicate timestamps
            if len(qrt_df) > len(qrt_df['DATETIME'].unique()):
                # Need to pivot to handle multiple zones per timestamp
                qrt_pivoted = qrt_df.pivot_table(
                    index='DATETIME',
                    columns='ZONE',
                    values='TEMPERATURE',
                    aggfunc='mean'
                )
                # Rename columns to QRT_ZONE_X format
                qrt_pivoted.columns = [f'QRT_ZONE_{zone}' for zone in qrt_pivoted.columns]
                data['qrt'] = qrt_pivoted
            else:
                qrt_df.set_index('DATETIME', inplace=True)
                data['qrt'] = qrt_df
        
        # Load Zone Temperature data (1-2 min)
        zone_path = os.path.join(data_dir, 'zone_temperature.csv')
        if os.path.exists(zone_path):
            data['zone'] = pd.read_csv(zone_path, parse_dates=['DATETIME'])
            data['zone'].set_index('DATETIME', inplace=True)
            
            # Check for duplicate timestamps
            if data['zone'].index.duplicated().any():
                # Handle duplicates by taking the mean
                data['zone'] = data['zone'].groupby(level=0).mean()
        
        # Load accretion events data
        events_path = os.path.join(data_dir, 'accretion_events.csv')
        if os.path.exists(events_path):
            data['events'] = pd.read_csv(events_path, parse_dates=['START_DATE', 'CRITICAL_DATE', 'CLEARED_DATE'])
        
        return data
    
    def align_time_series(self, data, target_freq='1h'):
        """
        Align all time series data to a common frequency
        
        Args:
            data (dict): Dictionary of dataframes
            target_freq (str): Pandas frequency string for resampling
            
        Returns:
            pd.DataFrame: Aligned dataframe with all features
        """
        # Filter out dataframes that don't have a datetime index
        time_indexed_dfs = {k: df for k, df in data.items() 
                          if isinstance(df, pd.DataFrame) and hasattr(df.index, 'min') 
                          and hasattr(df.index.dtype, 'kind') and df.index.dtype.kind == 'M'}
        
        if not time_indexed_dfs:
            raise ValueError("No time-indexed dataframes found. Cannot align time series.")
        
        # Create empty dataframe with datetime index at target frequency
        start_date = min([df.index.min() for df in time_indexed_dfs.values()])
        end_date = max([df.index.max() for df in time_indexed_dfs.values()])
        
        aligned_index = pd.date_range(start=start_date, end=end_date, freq=target_freq)
        aligned_df = pd.DataFrame(index=aligned_index)
        
        # Resample MIS data (daily)
        if 'mis' in data:
            mis_resampled = data['mis'].resample(target_freq).ffill()
            # Add prefix to column names
            mis_resampled.columns = [f'mis_{col}' for col in mis_resampled.columns]
            aligned_df = aligned_df.join(mis_resampled)
        
        # Resample Air Calibration data (daily)
        if 'air' in data:
            air_resampled = data['air'].resample(target_freq).ffill()
            # Add prefix to column names if not already there
            if not all(col.startswith('air_') for col in air_resampled.columns):
                air_resampled.columns = [f'air_{col}' for col in air_resampled.columns]
            aligned_df = aligned_df.join(air_resampled)
        
        # Resample QRT Temperature data (2 hourly)
        if 'qrt' in data:
            qrt_resampled = data['qrt'].resample(target_freq).interpolate(method='linear')
            # Add prefix to column names if not already there
            if not all(col.startswith('qrt_') for col in qrt_resampled.columns):
                qrt_resampled.columns = [f'qrt_{col}' for col in qrt_resampled.columns]
            aligned_df = aligned_df.join(qrt_resampled)
        
        # Resample Zone Temperature data (1-2 min)
        if 'zone' in data:
            # For high frequency data, use mean instead of forward fill
            zone_resampled = data['zone'].resample(target_freq).mean()
            # Add prefix to column names
            zone_resampled.columns = [f'zone_{col}' for col in zone_resampled.columns]
            aligned_df = aligned_df.join(zone_resampled)
        
        return aligned_df
    
    def impute_missing_values(self, df, method='simple'):
        """
        Impute missing values with a simpler approach
        
        Args:
            df (pd.DataFrame): Input dataframe with missing values
            method (str): Imputation method ('simple' or 'knn')
            
        Returns:
            pd.DataFrame: Dataframe with imputed values
        """
        logger.info(f"Imputing missing values using {method} method")
        
        # Copy the dataframe to avoid modifying the original
        imputed_df = df.copy()
        
        if method == 'simple':
            # Use simple imputation (mean for numeric, most frequent for categorical)
            numeric_cols = df.select_dtypes(include=['number']).columns
            
            # For numeric columns
            if len(numeric_cols) > 0:
                imputer = SimpleImputer(strategy='mean')
                imputed_values = imputer.fit_transform(df[numeric_cols])
                imputed_df[numeric_cols] = imputed_values
                self.imputers['numeric'] = imputer
            
            # For other columns, forward fill and backward fill
            other_cols = df.columns.difference(numeric_cols)
            if len(other_cols) > 0:
                imputed_df[other_cols] = imputed_df[other_cols].fillna(method='ffill')
                imputed_df[other_cols] = imputed_df[other_cols].fillna(method='bfill')
        
        elif method == 'knn':
            # Less memory-intensive KNN imputation with reduced neighbors
            from sklearn.impute import KNNImputer
            imputer = KNNImputer(n_neighbors=3)  # Use fewer neighbors for speed
            
            # Process in batches to save memory
            columns_to_impute = df.columns[df.isna().any()]
            for i in range(0, len(columns_to_impute), 50):
                batch_cols = columns_to_impute[i:i+50]
                if len(batch_cols) > 0:
                    try:
                        # Only apply KNN to numeric columns
                        numeric_batch = [col for col in batch_cols if np.issubdtype(df[col].dtype, np.number)]
                        if numeric_batch:
                            imputer = KNNImputer(n_neighbors=3)
                            batch_imputed = imputer.fit_transform(df[numeric_batch])
                            imputed_df[numeric_batch] = batch_imputed
                    except Exception as e:
                        logger.warning(f"Error during KNN imputation: {e}. Falling back to mean imputation.")
                        for col in batch_cols:
                            if pd.api.types.is_numeric_dtype(df[col]):
                                col_mean = df[col].mean()
                                imputed_df[col] = df[col].fillna(col_mean)
            
            # For any remaining missing values, use forward and backward fill
            imputed_df = imputed_df.fillna(method='ffill')
            imputed_df = imputed_df.fillna(method='bfill')
        
        else:
            raise ValueError(f"Unknown imputation method: {method}")
            
        # Simple check to make sure all NaNs were filled
        if imputed_df.isna().sum().sum() > 0:
            # If still NaNs exist, simple fill
            imputed_df = imputed_df.fillna(method='ffill')
            imputed_df = imputed_df.fillna(method='bfill')
            # Last resort: fill with zeros
            imputed_df = imputed_df.fillna(0)
        
        return imputed_df
    
    def create_simple_lagged_features(self, df):
        """
        Create basic lagged features without the overhead of the full implementation
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with lagged features
        """
        logger.info("Creating simplified lagged features")
        
        # Create a copy to avoid modifying the original
        lagged_df = df.copy()
        
        # Only process numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        # Use batch operations to reduce DataFrame fragmentation
        for lag in self.config['lag_window_sizes']:
            try:
                # Create lag features for all columns at once
                lag_features = {}
                lag_name_prefix = f"_lag_{lag}"
                
                for col in tqdm(numeric_cols, desc=f"Creating lag {lag} features"):
                    lag_name = f"{col}{lag_name_prefix}"
                    lag_features[lag_name] = df[col].shift(lag)
                
                # Add all lag features at once
                lag_df = pd.DataFrame(lag_features, index=df.index)
                lagged_df = pd.concat([lagged_df, lag_df], axis=1)
                
                # Force garbage collection to reduce memory pressure
                del lag_df
                gc.collect()
                
                logger.info(f"Added {len(numeric_cols)} lag features with lag {lag}")
            except Exception as e:
                logger.warning(f"Error creating lag {lag} features: {e}")
        
        return lagged_df
    
    def create_simple_rolling_stats(self, df):
        """
        Create simplified rolling statistics features
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with rolling statistics features
        """
        logger.info("Creating simplified rolling statistics")
        
        # Create a copy to avoid modifying the original
        rolling_df = df.copy()
        
        # Only process numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        # Process each window size separately
        for window_size in self.config['rolling_window_sizes']:
            try:
                # Batch operations to reduce DataFrame fragmentation
                rolling_features = {}
                
                # Calculate all rolling statistics at once 
                for col in tqdm(numeric_cols, desc=f"Creating rolling stats window {window_size}"):
                    rolling_window = df[col].rolling(window=window_size, min_periods=max(2, window_size//4))
                    feature_name = f"{col}_rolling_mean_{window_size}"
                    rolling_features[feature_name] = rolling_window.mean()
                
                # Add all rolling features at once
                rolling_stats_df = pd.DataFrame(rolling_features, index=df.index)
                rolling_df = pd.concat([rolling_df, rolling_stats_df], axis=1)
                
                # Force garbage collection to reduce memory pressure
                del rolling_stats_df
                gc.collect()
                
                logger.info(f"Added rolling mean features for {len(numeric_cols)} columns with window {window_size}")
            except Exception as e:
                logger.warning(f"Error creating rolling stats with window {window_size}: {e}")
        
        return rolling_df
    
    def create_temperature_differential_features(self, df):
        """
        Create temperature differential features
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with temperature differential features
        """
        logger.info("Creating temperature differential features")
        
        # Create a copy to avoid modifying the original
        temp_diff_df = df.copy()
        
        # Create differential features for temperature zones
        zone_cols = [col for col in df.columns if 'zone_ZONE_' in col or 'QRT_ZONE_' in col]
        
        # Batch creation of differential features to reduce fragmentation
        diff_features = {}
        feature_count = 0
        
        # Group pairs by zone type for batch processing
        zone_pairs = self.config['zone_pairs']
        
        # Create differential features for zone_ZONE type
        zone_pairs_processed = set()  # Track processed pairs to avoid duplicates
        
        for zone_type_prefix in ['zone_ZONE_', 'QRT_ZONE_']:
            # Filter columns by type
            type_cols = [col for col in zone_cols if zone_type_prefix in col]
            
            if not type_cols:
                continue
                
            for pair in zone_pairs:
                zone_a, zone_b = pair
                if pair in zone_pairs_processed:
                    continue
                    
                # Find corresponding zone columns
                col_a_pattern = f"{zone_type_prefix}{zone_a}"
                col_b_pattern = f"{zone_type_prefix}{zone_b}"
                
                zone_a_cols = [col for col in type_cols if col_a_pattern in col]
                zone_b_cols = [col for col in type_cols if col_b_pattern in col]
                
                # Create differential features in batch
                for col_a in zone_a_cols:
                    for col_b in zone_b_cols:
                        try:
                            diff_name = f"temp_diff_{zone_a}_to_{zone_b}"
                            diff_features[diff_name] = df[col_a] - df[col_b]
                            feature_count += 1
                            
                            # Add to processed pairs
                            zone_pairs_processed.add(pair)
                        except Exception as e:
                            logger.warning(f"Error creating temperature differential: {e}")
        
        # Add all differential features at once
        if diff_features:
            temp_diff_df = pd.concat([temp_diff_df, pd.DataFrame(diff_features, index=df.index)], axis=1)
            logger.info(f"Added {feature_count} temperature differential features")
        
        return temp_diff_df
    
    def create_material_ratio_features(self, df):
        """
        Create features for material ratios
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with ratio features
        """
        logger.info("Creating material ratio features")
        
        # Create a copy to avoid modifying the original
        ratio_df = df.copy()
        
        # Define common material ratio calculations
        try:
            if 'mis_IRON ORE CONSUMPTION' in df.columns and 'mis_GROSS COAL CONSUMPTION' in df.columns:
                # Iron ore to coal ratio
                ratio_df['mis_IRON_ORE_TO_COAL_RATIO'] = df['mis_IRON ORE CONSUMPTION'] / df['mis_GROSS COAL CONSUMPTION']
        except Exception as e:
            logger.warning(f"Error creating material ratio features: {e}")
            
        # Add more ratio calculations if needed
        
        return ratio_df
    
    def create_target_variables(self, df, events_df):
        """
        Create target variables for accretion prediction
        
        Args:
            df (pd.DataFrame): Feature dataframe
            events_df (pd.DataFrame): Accretion events dataframe
            
        Returns:
            pd.DataFrame: Dataframe with target variables
        """
        logger.info("Creating target variables")
        
        # Create a copy to avoid modifying the original
        target_df = df.copy()
        
        # Initialize target columns
        target_df['accretion_next_24h'] = 0
        target_df['accretion_next_48h'] = 0
        target_df['accretion_next_72h'] = 0
        
        # Initialize zone as integer to avoid str/float conversion issues
        target_df['accretion_zone'] = -1
        
        if events_df is None or len(events_df) == 0:
            return target_df
            
        # Convert index to datetime if needed
        if not isinstance(target_df.index, pd.DatetimeIndex):
            logger.warning("Target dataframe index is not a DatetimeIndex. Cannot create time-based targets.")
            return target_df
        
        # Check if EVENT_TYPE column exists in events_df and contains 'Material bridging'
        has_event_type = 'EVENT_TYPE' in events_df.columns
        if has_event_type:
            # Create a numerical mapping for event types 
            event_types = events_df['EVENT_TYPE'].unique()
            event_type_map = {event: idx for idx, event in enumerate(event_types)}
            logger.info(f"Found event types: {event_types}")
            logger.info(f"Event type mapping: {event_type_map}")
            
            # Create a new column for the event type code
            target_df['accretion_type'] = -1
        
        # For each accretion event, mark the target windows
        for _, event in events_df.iterrows():
            try:
                critical_date = pd.Timestamp(event['CRITICAL_DATE'])
                
                # Handle zone as integer to prevent conversion issues
                if 'ZONE' in event:
                    zone = int(event['ZONE']) if pd.notna(event['ZONE']) else -1
                else:
                    zone = -1
                
                # Handle event type if present
                if has_event_type and 'EVENT_TYPE' in event:
                    event_type = event['EVENT_TYPE']
                    # Map the event type to a numeric code
                    event_type_code = event_type_map.get(event_type, -1)
                
                # Mark timestamps within 24, 48, and 72 hours before the critical event
                for hours, col in [(24, 'accretion_next_24h'), 
                                  (48, 'accretion_next_48h'), 
                                  (72, 'accretion_next_72h')]:
                    
                    # Find all timestamps within the window
                    window_start = critical_date - pd.Timedelta(hours=hours)
                    window_mask = (target_df.index >= window_start) & (target_df.index < critical_date)
                    
                    # Mark positive examples
                    target_df.loc[window_mask, col] = 1
                    target_df.loc[window_mask, 'accretion_zone'] = zone
                    
                    # If we have event type, also mark that
                    if has_event_type:
                        target_df.loc[window_mask, 'accretion_type'] = event_type_code
                    
            except Exception as e:
                logger.warning(f"Error processing event: {e}")
                
        return target_df
    
    def scale_features(self, df, feature_cols=None):
        """
        Scale features to a common range
        
        Args:
            df (pd.DataFrame): Input dataframe
            feature_cols (list): List of columns to scale
            
        Returns:
            pd.DataFrame: Dataframe with scaled features
        """
        logger.info("Scaling features")
        
        if feature_cols is None:
            # Only scale numeric columns
            feature_cols = df.select_dtypes(include=['number']).columns
        
        # Create a copy to avoid modifying the original
        df_scaled = df.copy()
        
        # First, identify non-numeric columns that might cause issues later
        non_numeric_cols = df.select_dtypes(exclude=['number']).columns
        logger.info(f"Found {len(non_numeric_cols)} non-numeric columns: {list(non_numeric_cols)}")
        
        # Convert any identified non-numeric feature columns to categorical codes or dummy variables
        for col in non_numeric_cols:
            # Skip target columns 
            if col.startswith('accretion_') and col != 'accretion_zone':
                continue
            
            # For categorical columns, convert to numeric codes
            try:
                # Check if column needs to be processed
                if df[col].nunique() <= 1:
                    logger.info(f"Skipping column {col} with only {df[col].nunique()} unique values")
                    continue
                
                # Check the actual type of the column's data
                sample_value = df[col].iloc[0]
                
                if isinstance(sample_value, (str, int, float)):
                    # Standard scalar values can be converted directly
                    logger.info(f"Converting non-numeric column {col} to category codes")
                    
                    # Check if this is a Material bridging column (important for domain-specific knowledge)
                    if isinstance(sample_value, str) and ('Material bridging' in df[col].values or 
                                                         'material bridging' in df[col].astype(str).str.lower().values):
                        logger.info(f"Column {col} contains 'Material bridging' values")
                    
                    # Convert to category codes
                    df_scaled[f"{col}_code"] = df[col].astype('category').cat.codes
                    
                    # Drop the original string column to avoid ML training issues
                    df_scaled.drop(columns=[col], inplace=True)
                    logger.info(f"Converted {col} to {col}_code")
                    
                elif isinstance(sample_value, np.ndarray):
                    # Handle numpy array values - convert to dummy variables instead
                    logger.info(f"Column {col} contains numpy arrays. Converting to dummy variables.")
                    
                    # For array-like values, one approach is to create a string representation and then dummies
                    try:
                        # Convert arrays to strings for dummification
                        str_series = df[col].apply(lambda x: str(x) if isinstance(x, np.ndarray) else str(x))
                        
                        # Create dummy variables (limit to top categories if there are many)
                        if str_series.nunique() > 10:
                            # Too many unique values, use top categories
                            top_cats = str_series.value_counts().nlargest(10).index.tolist()
                            dummies = pd.get_dummies(str_series.apply(lambda x: x if x in top_cats else 'other'),
                                                    prefix=col, drop_first=True)
                        else:
                            dummies = pd.get_dummies(str_series, prefix=col, drop_first=True)
                        
                        # Add dummy columns to result
                        df_scaled = pd.concat([df_scaled, dummies], axis=1)
                        
                        # Drop the original column
                        df_scaled.drop(columns=[col], inplace=True)
                        logger.info(f"Converted {col} to {len(dummies.columns)} dummy variables")
                    except Exception as e:
                        logger.warning(f"Error creating dummies for array column {col}: {e}")
                        # If dummy creation fails, drop the problematic column
                        df_scaled.drop(columns=[col], inplace=True)
                        logger.warning(f"Dropped column {col} due to conversion error")
                
                else:
                    # For other complex types, just drop the column
                    logger.warning(f"Dropping column {col} with unsupported type: {type(sample_value)}")
                    df_scaled.drop(columns=[col], inplace=True)
                    
            except Exception as e:
                logger.warning(f"Error converting column {col}: {e}")
                # If conversion fails, drop the problematic column
                if col in df_scaled.columns:
                    df_scaled.drop(columns=[col], inplace=True)
                    logger.warning(f"Dropped column {col} due to conversion error")
        
        # Scale each numeric feature individually
        for col in feature_cols:
            try:
                # Skip target columns, newly created code columns, and columns with all same values
                if col.startswith('accretion_') or col.endswith('_code') or df[col].nunique() <= 1:
                    continue
                    
                scaler = StandardScaler()
                # Reshape for sklearn (expects 2D arrays)
                values = df[col].values.reshape(-1, 1)
                
                scaled_values = scaler.fit_transform(values)
                
                # Store scaler for later use
                self.scalers[col] = scaler
                
                # Add to result dataframe
                df_scaled[col] = scaled_values.flatten()
            except Exception as e:
                logger.warning(f"Error scaling feature {col}: {e}")
        
        return df_scaled
    
    def _handle_non_numeric_columns(self, df):
        """
        Helper method to handle non-numeric columns in a batch operation,
        with special handling for numpy.ndarray objects
        
        Args:
            df (pd.DataFrame): DataFrame with non-numeric columns
            
        Returns:
            pd.DataFrame: Modified DataFrame with numeric representations
        """
        # Find non-numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        non_numeric_cols = df.select_dtypes(exclude=['number']).columns
        
        if len(non_numeric_cols) == 0:
            logger.info("No non-numeric columns found.")
            return df
            
        logger.info(f"Found {len(non_numeric_cols)} non-numeric columns: {list(non_numeric_cols)}")
        
        # Process each non-numeric column
        for col in non_numeric_cols:
            logger.info(f"Converting non-numeric column {col} to category codes")
            
            # Check sample value to determine handling approach
            sample_vals = df[col].dropna().head(5).tolist()
            if not sample_vals:
                # No non-null values, just drop the column
                logger.info(f"Column {col} has only null values. Dropping.")
                df = df.drop(columns=[col])
                continue
                
            sample_val = sample_vals[0]
            
            if isinstance(sample_val, (str, int, float)):
                # Standard scalar values - use category codes
                try:
                    df[f"{col}_code"] = df[col].astype('category').cat.codes
                    logger.info(f"Converted {col} to {col}_code")
                    # Drop the original column
                    df = df.drop(columns=[col])
                except Exception as e:
                    logger.warning(f"Error converting column {col}: {e}")
                    # If conversion fails, drop the column
                    df = df.drop(columns=[col])
                    
            elif isinstance(sample_val, np.ndarray):
                # For numpy array columns, extract useful features from arrays
                logger.info(f"Column {col} contains numpy arrays. Converting to numeric features.")
                try:
                    # Create multiple useful features from arrays
                    array_features = {}
                    
                    # Extract array sizes
                    array_features[f"{col}_size"] = []
                    
                    # Extract statistical features
                    array_features[f"{col}_mean"] = []
                    array_features[f"{col}_std"] = []
                    array_features[f"{col}_min"] = []
                    array_features[f"{col}_max"] = []
                    
                    # Process each array
                    for val in df[col]:
                        if isinstance(val, np.ndarray) and val.size > 0:
                            # Basic stats from array
                            try:
                                flat_vals = val.flatten()
                                if np.issubdtype(flat_vals.dtype, np.number):  # Check if array contains numbers
                                    array_features[f"{col}_size"].append(val.size)
                                    array_features[f"{col}_mean"].append(np.mean(flat_vals))
                                    array_features[f"{col}_std"].append(np.std(flat_vals))
                                    array_features[f"{col}_min"].append(np.min(flat_vals))
                                    array_features[f"{col}_max"].append(np.max(flat_vals))
                                else:
                                    # Non-numeric array
                                    array_features[f"{col}_size"].append(val.size)
                                    array_features[f"{col}_mean"].append(0)
                                    array_features[f"{col}_std"].append(0)
                                    array_features[f"{col}_min"].append(0)
                                    array_features[f"{col}_max"].append(0)
                            except:
                                # Fallback values if processing fails
                                array_features[f"{col}_size"].append(0)
                                array_features[f"{col}_mean"].append(0)
                                array_features[f"{col}_std"].append(0)
                                array_features[f"{col}_min"].append(0)
                                array_features[f"{col}_max"].append(0)
                        else:
                            # Default values for non-array or empty array
                            array_features[f"{col}_size"].append(0)
                            array_features[f"{col}_mean"].append(0)
                            array_features[f"{col}_std"].append(0)
                            array_features[f"{col}_min"].append(0)
                            array_features[f"{col}_max"].append(0)
                    
                    # Add extracted features to DataFrame
                    for feat_name, feat_values in array_features.items():
                        df[feat_name] = feat_values
                        logger.info(f"Created {feat_name} feature from {col}")
                    
                    # Drop the original column
                    df = df.drop(columns=[col])
                    logger.info(f"Successfully converted numpy array column {col} to {len(array_features)} numeric features")
                except Exception as e:
                    logger.warning(f"Error processing numpy array column {col}: {e}")
                    # If conversion fails, drop the column
                    df = df.drop(columns=[col])
                    logger.warning(f"Dropped problematic array column {col}")
            else:
                # Unknown or complex types - drop them
                logger.warning(f"Dropping column {col} with unsupported type: {type(sample_val)}")
                df = df.drop(columns=[col])
        
        return df

    def process(self, data_dir, save_dir, batch_size=1000):
        """
        Main preprocessing pipeline that handles all steps
        
        Args:
            data_dir (str): Directory containing the raw data files
            save_dir (str): Directory to save processed data and preprocessing artifacts
            batch_size (int): Number of samples to process at once to reduce memory usage
            
        Returns:
            tuple: (processed_df, self) containing the processed dataframe and the processor instance
        """
        # Start preprocessing timing
        preprocessing_start_time = time.time()
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Memory tracking - try using psutil if available, otherwise skip memory tracking
        try:
            import psutil
            process_memory = lambda: psutil.Process().memory_info().rss / (1024 ** 3)  # GB
            initial_memory = process_memory()
            memory_tracking_enabled = True
        except ImportError:
            process_memory = lambda: 0  # Dummy function
            memory_tracking_enabled = False
            logger.info("psutil not available, memory tracking disabled")
            
        # 1. Load raw data
        start_time = time.time()
        data = self.load_data(data_dir)
        duration = time.time() - start_time
        logger.info(f"Data loading - Duration: {duration:.2f}s, Memory: {process_memory():.2f} GB")
        
        # 2. Align time series to common frequency
        start_time = time.time()
        df = self.align_time_series(data, target_freq='1h')
        duration = time.time() - start_time
        logger.info(f"Time series alignment - Duration: {duration:.2f}s, Memory: {process_memory():.2f} GB")
        
        # 3. Impute missing values (simplified approach)
        logger.info("Imputing missing values using simple method")
        start_time = time.time()
        
        # Simple imputation strategy for all numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        if numeric_cols.size > 0:
            imputer = SimpleImputer(strategy='median')
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
            self.imputers['numeric'] = imputer
            
            # Force garbage collection after large operation
            gc.collect()
        
        duration = time.time() - start_time
        logger.info(f"Missing value imputation - Duration: {duration:.2f}s, Memory: {process_memory():.2f} GB")
        
        # 4. Create lag features (simplified and batched)
        logger.info("Creating simplified lagged features")
        start_time = time.time()
        
        # Only create lags for temperature columns which are most relevant
        temp_cols = [col for col in df.columns if 'zone' in col.lower() or 'temp' in col.lower() or 'qrt' in col.lower()]
        
        # Process lag features in batches to reduce DataFrame fragmentation
        new_columns = {}
        for col in temp_cols:
            for lag in self.config['lag_window_sizes']:
                lag_name = f"{col}_lag_{lag}h"
                new_columns[lag_name] = df[col].shift(lag).values
                
        # Add all new columns at once
        df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)
        
        # Force garbage collection after large operation
        gc.collect()
        
        duration = time.time() - start_time
        logger.info(f"Lagged features creation - Duration: {duration:.2f}s, Memory: {process_memory():.2f} GB")
        
        # 5. Create rolling statistic features (simplified and batched)
        logger.info("Creating simplified rolling statistics")
        start_time = time.time()
          # Only calculate rolling stats for most important columns - ensuring they're all numeric
        candidate_cols = temp_cols + [col for col in df.columns if any(x in col.lower() for x in ['air', 'feed', 'fuel'])]
        important_cols = []
        
        # Filter to only include numeric columns
        for col in candidate_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                important_cols.append(col)
            else:
                logger.info(f"Excluding non-numeric column {col} from rolling statistics calculation")
                
        important_cols = list(set(important_cols))  # Remove duplicates
        
        # Calculate rolling statistics in batches to reduce DataFrame fragmentation
        new_roll_columns = {}
        for window in self.config['rolling_window_sizes']:
            # Process columns in sub-batches for better memory management
            for i in range(0, len(important_cols), max(1, batch_size // 10)):
                sub_batch_cols = important_cols[i:i+max(1, batch_size // 10)]
                  # Create temporary DataFrames for rolling calculations
                for col in sub_batch_cols:
                    # Only calculate rolling stats for numeric columns
                    if pd.api.types.is_numeric_dtype(df[col]):
                        try:
                            roll = df[col].rolling(window=window)
                            new_roll_columns[f"{col}_rolling_{window}h_mean"] = roll.mean().values
                            new_roll_columns[f"{col}_rolling_{window}h_std"] = roll.std().values
                        except Exception as e:
                            logger.warning(f"Error calculating rolling stats for column {col}: {e}")
                    else:
                        logger.warning(f"Skipping rolling stats for non-numeric column {col}")
                
                # Periodically force garbage collection during intensive operations
                if i % (batch_size // 2) == 0 and i > 0:
                    gc.collect()
            
        # Add all rolling statistics columns at once
        df = pd.concat([df, pd.DataFrame(new_roll_columns, index=df.index)], axis=1)
        
        # Force garbage collection after large operation
        gc.collect()
        
        duration = time.time() - start_time
        logger.info(f"Rolling statistics creation - Duration: {duration:.2f}s, Memory: {process_memory():.2f} GB")
        
        # 6. Create temperature differential features (batched)
        logger.info("Creating temperature differential features")
        start_time = time.time()
        
        # Create differential (gradient) features between adjacent zones
        new_diff_columns = {}
        for i, j in self.config['zone_pairs']:
            col_i = f"zone_ZONE_{i}"
            col_j = f"zone_ZONE_{j}"
            
            if col_i in df.columns and col_j in df.columns:
                # Temperature differential
                new_diff_columns[f"zone_diff_{i}_{j}"] = df[col_i].values - df[col_j].values
                
                # Rate of change (hourly)
                new_diff_columns[f"zone_{i}_hourly_change"] = df[col_i].diff().values
                new_diff_columns[f"zone_{j}_hourly_change"] = df[col_j].diff().values
        
        # Add all differential columns at once
        if new_diff_columns:
            df = pd.concat([df, pd.DataFrame(new_diff_columns, index=df.index)], axis=1)
            
            # Force garbage collection
            gc.collect()
        
        duration = time.time() - start_time
        logger.info(f"Temperature differential features - Duration: {duration:.2f}s, Memory: {process_memory():.2f} GB")
        
        # 7. Create material ratio features (batched)
        logger.info("Creating material ratio features")
        start_time = time.time()
        
        # Ratios between materials (if available)
        material_cols = [col for col in df.columns if any(x in col.lower() for x in ['feed', 'fuel', 'material'])]
        
        # Create all ratios in a batch
        new_ratio_columns = {}
        for i, col_i in enumerate(material_cols):
            for j, col_j in enumerate(material_cols):
                if i < j:  # Avoid duplicate ratios and self-ratios
                    col_name = f"ratio_{col_i}_{col_j}"
                    # Using safe division to avoid divide by zero errors
                    with np.errstate(divide='ignore', invalid='ignore'):
                        ratio_vals = np.where(
                            df[col_j].values == 0, 
                            0,  # Default value when denominator is zero
                            df[col_i].values / df[col_j].values
                        )
                    # Fix any remaining NaN or infinite values
                    ratio_vals = np.nan_to_num(ratio_vals, nan=0, posinf=0, neginf=0)
                    new_ratio_columns[col_name] = ratio_vals
        
        # Add all ratio columns at once
        if new_ratio_columns:
            df = pd.concat([df, pd.DataFrame(new_ratio_columns, index=df.index)], axis=1)
            
            # Force garbage collection
            gc.collect()
                    
        duration = time.time() - start_time
        logger.info(f"Material ratio features - Duration: {duration:.2f}s, Memory: {process_memory():.2f} GB")
        
        # 8. Create target variables
        logger.info("Creating target variables")
        start_time = time.time()
        
        # Use events data to create target variables if available
        if 'events' in data:
            events_df = data['events']
            
            # Create target columns: will accretion form in next 24/48/72 hours
            df['accretion_next_24h'] = 0
            df['accretion_next_48h'] = 0
            df['accretion_next_72h'] = 0
            df['accretion_zone'] = -1  # -1 means no accretion
            
            # Loop through events and mark corresponding target values
            for _, event in events_df.iterrows():
                start_date = event['START_DATE']
                critical_date = event['CRITICAL_DATE']
                zone = event['ZONE']
                
                # Define the lookback windows for target variables
                window_24h = pd.Timedelta(hours=24)
                window_48h = pd.Timedelta(hours=48)
                window_72h = pd.Timedelta(hours=72)
                
                # Mark all observations within the lookback windows
                mask_24h = (df.index >= start_date - window_24h) & (df.index <= start_date)
                mask_48h = (df.index >= start_date - window_48h) & (df.index <= start_date)
                mask_72h = (df.index >= start_date - window_72h) & (df.index <= start_date)
                
                # Set the target variables
                df.loc[mask_24h, 'accretion_next_24h'] = 1
                df.loc[mask_48h, 'accretion_next_48h'] = 1
                df.loc[mask_72h, 'accretion_next_72h'] = 1
                
                # Mark the affected zone
                mask_any = mask_24h | mask_48h | mask_72h
                df.loc[mask_any, 'accretion_zone'] = zone
        
        duration = time.time() - start_time
        logger.info(f"Target variables creation - Duration: {duration:.2f}s, Memory: {process_memory():.2f} GB")
        
        # 9. Handle non-numeric columns with improved processing
        logger.info("Scaling features")
        start_time = time.time()
        
        # Process non-numeric columns with specialized handling
        df = self._handle_non_numeric_columns(df)
        
        # Scale all numeric features
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        # Don't scale target columns
        target_cols = ['accretion_next_24h', 'accretion_next_48h', 'accretion_next_72h', 'accretion_zone']
        cols_to_scale = [col for col in numeric_cols if col not in target_cols]
        
        if cols_to_scale:
            # Scale features in batches to reduce memory usage
            scaler = StandardScaler()
            
            # Process in smaller batches to avoid memory issues
            for i in range(0, len(cols_to_scale), batch_size):
                # Get batch of columns
                batch_cols = cols_to_scale[i:i+batch_size]
                
                # Fit and transform batch
                # Use numpy arrays directly to avoid DataFrame fragmentation
                transformed = scaler.fit_transform(df[batch_cols])
                for j, col in enumerate(batch_cols):
                    df[col] = transformed[:, j]
                
                # Force garbage collection after each batch
                if i % (batch_size * 3) == 0 and i > 0:
                    gc.collect()
                
            self.scalers['features'] = scaler
            
        duration = time.time() - start_time
        logger.info(f"Feature scaling - Duration: {duration:.2f}s, Memory: {process_memory():.2f} GB")
        
        # 10. Optimize memory usage by converting to appropriate dtypes
        logger.info("Optimizing memory usage")
        start_time = time.time()
        
        # Loop through columns in batches
        for i in range(0, len(df.columns), batch_size):
            batch_cols = df.columns[i:i+batch_size]
            
            for col in batch_cols:
                if col in target_cols:
                    # Ensure target columns are properly formatted
                    if col == 'accretion_zone':
                        df[col] = df[col].astype('int32')
                    else:  # Binary targets
                        df[col] = df[col].astype('int8')
                        
                elif pd.api.types.is_float_dtype(df[col]):
                    # For float columns, downcast to float32 to save memory
                    df[col] = pd.to_numeric(df[col], downcast='float')
                    
                elif pd.api.types.is_integer_dtype(df[col]):
                    # For integer columns, downcast to smallest appropriate type
                    df[col] = pd.to_numeric(df[col], downcast='integer')
        
        # 11. Drop rows with NAs in target variables
        if 'accretion_next_24h' in df.columns:
            df = df.dropna(subset=['accretion_next_24h'])
            
        # Force garbage collection
        gc.collect()
        
        duration = time.time() - start_time
        logger.info(f"Memory optimization - Duration: {duration:.2f}s, Memory: {process_memory():.2f} GB")
        
        # Save processed dataframe
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        save_path = os.path.join(save_dir, f"processed_data_{timestamp}.csv")
        df.to_csv(save_path, index=True)
        
        # Save preprocessor (self)
        joblib.dump(self, os.path.join(save_dir, "preprocessor.joblib"))
        
        # Report total preprocessing time
        total_duration = time.time() - preprocessing_start_time
        logger.info(f"Total preprocessing time: {total_duration:.2f} seconds")
        logger.info(f"Processed dataframe shape: {df.shape}")
        
        return df, self
