import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import KNNImputer
import joblib
import os
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from tqdm import tqdm
import warnings
import gc
import multiprocessing as mp
from joblib import Parallel, delayed
import psutil
from numba import jit, prange, set_num_threads, njit, config, types
from functools import partial
import time

# Determine number of cores to use for parallel processing (leave 1 core free)
N_CORES = max(1, mp.cpu_count() - 1)
# Configure Numba for best performance
set_num_threads(N_CORES)
# Enable fastmath for Numba (can improve performance at slight cost to precision)
config.THREADING_LAYER = 'threadsafe'

# Utility function for memory logging
def get_memory_usage():
    """Helper function to get current memory usage"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_usage_gb = memory_info.rss / (1024 ** 3)  # Convert to GB
    return memory_usage_gb

# Suppress warnings
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

class KilnDataPreprocessor:
    """
    Preprocesses raw kiln data for machine learning:
    - Aligns different frequency data to common timeline
    - Handles missing values
    - Creates lagged features
    - Detects anomalies in temperature profiles with focus on temperature drops
    - Engineers features related to accretion indicators:
      - Temperature drops and cooling patterns across zones
      - Production quality shifts (decreased lumps, increased pellets/fines)
      - Coal consumption increases
      - Complex temporal patterns indicating early accretion formation
    - Creates combined accretion early warning indicators
    """
    def __init__(self, config=None):
        """
        Initialize preprocessor with configuration
        
        Args:
            config (dict): Configuration parameters including:
                - lag_window_sizes: List of lag window sizes to create
                - rolling_window_sizes: List of sizes for rolling statistics
                - zone_pairs: List of zone pairs to calculate differentials
                - important_columns: List of columns to focus feature generation on
                - temperature_drop_threshold: Min percentage drop to flag as significant
                - coal_increase_threshold: Min percentage increase to flag as significant
        """
        self.config = config or {
            'lag_window_sizes': [6, 24, 72],  # Reduced from 7 to 3 windows
            'rolling_window_sizes': [24, 72],  # Reduced from 6 to 2 windows
            'zone_pairs': [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10)],  # Reduced pairs
            'important_columns': [],  # Will be populated during processing
            'max_features': 500,  # Target maximum number of columns in final dataset
            'temperature_drop_threshold': 5.0,  # 5% drop considered significant
            'coal_increase_threshold': 15.0,    # 15% increase considered significant
            'quality_shift_threshold': 10.0     # 10% quality shift considered significant
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
          # Load Shell Temperature data (daily)
        shell_path = os.path.join(data_dir, 'shell_temperature.csv')
        if os.path.exists(shell_path):
            shell_df = pd.read_csv(shell_path, parse_dates=['DATE'])
            # Handle duplicate dates by pivoting or averaging if needed
            if len(shell_df) > len(shell_df['DATE'].unique()):
                # Need to pivot to handle multiple positions per date
                shell_pivoted = shell_df.pivot_table(
                    index='DATE',
                    columns='POSITION',
                    values=[col for col in shell_df.columns if col.startswith('SHELL_TEMP')],
                    aggfunc='mean'
                )
                # Flatten the column multi-index
                shell_pivoted.columns = [f'{col[0]}_{col[1]}'.replace(' ', '_').replace('-', '_') 
                                      for col in shell_pivoted.columns]
                data['shell'] = shell_pivoted
            else:
                shell_df.set_index('DATE', inplace=True)
                data['shell'] = shell_df
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
            # Update column names to match what's generated in data_generator.py
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
            # Add prefix to column names
            air_resampled.columns = [f'air_{col}' for col in air_resampled.columns]
            aligned_df = aligned_df.join(air_resampled)
        
        # Resample Shell Temperature data (daily)
        if 'shell' in data:
            shell_resampled = data['shell'].resample(target_freq).ffill()
            # Add prefix to column names
            shell_resampled.columns = [f'shell_{col}' for col in shell_resampled.columns]
            aligned_df = aligned_df.join(shell_resampled)
        
        # Resample QRT Temperature data (2 hourly)
        if 'qrt' in data:
            qrt_resampled = data['qrt'].resample(target_freq).interpolate()
            # Add prefix to column names
            qrt_resampled.columns = [f'qrt_{col}' for col in qrt_resampled.columns]
            aligned_df = aligned_df.join(qrt_resampled)
        
        # Resample Zone Temperature data (1-2 min)
        if 'zone' in data:
            # Group by zone and resample
            zone_cols = [col for col in data['zone'].columns if col.startswith('ZONE_')]
            
            for zone_col in zone_cols:
                zone_df = data['zone'][[zone_col]].resample(target_freq).mean()
                aligned_df = aligned_df.join(zone_df)
        
        return aligned_df
    def impute_missing_values(self, df, method='knn', n_neighbors=5):
        """
        Impute missing values in the dataframe
        
        Args:
            df (pd.DataFrame): Input dataframe
            method (str): Imputation method ('knn', 'mean', 'median', 'ffill')
            n_neighbors (int): Number of neighbors for KNN imputation
            
        Returns:
            pd.DataFrame: Dataframe with imputed values
        """
        import numpy as np
        df_imputed = df.copy()
        
        # Check for non-numeric columns
        non_numeric_cols = df_imputed.select_dtypes(exclude=['number']).columns.tolist()
        
        # Save non-numeric columns to handle separately
        non_numeric_data = {}
        for col in non_numeric_cols:
            non_numeric_data[col] = df_imputed[col]
            df_imputed = df_imputed.drop(columns=[col])
        
        # Now proceed with numeric imputation
        if method == 'knn' and not df_imputed.empty:
            imputer = KNNImputer(n_neighbors=n_neighbors)
            imputed_array = imputer.fit_transform(df_imputed.values)
            df_imputed = pd.DataFrame(imputed_array, index=df.index, columns=df_imputed.columns)
            self.imputers['main'] = imputer
        elif method == 'mean' and not df_imputed.empty:
            df_imputed = df_imputed.fillna(df_imputed.mean())
        elif method == 'median' and not df_imputed.empty:
            df_imputed = df_imputed.fillna(df_imputed.median())
        elif method == 'ffill' and not df_imputed.empty:
            df_imputed = df_imputed.ffill().bfill()
        
        # Add back non-numeric columns
        for col, values in non_numeric_data.items():
            # For non-numeric, use ffill/bfill
            df_imputed[col] = values.ffill().bfill()
        
        return df_imputed
    
    def scale_features(self, df, method='standard'):
        """
        Scale features in the dataframe
        
        Args:
            df (pd.DataFrame): Input dataframe
            method (str): Scaling method ('standard', 'minmax')
            
        Returns:
            pd.DataFrame: Scaled dataframe
        """
        df_scaled = pd.DataFrame(index=df.index)
        
        for col in df.columns:
            if method == 'standard':
                scaler = StandardScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
                
            # Reshape for sklearn
            values = df[col].values.reshape(-1, 1)
            scaled_values = scaler.fit_transform(values)
            
            # Store scaler for later use
            self.scalers[col] = scaler
            
            # Add to result dataframe            df_scaled[col] = scaled_values.flatten()        
        return df_scaled
    
    def create_lagged_features(self, df, feature_cols=None, lag_hours=None, chunk_size=50):
        """
        Create lagged features for selected columns using parallel processing with JIT acceleration
        
        Args:
            df (pd.DataFrame): Input dataframe
            feature_cols (list): Columns to create lagged features for
            lag_hours (list): List of lag hours to create
            chunk_size (int): Number of columns to process in each batch
            
        Returns:
            pd.DataFrame: Dataframe with lagged features
        """
        import numpy as np
        import time
          # Compile a fast JIT function for creating lags with explicit type signatures
        @njit(cache=True, fastmath=True)
        def create_lag(values, lag_periods):
            """Numba-optimized function to create lags"""
            # Ensure we're working with float64 arrays
            values_float = values.astype(np.float64)
            n = len(values_float)
            result = np.full(n, np.nan, dtype=np.float64)
            
            # Simple bounds check for safety
            if lag_periods >= n:
                return result
                
            # Vectorized assignment is faster than a loop
            result[lag_periods:] = values_float[:(n-lag_periods)]
            return result
              # Function to process multiple lags for one column
        def process_column_lags(col_name, col_values, lag_periods_list):
            """Process all lags for a single column using JIT acceleration"""
            # Dictionary to store results for this column
            col_results = {}
            start_time = time.time()
            
            try:
                # Process each lag
                for lag in lag_periods_list:
                    # Create lag feature using optimized function
                    lag_values = create_lag(col_values, lag)
                    
                    # Store in result dictionary with standardized column name
                    col_results[f'{col_name}_lag_{lag}h'] = lag_values
            except Exception as e:
                print(f"ERROR processing lags for column {col_name}: {str(e)}")
            
            elapsed = time.time() - start_time
            return col_results, elapsed
          # If no specific columns provided, identify important ones
        if feature_cols is None:
            # Identify important columns if not already set
            if not self.config['important_columns']:
                print("Identifying important columns for targeted feature generation...")
                # Select temperature columns
                temp_cols = [col for col in df.columns if 'ZONE_' in col 
                            or 'qrt_' in col or 'shell_' in col]
                
                # Select a smaller subset of production/operational columns
                production_cols = [col for col in df.columns if 'mis_' in col][:10]
                
                # Select calibration columns
                calibration_cols = [col for col in df.columns if 'air_' in col][:5]
                
                # Combine important columns
                self.config['important_columns'] = temp_cols + production_cols + calibration_cols
                
                print(f"Selected {len(self.config['important_columns'])} important columns for feature generation")
                
            # Use only important columns for lag features
            feature_cols = self.config['important_columns']
            print(f"Creating lag features for {len(feature_cols)} columns instead of all {len(df.columns)}")
        
        if lag_hours is None:
            lag_hours = self.config['lag_window_sizes']
        
        # Get frequency of DataFrame
        freq = pd.infer_freq(df.index)
        if freq is None:
            # Default to hourly if can't infer
            freq = '1h'
        
        # Calculate optimal chunk size based on available memory
        memory_usage = get_memory_usage()
        if memory_usage > 12:  # High memory system
            chunk_size = max(50, min(200, len(feature_cols) // 4))
        elif memory_usage > 8:  # Medium memory system
            chunk_size = max(25, min(100, len(feature_cols) // 6))
        else:  # Low memory system
            chunk_size = max(10, min(50, len(feature_cols) // 8))
        
        # Create result dataframe more efficiently
        # Only start with essential columns to save memory
        essential_cols = []
        for col in df.columns:
            if col in feature_cols or any(col.startswith(f'{fc}_lag_') for fc in feature_cols):
                essential_cols.append(col)
        
        # Start with empty dataframe and copy over original columns as needed
        result_df = pd.DataFrame(index=df.index)
        
        print(f"Creating lagged features using {N_CORES} cores with JIT acceleration")
        overall_start = time.time()
        total_processed = 0
        
        # Use optimal backend based on system
        backend = "threading"  # Threading is usually more efficient for NumPy operations
        
        # Process in chunks to manage memory
        total_chunks = (len(feature_cols) + chunk_size - 1) // chunk_size
        
        with tqdm(total=len(feature_cols), desc="Creating lagged features", unit="cols") as pbar:
            for i in range(0, len(feature_cols), chunk_size):
                # Get chunk columns
                chunk_end = min(i + chunk_size, len(feature_cols))
                chunk_cols = feature_cols[i:chunk_end]
                chunk_size_actual = len(chunk_cols)
                chunk_idx = i // chunk_size + 1
                
                print(f"Processing column chunk {chunk_idx}/{total_chunks} ({chunk_size_actual} columns)")
                chunk_start_time = time.time()
                
        # Pre-extract column values for efficiency and ensure they are float64
                col_values = {}
                for col in chunk_cols:
                    try:
                        # Try to convert to float64, skip columns that can't be converted
                        col_values[col] = df[col].astype(np.float64).values
                    except (ValueError, TypeError) as e:
                        print(f"WARNING: Skipping column {col} - cannot convert to numeric: {str(e)}")
                        continue
                  # Process columns in parallel - only process columns that were successfully converted
                results = Parallel(n_jobs=N_CORES, backend=backend)(
                    delayed(process_column_lags)(col, col_values[col], lag_hours) 
                    for col in chunk_cols if col in col_values
                )
                  # Extract results and timing info with error handling
                result_dicts = []
                timing_info = []
                for r in results:
                    if r and len(r) == 2:  # Ensure result is valid
                        result_dicts.append(r[0])
                        timing_info.append(r[1])
                    else:
                        print(f"WARNING: Received invalid result from parallel processing: {r}")
                
                # Log performance stats
                batch_time = time.time() - chunk_start_time
                avg_col_time = sum(timing_info) / len(timing_info) if timing_info else 0
                print(f"Chunk {chunk_idx} processed in {batch_time:.2f}s " 
                      f"(avg {avg_col_time:.4f}s/column, {batch_time/chunk_size_actual:.4f}s/column with overhead)")
                
                # Add original columns first if not already present
                for col in chunk_cols:
                    if col not in result_df.columns:
                        result_df[col] = df[col]
                
                # Add results to dataframe
                for col_result in result_dicts:
                    for col_name, values in col_result.items():
                        result_df[col_name] = values
                
                # Update progress
                pbar.update(chunk_size_actual)
                total_processed += chunk_size_actual
                
                # Estimate time remaining
                elapsed = time.time() - overall_start
                if total_processed > 0:
                    cols_per_sec = total_processed / elapsed
                    remaining_cols = len(feature_cols) - total_processed
                    est_remaining = remaining_cols / cols_per_sec if cols_per_sec > 0 else 0
                    print(f"Processed {total_processed}/{len(feature_cols)} columns. "
                          f"Est. remaining: {est_remaining:.1f}s")
                
                # Clean up to save memory
                del results, result_dicts, timing_info, col_values
                gc.collect()
        
        return result_df
    
    def create_rolling_stats(self, df, feature_cols=None, window_sizes=None, stats=None,
                          temp_dir='temp_rolling_stats', use_temp_files=True):
        """
        Create rolling statistics features using parallel processing with JIT optimization
        
        Args:
            df (pd.DataFrame): Input dataframe
            feature_cols (list): Columns to create rolling stats for
            window_sizes (list): List of window sizes in hours
            stats (list): List of statistics to compute ('mean', 'std', 'min', 'max')
            temp_dir (str): Directory to store temporary files
            use_temp_files (bool): Whether to use temporary files for memory efficiency
            
        Returns:
            pd.DataFrame: Dataframe with rolling statistics features
        """
        import time
        
        # Implement JIT-compiled functions for rolling statistics
        # These are much faster than pandas rolling functions for large datasets
        @njit(cache=True, fastmath=True)
        def rolling_mean_jit(values, window_size):
            """JIT-optimized rolling mean calculation"""
            n = len(values)
            result = np.full(n, np.nan, dtype=np.float64)
            
            # Handle edge case
            if window_size > n:
                return result
                
            # Calculate cumulative sum for efficient rolling mean
            cumsum = np.zeros(n + 1, dtype=np.float64)
            mask = ~np.isnan(values)
            valid_count = np.zeros(n + 1, dtype=np.int32)
            
            # First pass: calculate cumsum and valid counts
            for i in range(n):
                if mask[i]:
                    cumsum[i+1] = cumsum[i] + values[i]
                    valid_count[i+1] = valid_count[i] + 1
                else:
                    cumsum[i+1] = cumsum[i]
                    valid_count[i+1] = valid_count[i]
                    
            # Second pass: calculate means
            for i in range(window_size-1, n):
                window_valid_count = valid_count[i+1] - valid_count[i+1-window_size]
                if window_valid_count > 0:
                    result[i] = (cumsum[i+1] - cumsum[i+1-window_size]) / window_valid_count
            
            return result
                
        @njit(cache=True, fastmath=True)
        def rolling_std_jit(values, window_size):
            """JIT-optimized rolling std calculation"""
            n = len(values)
            result = np.full(n, np.nan, dtype=np.float64)
            
            # Handle edge case
            if window_size > n:
                return result
                
            # Calculate means first
            means = rolling_mean_jit(values, window_size)
            
            # Calculate rolling variance using means
            for i in range(window_size-1, n):
                window_start = i - window_size + 1
                window = values[window_start:i+1]
                window_mean = means[i]
                
                if np.isnan(window_mean):
                    continue
                
                # Calculate variance
                var_sum = 0.0
                count = 0
                for j in range(window_size):
                    if not np.isnan(window[j]):
                        var_sum += (window[j] - window_mean) ** 2
                        count += 1
                
                if count > 1:
                    result[i] = np.sqrt(var_sum / count)
            
            return result
                
        @njit(cache=True, fastmath=True)
        def rolling_min_jit(values, window_size):
            """JIT-optimized rolling min calculation"""
            n = len(values)
            result = np.full(n, np.nan, dtype=np.float64)
            
            # Handle edge case
            if window_size > n:
                return result
                
            # Calculate rolling min
            for i in range(window_size-1, n):
                window_start = i - window_size + 1
                window = values[window_start:i+1]
                
                # Filter out NaNs
                valid_values = []
                for j in range(window_size):
                    if not np.isnan(window[j]):
                        valid_values.append(window[j])
                
                if valid_values:
                    result[i] = min(valid_values)
            
            return result
                
        @njit(cache=True, fastmath=True)
        def rolling_max_jit(values, window_size):
            """JIT-optimized rolling max calculation"""
            n = len(values)
            result = np.full(n, np.nan, dtype=np.float64)
            
            # Handle edge case
            if window_size > n:
                return result
                
            # Calculate rolling max
            for i in range(window_size-1, n):
                window_start = i - window_size + 1
                window = values[window_start:i+1]
                
                # Filter out NaNs
                valid_values = []
                for j in range(window_size):
                    if not np.isnan(window[j]):
                        valid_values.append(window[j])
                
                if valid_values:
                    result[i] = max(valid_values)
            
            return result
        
        # Function to process one column and window size combination with JIT
        def process_column_window(col, window, stats_list):
            """Process rolling stats for one column with JIT optimization"""
            start_time = time.time()
            
            # Get column values as numpy array for JIT processing
            values = df[col].values
            
            # Create results dict
            result_dict = {}
            
            # Calculate requested statistics using JIT functions
            for stat in stats_list:                
                if stat == 'mean':
                    result = rolling_mean_jit(values, window)
                    result_dict[f'{col}_roll_{window}h_mean'] = result
                elif stat == 'std':
                    result = rolling_std_jit(values, window)
                    result_dict[f'{col}_roll_{window}h_std'] = result
                elif stat == 'min':
                    result = rolling_min_jit(values, window)
                    result_dict[f'{col}_roll_{window}h_min'] = result
                elif stat == 'max':
                    result = rolling_max_jit(values, window)
                    result_dict[f'{col}_roll_{window}h_max'] = result
            
            elapsed = time.time() - start_time
            return result_dict, elapsed
        
        if feature_cols is None:
            # Use important columns if they've been identified
            if self.config['important_columns']:
                feature_cols = self.config['important_columns']
                print(f"Creating rolling stats for {len(feature_cols)} important columns instead of all {len(df.columns)}")
            else:
                # Just use a subset of columns if we haven't identified important ones
                numeric_cols = df.select_dtypes(include=['number']).columns
                # Take at most 50 columns
                feature_cols = numeric_cols[:min(50, len(numeric_cols))].tolist()
                print(f"Limited rolling stats to {len(feature_cols)} columns")
            
        if window_sizes is None:
            window_sizes = self.config['rolling_window_sizes']
            
        if stats is None:
            # Reduce the number of statistics generated
            stats = ['mean', 'std']  # Removed min/max to reduce column count
            
        # Filter out non-numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        numeric_feature_cols = [col for col in feature_cols if col in numeric_cols]
        
        # Start with just the non-numeric columns to avoid copying the entire dataframe
        non_numeric_cols = [col for col in df.columns if col not in numeric_cols]
        result_df = df[non_numeric_cols].copy() if non_numeric_cols else pd.DataFrame(index=df.index)
        
        # Create temporary directory if needed
        if use_temp_files and not os.path.exists(temp_dir):
            os.makedirs(temp_dir, exist_ok=True)
        
        # Generate all tasks
        all_tasks = []
        for col in numeric_feature_cols:
            for window in window_sizes:
                all_tasks.append((col, window, stats))
        
        # Calculate optimal batch size based on number of tasks and cores
        total_tasks = len(all_tasks)
        batch_size = max(10, min(100, total_tasks // (N_CORES * 2)))
        
        # Process tasks in parallel with progress bar
        print(f"Processing rolling statistics using {N_CORES} cores with JIT acceleration")
        
        # Track overall progress
        overall_start = time.time()
        processed_results = []
        total_processed = 0
        
        # Use tqdm to create a progress bar for tasks
        with tqdm(total=total_tasks, desc="Creating rolling statistics", position=0) as pbar:
            # Process in batches to update the progress bar more frequently
            for batch_start in range(0, total_tasks, batch_size):
                batch_end = min(batch_start + batch_size, total_tasks)
                batch_tasks = all_tasks[batch_start:batch_end]
                batch_size_actual = len(batch_tasks)
                
                print(f"Processing batch {batch_start//batch_size + 1}/{(total_tasks+batch_size-1)//batch_size} " 
                      f"({batch_size_actual} tasks)")
                batch_start_time = time.time()
                
                # Process batch in parallel using threading for numpy operations
                batch_results = Parallel(n_jobs=N_CORES, backend="threading")(
                    delayed(process_column_window)(col, window, stat_list) 
                    for col, window, stat_list in batch_tasks
                )
                
                # Extract results and timing info
                results_dicts = [r[0] for r in batch_results]
                timing_info = [r[1] for r in batch_results]
                
                # Process results
                processed_results.extend(results_dicts)
                
                # Log performance stats
                batch_time = time.time() - batch_start_time
                avg_task_time = sum(timing_info) / len(timing_info) if timing_info else 0
                print(f"Batch processed in {batch_time:.2f}s " 
                      f"(avg {avg_task_time:.4f}s/task, {batch_time/batch_size_actual:.4f}s/task with overhead)")
                
                # Update progress
                pbar.update(batch_size_actual)
                total_processed += batch_size_actual
                
                # Estimate time remaining
                elapsed = time.time() - overall_start
                if total_processed > 0:
                    tasks_per_sec = total_processed / elapsed
                    remaining_tasks = total_tasks - total_processed
                    est_remaining = remaining_tasks / tasks_per_sec if tasks_per_sec > 0 else 0
                    print(f"Processed {total_processed}/{total_tasks} tasks. "
                          f"Est. remaining: {est_remaining:.1f}s")
                
                # Free memory
                del batch_results, results_dicts, timing_info
                gc.collect()
        
        # Create small dataframes and save to disk if needed
        if use_temp_files:
            temp_files = []
            batch_size = max(1, len(processed_results) // 10)  # Split into 10 batches max
            
            for i in range(0, len(processed_results), batch_size):
                batch_chunk = processed_results[i:i+batch_size]
                
                # Combine dictionaries in this chunk
                combined_dict = {}
                for result_dict in batch_chunk:
                    combined_dict.update(result_dict)
                
                # Convert to DataFrame
                batch_df = pd.DataFrame({col: series for col, series in combined_dict.items()}, index=df.index)
                
                # Save to disk
                temp_file = os.path.join(temp_dir, f'rolling_stats_batch_{i//batch_size}.pkl')
                batch_df.to_pickle(temp_file)
                temp_files.append(temp_file)
                
                # Free memory
                del batch_df, combined_dict, batch_chunk
                gc.collect()
            
            # Clear the main results to save memory
            del processed_results
            gc.collect()
            
            # Merge and concatenate all batches
            print("Merging rolling statistics batches")
            for file_idx, file_path in enumerate(tqdm(temp_files, desc="Merging batches", position=0)):
                try:
                    batch_df = pd.read_pickle(file_path)
                    
                    # Add original columns as needed
                    for col in batch_df.columns:
                        base_col = col.split('_roll_')[0] if '_roll_' in col else col
                        if base_col in numeric_feature_cols and base_col not in result_df.columns:
                            result_df[base_col] = df[base_col].copy()
                    
                    # Merge with result
                    result_df = pd.concat([result_df, batch_df], axis=1)
                    
                    # Clean up
                    del batch_df
                    os.remove(file_path)
                    gc.collect()
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
        else:
            # Process results in memory (for smaller datasets)
            for result_dict in tqdm(processed_results, desc="Combining results", position=0):
                for col_name, values in result_dict.items():
                    base_col = col_name.split('_roll_')[0] if '_roll_' in col_name else col_name
                    
                    # Add original column if needed
                    if base_col in numeric_feature_cols and base_col not in result_df.columns:
                        result_df[base_col] = df[base_col].copy()
                    
                    # Add rolling stat column
                    result_df[col_name] = values
        
        # Remove duplicate columns if any
        result_df = result_df.loc[:, ~result_df.columns.duplicated()]
          # Force final garbage collection
        gc.collect()
        
        return result_df
    
    def create_temperature_differential_features(self, df):
        """
        Create temperature differential features between adjacent zones using parallel processing
        with special focus on detecting temperature drop patterns characteristic of accretion formation
        
        Args:
            df (pd.DataFrame): Input dataframe with zone temperature columns
            
        Returns:
            pd.DataFrame: Dataframe with differential features and accretion-specific temperature patterns
        """
        # Start with a copy of the dataframe (avoid copying all columns at once)
        result_df = pd.DataFrame(index=df.index)
        
        # Identify zone columns - only use ZONE_ columns
        zone_cols = [col for col in df.columns if col.startswith('ZONE_')]
        
        # Add original zone columns first 
        for col in zone_cols:
            result_df[col] = df[col].copy()
        
        # Function to calculate differential between two zones
        def calculate_differential(zone1, zone2):
            diff_name = f'diff_{zone1}_{zone2}'
            return diff_name, df[zone1] - df[zone2]
        
        # Function to calculate gradient for a zone
        def calculate_gradient(zone):
            grad_name = f'gradient_{zone}'
            return grad_name, df[zone].diff() / df.index.to_series().diff().dt.total_seconds()
            
        # Function to calculate rate of change (used to detect rapid temperature drops)
        def calculate_rate_of_change(zone):
            # Calculate percentage change over 3 periods
            roc_name = f'roc_{zone}'
            pct_change = df[zone].pct_change(periods=3) * 100
            return roc_name, pct_change
            
        # Function to detect cooling trends in a zone
        def calculate_cooling_trend(zone):
            # Cooling is defined as 3 consecutive periods of decreasing temperature
            cooling_name = f'cooling_{zone}'
            # Calculate if current value is less than previous
            decreasing = (df[zone].diff() < 0).astype(int)
            # Calculate rolling sum of decreasing flags (3+ consecutive drops = cooling trend)
            cooling_trend = decreasing.rolling(window=3, min_periods=3).sum() >= 3
            return cooling_name, cooling_trend.astype(int)
        
        # Prepare the differential tasks
        diff_tasks = []
        for i in range(len(zone_cols) - 1):
            diff_tasks.append((zone_cols[i], zone_cols[i+1]))
            
        # Also add some non-adjacent zone differentials that can help detect accretion
        # Accretion often causes temperature patterns across multiple zones
        if len(zone_cols) >= 4:
            # Add differentials between zones that are 2 and 3 positions apart
            for i in range(len(zone_cols) - 2):
                diff_tasks.append((zone_cols[i], zone_cols[i+2]))  # 2 positions apart
                
            for i in range(len(zone_cols) - 3):
                diff_tasks.append((zone_cols[i], zone_cols[i+3]))  # 3 positions apart
        
        print(f"Creating zone differentials using {N_CORES} cores")
        
        # Process zone differentials in parallel
        results = Parallel(n_jobs=N_CORES, backend="threading")(
            delayed(calculate_differential)(zone1, zone2) 
            for zone1, zone2 in tqdm(diff_tasks, desc="Creating zone differentials")
        )
        
        # Add differential results
        for name, series in results:
            result_df[name] = series
        
        print(f"Creating gradient features using {N_CORES} cores")
        
        # Process gradients in parallel
        grad_results = Parallel(n_jobs=N_CORES, backend="threading")(
            delayed(calculate_gradient)(zone) 
            for zone in tqdm(zone_cols, desc="Creating gradient features")
        )
        
        # Add gradient results
        for name, series in grad_results:
            result_df[name] = series
            
        print("Creating rate of change features")
        
        # Process rate of change in parallel
        roc_results = Parallel(n_jobs=N_CORES, backend="threading")(
            delayed(calculate_rate_of_change)(zone) 
            for zone in tqdm(zone_cols, desc="Creating rate of change features")
        )
        
        # Add rate of change results
        for name, series in roc_results:
            result_df[name] = series
            
        print("Creating cooling trend features")
        
        # Process cooling trends in parallel
        cooling_results = Parallel(n_jobs=N_CORES, backend="threading")(
            delayed(calculate_cooling_trend)(zone) 
            for zone in tqdm(zone_cols, desc="Creating cooling trend features")
        )
        
        # Add cooling trend results
        for name, series in cooling_results:
            result_df[name] = series
            
        # Create multi-zone cooling pattern detection
        # Accretion typically causes a distinct cooling pattern across multiple adjacent zones
        if len(zone_cols) >= 3:
            cooling_cols = [f'cooling_{zone}' for zone in zone_cols]
            if all(col in result_df.columns for col in cooling_cols):
                # Count how many consecutive zones are cooling
                cooling_counts = pd.DataFrame(index=df.index)
                for i in range(len(cooling_cols) - 2):
                    # Three consecutive zones cooling
                    pattern_name = f'cooling_pattern_{i}_{i+2}'
                    pattern = (
                        (result_df[cooling_cols[i]] == 1) & 
                        (result_df[cooling_cols[i+1]] == 1) & 
                        (result_df[cooling_cols[i+2]] == 1)
                    )
                    cooling_counts[pattern_name] = pattern.astype(int)
                
                # Add summary feature: any cooling pattern detected
                if not cooling_counts.empty:
                    result_df['any_cooling_pattern'] = (cooling_counts.sum(axis=1) > 0).astype(int)
                    # Count number of cooling patterns
                    result_df['cooling_pattern_count'] = cooling_counts.sum(axis=1)        # Add remaining columns from the original dataframe
        print("Adding remaining columns with progress tracking...")
        remaining_cols = [col for col in df.columns if col not in result_df.columns and col not in zone_cols]
        for col in tqdm(remaining_cols, desc="Adding remaining columns"):
            if col in df.columns:  # Verify column exists before attempting to copy
                try:
                    result_df[col] = df[col].copy()
                except Exception as e:  # Skip problematic columns and print warning
                    print(f"Warning: Could not add column {col} - {str(e)}")
                    continue
            else:
                print(f"Warning: Column {col} not found in dataframe, skipping")
        
        return result_df
        
    def detect_temp_anomalies(self, df, window_size=24, threshold=2.0):
        """
        Detect anomalies in temperature columns with special focus on temperature drops
        which are indicators of accretion formation
        
        Args:
            df (pd.DataFrame): Input dataframe
            window_size (int): Size of rolling window in hours
            threshold (float): Number of std devs for anomaly threshold
            
        Returns:
            pd.DataFrame: Dataframe with anomaly columns and specialized temperature drop features
        """
        # JIT-compiled function for anomaly detection (2x-10x speedup over pandas rolling)
        @njit(cache=True, fastmath=True)
        def detect_anomalies_jit(values, window_size, threshold):
            """Numba-optimized anomaly detection function"""
            n = len(values)
            anomalies = np.zeros(n, dtype=np.float64)
            drop_magnitudes = np.zeros(n, dtype=np.float64)
            
            # Need at least window_size points before we can detect anomalies
            if n <= window_size:
                return anomalies, drop_magnitudes
                
            # For each point calculate if it's an anomaly based on previous window
            for i in range(window_size, n):
                # Use previous window_size points to determine baseline
                window = values[i-window_size:i]
                
                # Remove NaN values from window
                valid_window = window[~np.isnan(window)]
                
                if len(valid_window) > window_size // 2:  # At least half the window should be valid
                    window_mean = np.mean(valid_window)
                    window_std = np.std(valid_window)
                    
                    # Avoid division by zero and very small stdevs
                    if window_std < 0.001:
                        window_std = 0.001
                        
                    current_val = values[i]
                    upper_bound = window_mean + (threshold * window_std)
                    lower_bound = window_mean - (threshold * window_std)
                    
                    # Store drop magnitude as a percentage of the window mean
                    # Only for values below the lower bound (temperature drops)
                    if current_val < lower_bound:
                        anomalies[i] = -1  # Negative anomaly (temperature drop)
                        # Calculate magnitude as percentage drop from mean
                        if window_mean > 0:
                            drop_magnitudes[i] = (window_mean - current_val) / window_mean * 100
                    elif current_val > upper_bound:
                        anomalies[i] = 1  # Positive anomaly (temperature spike)
                        
            return anomalies, drop_magnitudes
        
        # Select only temperature columns
        temp_cols = [col for col in df.columns if 'ZONE_' in col or 'qrt_' in col or 'shell_' in col]
        
        # Create result dataframe with index only to start
        result_df = pd.DataFrame(index=df.index)
        
        # First copy non-temperature columns in batches
        non_temp_cols = [col for col in df.columns if col not in temp_cols]
        
        print(f"Copying {len(non_temp_cols)} non-temperature columns")
        batch_size = 500
        for i in range(0, len(non_temp_cols), batch_size):
            batch_cols = non_temp_cols[i:i+batch_size]
            result_df = pd.concat([result_df, df[batch_cols]], axis=1)
            gc.collect()
        
        # Process temperature columns sequentially (avoid parallelism to prevent pickling errors)
        print(f"Processing {len(temp_cols)} temperature columns sequentially with JIT acceleration")
        print(f"Memory usage before anomaly detection: {get_memory_usage():.2f} GB")
        
        # Use small batch size to manage memory
        batch_size = 10  # Process and add just 10 columns at a time
        total_batches = (len(temp_cols) + batch_size - 1) // batch_size
        print(f"Total batches to process: {total_batches} (batch size: {batch_size})")
        
        # Track overall progress
        overall_start = time.time()
        total_processed = 0
        
        # Process temperature columns in small batches
        with tqdm(total=len(temp_cols), desc="Detecting temperature anomalies", unit="cols") as pbar:
            for batch_idx, batch_start in enumerate(range(0, len(temp_cols), batch_size)):
                batch_end = min(batch_start + batch_size, len(temp_cols))
                batch_cols = temp_cols[batch_start:batch_end]
                
                print(f"Processing batch {batch_idx+1}/{total_batches} ({len(batch_cols)} columns)")
                batch_start_time = time.time()
                
                # Create a temporary batch dataframe
                temp_batch_df = pd.DataFrame(index=df.index)
                
                # Process each column in the batch sequentially
                for col in batch_cols:
                    # Add the original column
                    temp_batch_df[col] = df[col]
                    
                    try:
                        # Process anomaly detection
                        anomaly_col = f'{col}_anomaly'
                        drop_col = f'{col}_drop_pct'
                        
                        # Get column values as numpy array
                        values = df[col].values.astype(np.float64)
                        
                        # Detect anomalies using JIT function - now returns drop magnitudes too
                        anomalies, drop_magnitudes = detect_anomalies_jit(values, window_size, threshold)
                        
                        # Add anomaly and drop magnitude columns
                        temp_batch_df[anomaly_col] = anomalies
                        temp_batch_df[drop_col] = drop_magnitudes
                        
                        # Add an additional feature for consecutive drops
                        # This can help identify persistent cooling patterns characteristic of accretion
                        temp_batch_df[f'{col}_consec_drops'] = (temp_batch_df[anomaly_col] == -1).astype(int).rolling(
                            window=12, min_periods=1).sum()
                        
                    except Exception as e:
                        print(f"Error processing column {col}: {e}")
                
                # Add batch to main dataframe
                result_df = pd.concat([result_df, temp_batch_df], axis=1)
                
                # Update progress
                batch_time = time.time() - batch_start_time
                pbar.update(len(batch_cols))
                total_processed += len(batch_cols)
                
                # Estimate time remaining
                elapsed = time.time() - overall_start
                if total_processed > 0:
                    cols_per_sec = total_processed / elapsed
                    remaining_cols = len(temp_cols) - total_processed
                    est_remaining = remaining_cols / cols_per_sec if cols_per_sec > 0 else 0
                    print(f"Processed {total_processed}/{len(temp_cols)} columns in {elapsed:.1f}s. "
                          f"Est. remaining: {est_remaining:.1f}s")
                
                # Aggressive memory cleanup
                del temp_batch_df, values, anomalies, drop_magnitudes
                for _ in range(3):  # Multiple GC calls can help recover more memory
                    gc.collect()
                
                # Log memory usage
                current_memory = get_memory_usage()
                print(f"Memory usage after batch {batch_idx+1}: {current_memory:.2f} GB")
        
        # Create aggregate zone drop features that can indicate early accretion formation
        zone_cols = [col for col in temp_cols if 'ZONE_' in col]
        if zone_cols:
            # Create features for average and max drop percentage across all zones
            drop_cols = [f'{col}_drop_pct' for col in zone_cols if f'{col}_drop_pct' in result_df.columns]
            if drop_cols:
                result_df['all_zones_avg_drop_pct'] = result_df[drop_cols].mean(axis=1)
                result_df['all_zones_max_drop_pct'] = result_df[drop_cols].max(axis=1)
                
                # Create count of zones with significant drops (potentially indicating accretion formation)
                result_df['zones_with_drops'] = (result_df[drop_cols] > 5).sum(axis=1)  # Count zones with >5% drops
        
        # Final cleanup
        gc.collect()
        return result_df
    
    def create_material_ratio_features(self, df):
        """
        Create features based on material ratios and quality metrics
        with special focus on indicators of accretion:
        - Increased coal consumption 
        - Quality shifts (decrease in Quality A, increase in Quality B and fines)
        - Energy efficiency metrics
        
        Args:
            df (pd.DataFrame): Input dataframe with MIS report columns
            
        Returns:
            pd.DataFrame: Dataframe with ratio features and quality shift indicators
        """
        result_df = df.copy()
        
        # Calculate basic material ratios if required columns exist
        if all(col in df.columns for col in ['mis_IRON ORE CONSUMPTION', 'mis_GROSS COAL CONSUMPTION']):
            # Calculate ore to coal ratio - drops during accretion when more coal is needed
            result_df['ratio_ore_coal'] = df['mis_IRON ORE CONSUMPTION'] / df['mis_GROSS COAL CONSUMPTION']
            
            # Add rolling indicators to track changes over time
            result_df['ratio_ore_coal_pct_change'] = result_df['ratio_ore_coal'].pct_change(periods=3)
            result_df['ratio_ore_coal_ma7'] = result_df['ratio_ore_coal'].rolling(window=7, min_periods=1).mean()
            
            # Calculate efficiency indicator (how much ore processed per unit of coal)
            # Decreasing efficiency can indicate accretion formation
            if 'mis_DRI PRODUCTION' in df.columns:
                result_df['efficiency_coal_production'] = df['mis_DRI PRODUCTION'] / df['mis_GROSS COAL CONSUMPTION']
                result_df['efficiency_pct_change'] = result_df['efficiency_coal_production'].pct_change(periods=3)
                result_df['efficiency_ma7'] = result_df['efficiency_coal_production'].rolling(window=7, min_periods=1).mean()
        
        # Track coal consumption increases
        if 'mis_GROSS COAL CONSUMPTION' in df.columns:
            # Calculate moving averages to smooth out daily variations
            result_df['coal_consumption_ma7'] = df['mis_GROSS COAL CONSUMPTION'].rolling(window=7, min_periods=1).mean()
            
            # Calculate percentage change from baseline (7-day vs 30-day average)
            # Higher coal usage compared to baseline can indicate accretion
            coal_ma7 = df['mis_GROSS COAL CONSUMPTION'].rolling(window=7, min_periods=1).mean()
            coal_ma30 = df['mis_GROSS COAL CONSUMPTION'].rolling(window=30, min_periods=7).mean()
            
            # Avoid division by zero
            valid_indexes = coal_ma30 > 0
            result_df['coal_consumption_vs_baseline'] = np.nan
            result_df.loc[valid_indexes, 'coal_consumption_vs_baseline'] = (
                coal_ma7[valid_indexes] / coal_ma30[valid_indexes] - 1) * 100  # as percentage
                
            # Flag potential accretion periods based on increased coal consumption
            result_df['high_coal_consumption'] = (result_df['coal_consumption_vs_baseline'] > 15).astype(int)
        
        # Quality metrics - track shifts in quality that can indicate accretion
        if all(col in df.columns for col in ['mis_DRI LUMPS', 'mis_DRI PELLETS', 'mis_DRI FINES']):
            # Calculate total DRI production
            result_df['total_dri'] = (
                df['mis_DRI LUMPS'] + df['mis_DRI PELLETS'] + df['mis_DRI FINES']
            ).replace(0, np.nan)  # Replace zeros with NaN to avoid division issues
            
            # Calculate quality ratios - during accretion, lumps decrease, pellets and fines increase
            for quality in ['mis_DRI LUMPS', 'mis_DRI PELLETS', 'mis_DRI FINES']:
                quality_col = quality.replace('mis_', '')
                ratio_col = f'ratio_{quality_col}'
                
                # Skip calculation if total_dri has no valid values
                if result_df['total_dri'].notna().sum() == 0:
                    continue
                
                # Calculate ratio of each quality to total production
                result_df[ratio_col] = df[quality] / result_df['total_dri']
                
                # Calculate moving average and percentage change
                result_df[f'{ratio_col}_ma7'] = result_df[ratio_col].rolling(window=7, min_periods=1).mean()
                result_df[f'{ratio_col}_pct_change'] = result_df[ratio_col].pct_change(periods=3) * 100
            
            # Create features specifically tracking quality shifts that could indicate accretion
            # During accretion, we expect LUMPS to decrease and PELLETS/FINES to increase
            if 'ratio_DRI LUMPS' in result_df.columns and 'ratio_DRI PELLETS' in result_df.columns:
                # Quality shift indicator: decreasing lumps and increasing pellets+fines
                lump_decreasing = (result_df['ratio_DRI LUMPS_pct_change'] < -5).astype(int)
                pellet_increasing = (result_df['ratio_DRI PELLETS_pct_change'] > 5).astype(int)
                
                # Combined indicator: both conditions are true
                result_df['quality_shift_indicator'] = (lump_decreasing & pellet_increasing).astype(int)
                
                # Calculate spread between quality metrics - widens during accretion
                result_df['lumps_to_pellets_spread'] = (
                    result_df['ratio_DRI LUMPS'] - result_df['ratio_DRI PELLETS']
                )
                
                # Calculate rolling Z-score of the spread to detect unusual changes
                spread_mean = result_df['lumps_to_pellets_spread'].rolling(window=30, min_periods=7).mean()
                spread_std = result_df['lumps_to_pellets_spread'].rolling(window=30, min_periods=7).std()
                
                # Calculate Z-score where std is not near zero
                valid_std = spread_std > 0.001
                result_df['spread_zscore'] = np.nan
                result_df.loc[valid_std, 'spread_zscore'] = (
                    (result_df['lumps_to_pellets_spread'] - spread_mean) / spread_std
                ).loc[valid_std]
                
                # Extreme negative values indicate potential accretion
                result_df['spread_anomaly'] = (result_df['spread_zscore'] < -2).astype(int)
        
        # Create combined accretion indicator based on multiple factors
        accretion_indicators = []
        
        if 'high_coal_consumption' in result_df.columns:
            accretion_indicators.append('high_coal_consumption')
            
        if 'quality_shift_indicator' in result_df.columns:
            accretion_indicators.append('quality_shift_indicator')
            
        if 'spread_anomaly' in result_df.columns:
            accretion_indicators.append('spread_anomaly')
            
        # Add temperature-based indicators if available
        if 'zones_with_drops' in result_df.columns:
            # Consider it an indicator if more than 2 zones show temperature drops
            result_df['temp_drop_indicator'] = (result_df['zones_with_drops'] > 2).astype(int)
            accretion_indicators.append('temp_drop_indicator')
        
        # Create a combined score (sum of all indicators)
        if accretion_indicators:
            result_df['accretion_indicator_score'] = result_df[accretion_indicators].sum(axis=1)
            
        return result_df
    
    def create_accretion_indicator_features(self, df):
        """
        Create composite accretion indicator features by combining signals from:
        - Temperature drops in multiple zones
        - Quality shifts (decrease in Quality A, increase in Quality B and fines)
        - Increased coal/fuel consumption trends
        
        These composite indicators serve as early warning features for accretion formation.
        
        Args:
            df (pd.DataFrame): Input dataframe with all previously created features
            
        Returns:
            pd.DataFrame: Dataframe with composite accretion indicator features
        """
        result_df = df.copy()
        
        # 1. Create temperature-based accretion risk indicator
        # Identify columns with temperature anomaly information
        temp_anomaly_cols = [col for col in df.columns if 'temp_anomaly_' in col]
        drop_magnitude_cols = [col for col in df.columns if 'drop_magnitude_' in col]
        cooling_trend_cols = [col for col in df.columns if 'cooling_trend_' in col]
        
        if temp_anomaly_cols:
            # Count number of zones showing temperature drops at the same time
            result_df['accretion_risk_temp_zones'] = df[temp_anomaly_cols].apply(
                lambda x: np.sum(x < 0), axis=1
            )
            
            # Calculate average drop magnitude across zones
            if drop_magnitude_cols:
                result_df['accretion_risk_drop_magnitude'] = df[drop_magnitude_cols].mean(axis=1)
                
                # Create severity levels based on magnitude ranges
                result_df['accretion_severity_temp'] = pd.cut(
                    result_df['accretion_risk_drop_magnitude'], 
                    bins=[0, 2, 5, 10, 100], 
                    labels=['None', 'Low', 'Medium', 'High']
                ).fillna('None')
            
            # Count cooling trends across zones
            if cooling_trend_cols:
                result_df['accretion_risk_cooling_trends'] = df[cooling_trend_cols].sum(axis=1)
        
        # 2. Create material consumption and quality-based indicators
        material_features = []
        
        # Coal consumption indicators
        if 'ratio_ore_coal_pct_change' in df.columns:
            material_features.append('ratio_ore_coal_pct_change')
            # Flag significant drops in ore/coal ratio (meaning more coal used per ore)
            result_df['accretion_risk_coal_increase'] = (
                df['ratio_ore_coal_pct_change'] < -0.05
            ).astype(int)
        
        # Efficiency indicators  
        if 'efficiency_pct_change' in df.columns:
            material_features.append('efficiency_pct_change')
            # Flag significant drops in efficiency
            result_df['accretion_risk_efficiency_drop'] = (
                df['efficiency_pct_change'] < -0.03
            ).astype(int)
            
        # Quality shift indicators
        quality_features = []
        if 'quality_A_pct_change' in df.columns:
            quality_features.append('quality_A_pct_change')
            # Flag quality A decrease
            result_df['accretion_risk_qualityA_drop'] = (
                df['quality_A_pct_change'] < -0.02
            ).astype(int)
            
        if 'quality_B_pct_change' in df.columns:
            quality_features.append('quality_B_pct_change')
            # Flag quality B increase
            result_df['accretion_risk_qualityB_increase'] = (
                df['quality_B_pct_change'] > 0.02
            ).astype(int)
            
        if 'quality_fines_pct_change' in df.columns:
            quality_features.append('quality_fines_pct_change')
            # Flag fines increase
            result_df['accretion_risk_fines_increase'] = (
                df['quality_fines_pct_change'] > 0.02
            ).astype(int)
            
        # 3. Create composite accretion risk score combining temperature, material and quality indicators
        accretion_risk_columns = [col for col in result_df.columns if 'accretion_risk_' in col]
        
        if accretion_risk_columns:
            # Create weighted risk score
            # Temperature indicators get higher weight (more direct signals of accretion)
            temp_indicators = [col for col in accretion_risk_columns if any(x in col for x in ['temp', 'drop', 'cooling'])]
            other_indicators = [col for col in accretion_risk_columns if col not in temp_indicators]
            
            # Normalize and combine indicators
            if temp_indicators:
                # Normalize temperature zone count (typical kiln has 4-8 zones)
                if 'accretion_risk_temp_zones' in result_df.columns:
                    max_zones = max(8, result_df['accretion_risk_temp_zones'].max())
                    result_df['temp_risk_normalized'] = result_df['accretion_risk_temp_zones'] / max_zones
                else:
                    result_df['temp_risk_normalized'] = 0
                
                # Normalize cooling trends
                if 'accretion_risk_cooling_trends' in result_df.columns:
                    max_trends = max(5, result_df['accretion_risk_cooling_trends'].max())
                    result_df['cooling_risk_normalized'] = result_df['accretion_risk_cooling_trends'] / max_trends
                else:
                    result_df['cooling_risk_normalized'] = 0
                    
                # Normalize drop magnitude
                if 'accretion_risk_drop_magnitude' in result_df.columns:
                    max_drop = max(15, result_df['accretion_risk_drop_magnitude'].max())
                    result_df['drop_risk_normalized'] = result_df['accretion_risk_drop_magnitude'] / max_drop
                else:
                    result_df['drop_risk_normalized'] = 0
                    
                # Temperature component of risk score (60% weight)
                result_df['temp_risk_component'] = (
                    result_df['temp_risk_normalized'] * 0.3 +
                    result_df['cooling_risk_normalized'] * 0.1 + 
                    result_df['drop_risk_normalized'] * 0.2
                )
            else:
                result_df['temp_risk_component'] = 0
                
            # Other indicators (40% weight)
            other_risk_columns = []
            # Material consumption indicators (25% weight)
            material_risk_indicators = [col for col in other_indicators if any(x in col for x in ['coal', 'efficiency'])]
            if material_risk_indicators:
                result_df['material_risk_component'] = df[material_risk_indicators].mean(axis=1) * 0.25
                other_risk_columns.append('material_risk_component')
            else:
                result_df['material_risk_component'] = 0
                other_risk_columns.append('material_risk_component')
                
            # Quality indicators (15% weight)
            quality_risk_indicators = [col for col in other_indicators if any(x in col for x in ['quality', 'fines'])]
            if quality_risk_indicators:
                result_df['quality_risk_component'] = df[quality_risk_indicators].mean(axis=1) * 0.15
                other_risk_columns.append('quality_risk_component')
            else:
                result_df['quality_risk_component'] = 0
                other_risk_columns.append('quality_risk_component')
                
            # Combine all components into final risk score (0-1 scale)
            result_df['accretion_risk_score'] = result_df['temp_risk_component'] + result_df[other_risk_columns].sum(axis=1)
            
            # Add smoothed 7-day moving average risk score for trend monitoring
            result_df['accretion_risk_score_ma7'] = result_df['accretion_risk_score'].rolling(window=7, min_periods=1).mean()
            
            # Create categorical risk level
            result_df['accretion_risk_level'] = pd.cut(
                result_df['accretion_risk_score'],
                bins=[0, 0.2, 0.4, 0.6, 1.0],
                labels=['Low', 'Moderate', 'High', 'Critical']
            ).fillna('Low')
            
            # Create early warning indicator (detects upward trends in risk score)
            result_df['accretion_risk_trend'] = result_df['accretion_risk_score'].diff(periods=3)
            result_df['accretion_early_warning'] = (result_df['accretion_risk_trend'] > 0.1).astype(int)
        
        return result_df
    
    def create_target_variables(self, df, events_df):
        """
        Create target variables for ML models based on accretion events
        
        Args:
            df (pd.DataFrame): Input dataframe
            events_df (pd.DataFrame): Dataframe with accretion events
            
        Returns:
            pd.DataFrame: Dataframe with target variables
        """
        print(f"Starting to create target variables from {len(events_df)} accretion events")
        print(f"Memory usage before target variable creation: {get_memory_usage():.2f} GB")
        
        # Print column names to debug
        print(f"Available columns in events_df: {events_df.columns.tolist()}")
        
        # Map column names from the actual CSV to the expected names
        column_mapping = {
            'START_DATE': 'symptom_start',
            'CRITICAL_DATE': 'critical_formation',
            'ZONE': 'zone'
        }
        
        # Create a copy of the events DataFrame with renamed columns
        events_df_mapped = events_df.copy()
        for original_col, target_col in column_mapping.items():
            if original_col in events_df.columns:
                events_df_mapped[target_col] = events_df[original_col]
            else:
                print(f"Warning: Column '{original_col}' not found in events DataFrame")
                
        result_df = df.copy()
        
        # Initialize target columns
        result_df['target_accretion_forming'] = 0  # 1 if accretion is forming (after symptoms start, before critical)
        result_df['target_accretion_critical'] = 0  # 1 if accretion is at critical stage
        result_df['days_to_critical'] = np.nan  # Days remaining until critical accretion        
        result_df['accretion_zone'] = np.nan  # Zone where accretion is forming
        print(f"Target columns initialized")
          # Fill target variables based on events
        print(f"Processing accretion events with detailed tracking...")
        for idx, event in tqdm(enumerate(events_df_mapped.iterrows()), desc="Creating target variables"):
            # iterrows() returns a tuple of (index, Series)
            _, event_row = event  # Unpack the tuple correctly
            
            # Check if the mapped columns exist
            if 'symptom_start' not in event_row or 'critical_formation' not in event_row or 'zone' not in event_row:
                print(f"Warning: Missing required columns in event {idx}. Available columns: {event_row.index.tolist()}")
                continue
                
            symptom_start = event_row['symptom_start']
            critical_formation = event_row['critical_formation']
            affected_zone = event_row['zone']
            
            # Flag forming period
            mask_forming = (result_df.index >= symptom_start) & (result_df.index < critical_formation)
            result_df.loc[mask_forming, 'target_accretion_forming'] = 1
            
            # Flag critical point
            critical_mask = (result_df.index >= critical_formation)
            result_df.loc[critical_mask, 'target_accretion_critical'] = 1
              
            if idx % 2 == 0:  # Print progress updates every few events
                print(f"Processing event {idx+1}/{len(events_df)}, symptom start: {symptom_start}, critical: {critical_formation}")
                
            try:
                # Use a different approach to calculate days to critical
                if sum(mask_forming) > 0:
                    # Get the timestamps where mask_forming is True
                    forming_timestamps = result_df.index[mask_forming]
                    
                    # Calculate days to critical for each timestamp
                    for timestamp in forming_timestamps:
                        days_to_critical = (critical_formation - timestamp).total_seconds() / (24 * 3600)
                        result_df.loc[timestamp, 'days_to_critical'] = days_to_critical
                    
                    print(f"Calculated days to critical for {sum(mask_forming)} data points")
                else:
                    print("No forming period data points found for this event")
            except Exception as e:
                print(f"Error calculating days to critical: {e}")
                
            # Mark affected zone            
            result_df.loc[mask_forming, 'accretion_zone'] = affected_zone
            result_df.loc[critical_mask, 'accretion_zone'] = affected_zone
        
        # Fill NaN values in days_to_critical with a default value
        # Use a large value (e.g., 365 days) to indicate "no accretion forming in near future"
        default_days = 365.0  # No accretion expected in the next year
        print(f"Filling NaN values in days_to_critical with default value {default_days}")
        result_df['days_to_critical'].fillna(default_days, inplace=True)
          # Convert zone numbers to integers for use in models (if not NaN)
        if 'accretion_zone' in result_df.columns:
            # Fill NaN values with -1 (no zone affected)
            result_df['accretion_zone'] = result_df['accretion_zone'].fillna(-1)
            
            # Remap zone values to expected model classes (0, 1, 2, 3)
            # Create a mapping of existing zone values to sequential integers starting from 0
            zone_values = sorted(result_df['accretion_zone'].unique())
            # Filter out -1 as it will remain -1 (no zone affected)
            valid_zones = [z for z in zone_values if z != -1 and not pd.isna(z)]
            
            if valid_zones:
                zone_mapping = {zone: i for i, zone in enumerate(valid_zones)}
                # Keep -1 as -1 (will be handled separately during model training)
                zone_mapping[-1] = -1
                
                # Apply the mapping
                result_df['accretion_zone'] = result_df['accretion_zone'].map(
                    lambda x: zone_mapping.get(x, -1) if not pd.isna(x) else -1
                ).astype(int)
                
                print(f"Remapped zone values: Original zones {valid_zones} -> Sequential zones {sorted(i for i in zone_mapping.values() if i != -1)}")
                print(f"Unique values in accretion_zone after remapping: {sorted(result_df['accretion_zone'].unique())}")
            else:
                # If no valid zones, just convert to int
                result_df['accretion_zone'] = result_df['accretion_zone'].astype(int)
            
        return result_df
    
    def reduce_dimension(self, df, max_features=500):
        """
        Reduce the dimensionality of the dataframe to a manageable size,
        prioritizing features relevant for accretion detection
        
        Args:
            df (pd.DataFrame): Input dataframe with potentially too many columns
            max_features (int): Maximum number of features to keep
            
        Returns:
            pd.DataFrame: Dataframe with reduced dimensionality
        """
        print(f"Reducing dimensionality from {df.shape[1]} columns to max {max_features}...")
        
        # If we already have fewer columns than the max, return as is
        if df.shape[1] <= max_features:
            print(f"No dimension reduction needed, already at {df.shape[1]} columns")
            return df
            
        # Always keep these columns - highest priority
        critical_cols = [col for col in df.columns if 
                        col.startswith('target_') or 
                        col.startswith('days_to_') or
                        col == 'accretion_zone' or
                        col == 'accretion_warning_level' or
                        col == 'accretion_early_warning_score' or
                        col == 'accretion_indicator_score' or
                        col == 'accretion_early_warning_count']
        
        # Prioritize the new accretion-specific indicator features - second highest priority
        accretion_indicator_cols = [col for col in df.columns if
                                 'accretion' in col.lower() or
                                 'quality_shift' in col or
                                 'coal_consumption_vs_baseline' in col or
                                 'high_coal_consumption' in col or
                                 'temp_drop' in col or
                                 'cooling_pattern' in col or
                                 'unusual_temp_divergence' in col or
                                 'spread_zscore' in col or
                                 'zones_with_drops' in col]
        
        # Add temperature drop features - high priority
        temp_drop_cols = [col for col in df.columns if
                       '_drop_pct' in col or
                       '_consec_drops' in col or
                       'cooling_' in col][:30]
        
        # Add important temperature columns - medium priority
        temp_cols = [col for col in df.columns if 
                    ('ZONE_' in col and not col.endswith('_anomaly') and not col.startswith('diff_') and not col.startswith('gradient_')) or
                    ('shell_' in col and not col.endswith('_roll_') and not col.endswith('_lag_'))][:20]
        
        # Add some important derived temperature features - medium priority
        derived_temp_cols = [col for col in df.columns if 
                           (col.startswith('diff_') or col.startswith('gradient_') or col.startswith('roc_')) and 'ZONE_' in col][:30]
        
        # Add important operational columns (especially coal and quality metrics) - medium priority
        op_cols = []
        
        # Coal consumption columns - higher priority
        coal_cols = [col for col in df.columns if 'COAL' in col or 'coal' in col]
        op_cols.extend(coal_cols)
        
        # Quality columns - higher priority
        quality_cols = [col for col in df.columns if 'DRI' in col or 'LUMPS' in col or 'PELLETS' in col or 'FINES' in col]
        op_cols.extend(quality_cols)
        
        # Other operational columns
        op_cols.extend([col for col in df.columns if col.startswith('mis_') and col not in op_cols][:20])
        
        # Add material ratio features - medium priority
        ratio_cols = [col for col in df.columns if 
                    col.startswith('ratio_') or 
                    'efficiency' in col][:30]
        
        # Add some lagged features (but not too many) - lower priority
        # Prioritize coal, quality, and temperature drops
        important_bases = coal_cols + quality_cols + temp_drop_cols + temp_cols[:5]
        lag_cols = []
        
        # First add lag features for important columns
        for base in important_bases:
            lag_features = [col for col in df.columns if '_lag_' in col and base in col][:5]  # Limit to 5 per base
            lag_cols.extend(lag_features)
            
        # Add some more lag features if we still have space
        additional_lag_cols = [col for col in df.columns if '_lag_' in col and col not in lag_cols][:50]
        lag_cols.extend(additional_lag_cols)
        
        # Add rolling statistics features - lower priority
        roll_cols = []
        
        # First add rolling features for important columns
        for base in important_bases:
            roll_features = [col for col in df.columns if ('_roll_' in col or '_ma' in col) and base in col][:5]
            roll_cols.extend(roll_features)
            
        # Add some more rolling features if we have space
        additional_roll_cols = [col for col in df.columns if ('_roll_' in col or '_ma' in col) and col not in roll_cols][:50]
        roll_cols.extend(additional_roll_cols)
        
        # Add anomaly indicators - medium priority since these detect unusual behavior
        anomaly_cols = [col for col in df.columns if col.endswith('_anomaly')][:50]
        
        # Combine all the selected columns
        keep_cols = list(set(critical_cols + accretion_indicator_cols + temp_drop_cols + 
                          temp_cols + derived_temp_cols + op_cols + ratio_cols +
                          lag_cols + roll_cols + anomaly_cols))
        
        # Remove duplicates while preserving order (as much as possible)
        keep_cols = list(dict.fromkeys(keep_cols))
        
        # If we still have too many columns, prioritize in order of importance:
        if len(keep_cols) > max_features:
            print(f"Too many columns ({len(keep_cols)}), reducing to {max_features}...")
            
            # Always keep these critical columns
            final_cols = critical_cols.copy()
            
            # Add accretion indicators - highest priority after critical columns
            remaining = max_features - len(final_cols)
            if remaining > 0:
                indicator_to_add = min(len(accretion_indicator_cols), remaining)
                final_cols.extend(accretion_indicator_cols[:indicator_to_add])
                
            # Add temperature drop features - next priority 
            remaining = max_features - len(final_cols)
            if remaining > 0:
                drops_to_add = min(len(temp_drop_cols), remaining)
                final_cols.extend(temp_drop_cols[:drops_to_add])
                
            # Add operation columns (coal and quality) - next priority
            remaining = max_features - len(final_cols)
            if remaining > 0:
                op_to_add = min(len(op_cols), remaining)
                final_cols.extend(op_cols[:op_to_add])
                
            # Add temperature and ratio columns - next priority
            remaining = max_features - len(final_cols)
            if remaining > 0:
                temp_ratio_cols = temp_cols + derived_temp_cols + ratio_cols
                # Allocate proportionally
                temp_ratio_to_add = min(len(temp_ratio_cols), remaining)
                final_cols.extend(temp_ratio_cols[:temp_ratio_to_add])
                
            # Add anomaly columns - next priority
            remaining = max_features - len(final_cols)
            if remaining > 0:
                anomaly_to_add = min(len(anomaly_cols), remaining)
                final_cols.extend(anomaly_cols[:anomaly_to_add])
                
            # Distribute remaining slots among lag and rolling features
            remaining = max_features - len(final_cols)
            if remaining > 0:
                lag_roll_cols = lag_cols + roll_cols
                lag_roll_to_add = min(len(lag_roll_cols), remaining)
                final_cols.extend(lag_roll_cols[:lag_roll_to_add])
                
            keep_cols = final_cols
            
        # Create the reduced dataframe
        reduced_df = df[keep_cols].copy()
        
        print(f"Reduced dimensions from {df.shape[1]} to {reduced_df.shape[1]} columns")
        print(f"Feature categories retained:")
        print(f"- {len([c for c in keep_cols if c in critical_cols])} critical columns")
        print(f"- {len([c for c in keep_cols if c in accretion_indicator_cols])} accretion indicator columns")
        print(f"- {len([c for c in keep_cols if c in temp_drop_cols])} temperature drop columns")
        print(f"- {len([c for c in keep_cols if c in temp_cols + derived_temp_cols])} temperature columns") 
        print(f"- {len([c for c in keep_cols if c in op_cols])} operational columns (incl. coal and quality)")
        print(f"- {len([c for c in keep_cols if c in ratio_cols])} ratio/efficiency columns")
        print(f"- {len([c for c in keep_cols if c in lag_cols])} lagged columns")
        print(f"- {len([c for c in keep_cols if c in roll_cols])} rolling stats columns")
        print(f"- {len([c for c in keep_cols if c in anomaly_cols])} anomaly columns")
              
        return reduced_df

    def process(self, data_dir, save_dir=None, use_temp_files=True, batch_size=None, 
               max_memory_gb=None, profile=True):
        """
        Complete data processing pipeline with memory management and multi-threading
        
        Args:
            data_dir (str): Directory containing raw data
            save_dir (str): Directory to save processed data
            use_temp_files (bool): Whether to use temporary files to save memory
            batch_size (int): Optional override for batch size in processing
            max_memory_gb (float): Target maximum memory usage in GB (if None, auto-detected)
            profile (bool): Whether to profile and log detailed performance metrics
            
        Returns:
            pd.DataFrame: Processed dataframe ready for ML
        """
        
        # Add CPU and memory profiling
        def get_cpu_usage():
            """Get current CPU usage percentage across all cores"""
            return psutil.cpu_percent(interval=0.1)
            
        def log_step_metrics(step_name, start_time):
            """Log comprehensive metrics for a processing step"""
            mem_usage = get_memory_usage()
            cpu_usage = get_cpu_usage()
            elapsed = time.time() - start_time
            print(f"\n{'='*80}")
            print(f"Step: {step_name}")
            print(f"Memory usage: {mem_usage:.2f} GB")
            print(f"CPU usage: {cpu_usage:.2f}%")
            print(f"Time taken: {elapsed:.2f}s ({elapsed/60:.2f}m)")
            print(f"{'='*80}\n")
            return {"step": step_name, "memory_gb": mem_usage, "cpu_percent": cpu_usage, "time_sec": elapsed}
            
        # Auto-detect optimal batch size based on available memory
        if max_memory_gb is None:
            system_memory = psutil.virtual_memory().total / (1024 ** 3)  # GB
            max_memory_gb = min(system_memory * 0.8, 12)  # Cap at 12GB or 80% of system memory
            print(f"Auto-detected system memory: {system_memory:.1f}GB, using max {max_memory_gb:.1f}GB")
        
        if batch_size is None:
            # Adjust batch size based on available memory
            if max_memory_gb > 12:  # High memory system
                batch_size = 100
            elif max_memory_gb > 8:  # Medium memory system
                batch_size = 50
            else:  # Low memory system
                batch_size = 25
            print(f"Auto-configured batch size: {batch_size}")
            
        metrics = []  # Store step metrics for final report
        start_time = time.time()
        
        # Set up a master progress bar for the overall process
        process_steps = 10  # Now including accretion indicator features
        print(f"Starting data preprocessing pipeline using {N_CORES} cores")
        print(f"Initial memory usage: {get_memory_usage():.2f} GB")
        master_pbar = tqdm(total=process_steps, desc="Data preprocessing pipeline", position=0)
        
        # Create temp directory for intermediate files if needed
        temp_dir = os.path.join(save_dir or ".", "temp_processing")
        if use_temp_files and not os.path.exists(temp_dir):
            os.makedirs(temp_dir, exist_ok=True)
        
        # Store events data separately to avoid losing it during cleanup
        events_data = None
        
        # For backward compatibility
        def log_memory(step_name):
            """Helper function for backwards compatibility"""
            mem_usage = get_memory_usage()
            print(f"Memory usage after {step_name}: {mem_usage:.2f} GB")
            return mem_usage
            
        # Load data
        step_start = time.time()
        master_pbar.set_description(f"Loading data (Memory: {get_memory_usage():.2f} GB)")
        data = self.load_data(data_dir)
        
        # Extract events if they exist
        if 'events' in data:
            events_data = data['events'].copy()
            
        metrics.append(log_step_metrics("Data loading", step_start))
        gc.collect()
        master_pbar.update(1)
          
        # Align time series to hourly frequency
        step_start = time.time()
        master_pbar.set_description(f"Aligning time series (Memory: {get_memory_usage():.2f} GB)")
        aligned_df = self.align_time_series(data, target_freq='1h')
        
        # Clean up data dictionary to free memory
        del data
        gc.collect()
        metrics.append(log_step_metrics("Time series alignment", step_start))
        master_pbar.update(1)
        
        # Impute missing values
        step_start = time.time()
        master_pbar.set_description(f"Imputing missing values (Memory: {get_memory_usage():.2f} GB)")
        imputed_df = self.impute_missing_values(aligned_df, method='knn')
        
        # Clean up to save memory
        del aligned_df
        gc.collect()
        metrics.append(log_step_metrics("Missing value imputation", step_start))
        master_pbar.update(1)
        
        # Create lagged features - limited to important columns only
        step_start = time.time()
        master_pbar.set_description(f"Creating lagged features (Memory: {get_memory_usage():.2f} GB)")
        lagged_df = self.create_lagged_features(imputed_df, chunk_size=batch_size)
        
        # Clean up to save memory
        del imputed_df
        gc.collect()
        metrics.append(log_step_metrics("Lagged features creation", step_start))
        master_pbar.update(1)
        
        # Create rolling statistics - limited to important columns only
        step_start = time.time()
        master_pbar.set_description(f"Creating rolling statistics (Memory: {get_memory_usage():.2f} GB)")
        rolling_stats_temp_dir = os.path.join(temp_dir, "rolling_stats") if use_temp_files else None
        rolling_df = self.create_rolling_stats(lagged_df, temp_dir=rolling_stats_temp_dir, use_temp_files=use_temp_files)
        
        # Clean up to save memory
        del lagged_df
        gc.collect()
        metrics.append(log_step_metrics("Rolling statistics creation", step_start))
        master_pbar.update(1)
        
        # Create temperature differential features
        step_start = time.time()
        master_pbar.set_description(f"Creating temperature differential features (Memory: {get_memory_usage():.2f} GB)")
        temp_diff_df = self.create_temperature_differential_features(rolling_df)
        
        # Clean up to save memory
        del rolling_df
        gc.collect()
        metrics.append(log_step_metrics("Temperature differential features", step_start))
        master_pbar.update(1)
        
        # Detect temperature anomalies - this is memory intensive
        step_start = time.time()
        master_pbar.set_description(f"Detecting temperature anomalies (Memory: {get_memory_usage():.2f} GB)")
        anomaly_df = self.detect_temp_anomalies(temp_diff_df)
        
        # Clean up to save memory
        del temp_diff_df
        gc.collect()
        metrics.append(log_step_metrics("Temperature anomaly detection", step_start))
        master_pbar.update(1)
        
        # Create material ratio features
        step_start = time.time()
        master_pbar.set_description(f"Creating material ratio features (Memory: {get_memory_usage():.2f} GB)")
        ratio_df = self.create_material_ratio_features(anomaly_df)
        
        # Clean up to save memory
        del anomaly_df
        gc.collect()
        metrics.append(log_step_metrics("Material ratio features", step_start))
        master_pbar.update(1)
        
        # Create accretion indicator features - new step
        step_start = time.time()
        master_pbar.set_description(f"Creating accretion indicator features (Memory: {get_memory_usage():.2f} GB)")
        indicator_df = self.create_accretion_indicator_features(ratio_df)
        
        # Clean up to save memory
        del ratio_df
        gc.collect()
        metrics.append(log_step_metrics("Accretion indicator features", step_start))
        master_pbar.update(1)
        
        # Create target variables
        step_start = time.time()
        if events_data is not None:
            master_pbar.set_description(f"Creating target variables (Memory: {get_memory_usage():.2f} GB)")
            final_df = self.create_target_variables(indicator_df, events_data)
            
            # Clean up
            del events_data
        else:
            final_df = indicator_df
            
        metrics.append(log_step_metrics("Target variables creation", step_start))
        master_pbar.update(1)
        
        # NEW STEP: Reduce dimensionality
        step_start = time.time()
        master_pbar.set_description(f"Reducing dimensionality (Memory: {get_memory_usage():.2f} GB)")
        
        # Get column counts before reduction
        col_count_before = final_df.shape[1]
        print(f"Dataset has {col_count_before} columns before dimension reduction")
        
        # Apply dimension reduction
        final_df = self.reduce_dimension(final_df, max_features=self.config['max_features'])
        
        # Record metrics
        metrics.append(log_step_metrics("Dimension reduction", step_start))
        master_pbar.update(1)
        
        # Clean up temp directory if we used it
        if use_temp_files and os.path.exists(temp_dir):
            import shutil
            try:
                shutil.rmtree(temp_dir)
                print(f"Cleaned up temporary directory: {temp_dir}")
           

            except Exception as e:
                print(f"Warning: Failed to clean up temp directory: {e}")
        
        # Report total time and performance metrics
        total_elapsed = time.time() - start_time
        hours, remainder = divmod(total_elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        print(f"\n{'='*80}")
        print(f"PIPELINE PERFORMANCE SUMMARY")
        print(f"{'='*80}")
        print(f"Total processing time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
        print(f"Final memory usage: {get_memory_usage():.2f} GB")
        print(f"CPU cores utilized: {N_CORES} of {mp.cpu_count()}")
        print(f"{'='*80}")
        
        # Display step-by-step timing breakdown
        print("\nPERFORMANCE BREAKDOWN BY STEP:")
        print(f"{'Step':<30} | {'Time (s)':<10} | {'Memory (GB)':<12} | {'% of Total':<10}")
        print(f"{'-'*30}-+-{'-'*10}-+-{'-'*12}-+-{'-'*10}")
        
        for m in metrics:
            step_time = m['time_sec']
            step_pct = (step_time / total_elapsed) * 100
            print(f"{m['step']:<30} | {step_time:<10.2f} | {m['memory_gb']:<12.2f} | {step_pct:<10.2f}%")
            
        # Identify bottlenecks
        if metrics:
            slowest_step = max(metrics, key=lambda x: x['time_sec'])
            highest_memory = max(metrics, key=lambda x: x['memory_gb'])
            
            print(f"\nBOTTLENECK ANALYSIS:")
            print(f"Slowest step: {slowest_step['step']} ({slowest_step['time_sec']:.2f}s, {(slowest_step['time_sec']/total_elapsed)*100:.1f}% of total)")
            print(f"Highest memory usage: {highest_memory['step']} ({highest_memory['memory_gb']:.2f}GB)")
            
        master_pbar.close()
            
        # Save preprocessed data and performance metrics
        # save_dir = True
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            final_df.to_csv(os.path.join(save_dir, 'preprocessed_data.csv'))
            
            # Save scalers and imputers
            joblib.dump(self.scalers, os.path.join(save_dir, 'scalers.joblib'))
            joblib.dump(self.imputers, os.path.join(save_dir, 'imputers.joblib'))
            
            # Save performance metrics if profiling was enabled
            if profile and metrics:
                import json
                metrics_serializable = []
                for m in metrics:
                    m_copy = m.copy()
                    m_copy['step'] = str(m_copy['step'])  # Ensure JSON serializable
                    metrics_serializable.append(m_copy)
                    
                with open(os.path.join(save_dir, 'performance_metrics.json'), 'w') as f:
                    json.dump({
                        'total_time': total_elapsed,
                        'cpu_cores': N_CORES,
                        'final_memory_gb': get_memory_usage(),
                        'step_metrics': metrics_serializable
                    }, f, indent=2)
        
        
        return final_df