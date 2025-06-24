import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import argparse
import joblib
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import gc
# Import Keras conditionally to avoid error if not installed
try:
    import mlflow.keras
except ImportError:
    print("Warning: Keras module not found. LSTM models will not be available.")

import logging
import time

from data_generator import KilnSimulator
from simple_pre_processing import SimpleKilnDataPreprocessor
from models import KilnAccretionPredictor, KilnAccretionPrescriptor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("train_models_simple.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('train_models')

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train kiln accretion prediction and prescription models')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory to load/save data')
    parser.add_argument('--model_dir', type=str, default='models',
                        help='Directory to save trained models')
    parser.add_argument('--model_type', type=str, default='xgb',
                        choices=['rf', 'xgb', 'lgbm', 'lstm'],
                        help='Type of model to train')
    parser.add_argument('--generate_data', action='store_true',
                        help='Generate new synthetic data')
    parser.add_argument('--start_date', type=str, default='2022-01-01',
                        help='Start date for synthetic data')
    parser.add_argument('--end_date', type=str, default='2025-06-01',
                        help='End date for synthetic data')
    parser.add_argument('--simple', action='store_true',
                        help='Use simplified preprocessing pipeline')
    return parser.parse_args()

def generate_synthetic_data(args):
    """Generate synthetic data for kiln operation and accretion events"""
    print(f"Generating synthetic data from {args.start_date} to {args.end_date}...")
    
    # Create data directories
    os.makedirs(args.data_dir, exist_ok=True)
    
    # Initialize data generator
    generator = KilnSimulator()
    
    # Generate data
    data = generator.generate_datasets()
    
    # Copy data from synthetic_data to data directory
    print("Copying data files to data directory...")
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)
    
    # Copy all CSV files from synthetic_data to args.data_dir
    source_dir = 'synthetic_data'
    for file in os.listdir(source_dir):
        if file.endswith('.csv'):
            source_file = os.path.join(source_dir, file)
            dest_file = os.path.join(args.data_dir, file)
            # Use pandas to read and write to ensure proper handling
            df = pd.read_csv(source_file)
            df.to_csv(dest_file, index=False)
            print(f"Copied {file} to {args.data_dir}")
    
    print("Synthetic data generation complete.")
    return data

def visualize_data(data, save_dir):
    """Create visualizations of the data"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Visualize zone temperatures
    if 'zone_temperature' in data:
        df = data['zone_temperature']
        plt.figure(figsize=(16, 8))
        
        # Plot a sample of zone temperatures
        sample_dates = pd.date_range(df['DATETIME'].min(), df['DATETIME'].max(), periods=10000)
        sample_df = df[df['DATETIME'].isin(sample_dates)]
        
        for zone in range(11):
            plt.plot(sample_df['DATETIME'], sample_df[f'ZONE_{zone}'], label=f'Zone {zone}')
        
        plt.title('Zone Temperatures Over Time (Sampled)')
        plt.xlabel('Date')
        plt.ylabel('Temperature (°C)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'zone_temperatures.png'))
        plt.close()
    
    # Visualize accretion events
    if 'accretion_events' in data:
        events_df = data['accretion_events']
        
        # Create timeline of events
        plt.figure(figsize=(16, 6))
        for i, row in events_df.iterrows():
            start = row['START_DATE']  # Updated column name to match our data
            end = row['CRITICAL_DATE']  # Updated column name to match our data
            zone = row['ZONE']  # Updated column name to match our data
            plt.plot([start, end], [zone, zone], linewidth=6, alpha=0.7)
            plt.scatter([start], [zone], color='green', s=100, zorder=10, label='Symptom Start' if i == 0 else "")
            plt.scatter([end], [zone], color='red', s=100, zorder=10, label='Critical Formation' if i == 0 else "")
        
        plt.title('Accretion Events Timeline')
        plt.xlabel('Date')
        plt.ylabel('Zone')
        plt.yticks(range(11))
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'accretion_events_timeline.png'))
        plt.close()
    
    print("Data visualization complete.")

def preprocess_data_simple(data_dir, save_dir):
    """
    Preprocess raw data using simplified pipeline for model training
    
    Args:
        data_dir (str): Directory containing raw data files
        save_dir (str): Directory to save processed data and preprocessing artifacts
        
    Returns:
        tuple: (processed_df, preprocessor) containing the processed dataframe and processor instance
    """
    print("Preprocessing data with simplified pipeline...")
    start_time = time.time()
    
    # Initialize simple preprocessor
    preprocessor = SimpleKilnDataPreprocessor()
    
    # Process data with simplified pipeline
    processed_df, preprocessor = preprocessor.process(
        data_dir=data_dir,
        save_dir=save_dir,
        batch_size=100,  # Smaller batch size for better memory usage
    )
    
    duration = time.time() - start_time
    print(f"Data preprocessing complete in {duration:.2f} seconds")
    print(f"Processed dataframe shape: {processed_df.shape}")
    
    return processed_df, preprocessor

def train_prediction_model(df, model_type='xgb', model_dir='models', verbose=True):
    """Train a model to predict accretion formation"""
    logger.info(f"Training {model_type} prediction model...")
    
    # Check if we have target variables
    if 'accretion_next_24h' not in df.columns:
        logger.error("No target variables found. Cannot train prediction model.")
        return None
        
    # Set up MLflow tracking
    run_id = setup_mlflow("kiln_accretion_prediction")
    
    # Create model save directory
    os.makedirs(model_dir, exist_ok=True)
    
    # Specify target variable and features
    target = 'accretion_next_24h'  # Predict accretion in next 24 hours
    
    # Drop other target variables and non-feature columns for training
    drop_cols = ['accretion_next_48h', 'accretion_next_72h', 'accretion_zone']
    
    # Make sure we're only using numeric columns for training to avoid the "string to float" error
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    # Check for any remaining non-numeric columns - these require special handling
    non_numeric_cols = df.select_dtypes(exclude=['number']).columns
    if len(non_numeric_cols) > 0:
        print(f"Found {len(non_numeric_cols)} non-numeric columns: {list(non_numeric_cols)}")
        
        # Process non-numeric columns in a batch to minimize DataFrame fragmentation
        categorical_cols = []
        dummy_vars_list = []  # Store dummy variables for each categorical column
        
        for col in non_numeric_cols:
            # Check if the column looks like a categorical variable (fewer than 20% unique values)
            if len(df[col].dropna().unique()) / len(df) < 0.2:
                categorical_cols.append(col)
                print(f"Converting categorical column {col} to dummy variables")
                
                # Check the actual type of data in the column
                sample_val = df[col].iloc[0] if not df[col].isna().all() else None
                
                if isinstance(sample_val, (str, int, float)):
                    # Standard scalar values - use category codes
                    df[f"{col}_code"] = df[col].astype('category').cat.codes
                    # Drop the original column
                    drop_cols.append(col)
                    print(f"Converted {col} to {col}_code using category codes")
                    
                elif isinstance(sample_val, np.ndarray):
                    # For array-like values, create string representation for dummification
                    print(f"Column {col} contains numpy arrays. Converting to dummy variables.")
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
                        
                        # Store dummies for adding all at once later
                        dummy_vars_list.append(dummies)
                        
                        # Add to columns to drop
                        drop_cols.append(col)
                        print(f"Created {len(dummies.columns)} dummy variables for {col}")
                    except Exception as e:
                        print(f"Error creating dummies for array column {col}: {e}")
                        # If dummy creation fails, just drop the problematic column
                        drop_cols.append(col)
                        print(f"Dropped column {col} due to conversion error")
                        
                else:
                    # Unknown or complex types - drop them
                    drop_cols.append(col)
                    print(f"Dropping column {col} with unsupported type: {type(sample_val)}")
            else:
                # Add to drop list if it doesn't look like a useful categorical feature
                drop_cols.append(col)
                print(f"Dropping column {col} - too many unique values to be useful as a categorical feature")
                
        # Add all dummy variables at once to minimize fragmentation
        if dummy_vars_list:
            all_dummies = pd.concat(dummy_vars_list, axis=1)
            df = pd.concat([df, all_dummies], axis=1)
            print(f"Added {all_dummies.shape[1]} dummy variables total")
            
    # Create feature set by dropping specified columns
    X = df.drop(columns=[target] + [col for col in drop_cols if col in df.columns])
    
    # Convert target to numeric and handle NaN values
    y = pd.to_numeric(df[target], errors='coerce').fillna(0).astype(int)
    
    # Log original shape
    print(f"Original X shape: {X.shape}")
    
    # Check memory usage before optimization
    memory_usage_before = X.memory_usage(deep=True).sum() / 1024**2  # in MB
    print(f"Memory usage before optimization: {memory_usage_before:.2f} MB")
    
    # Remove columns with zero variance
    X_var = X.var()
    zero_var_cols = X_var[X_var == 0].index.tolist()
    if zero_var_cols:
        print(f"Removing {len(zero_var_cols)} columns with zero variance")
        X = X.drop(columns=zero_var_cols)
    
    # Force numeric dtype conversion where needed
    cols_with_issues = []
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            print(f"Column {col} is not numeric. Converting to float.")
            try:
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
            except Exception as e:
                print(f"Error converting {col}: {e}")
                cols_with_issues.append(col)
        
        # Also check for inf values that can cause warnings
        inf_mask = np.isinf(X[col])
        if inf_mask.any():
            print(f"Column {col} contains {inf_mask.sum()} infinity values. Replacing with NaN.")
            X.loc[inf_mask, col] = np.nan
            X[col] = X[col].fillna(X[col].mean() if X[col].mean() != np.nan else 0)
    
    # Drop any columns that couldn't be converted
    if cols_with_issues:
        print(f"Dropping {len(cols_with_issues)} columns that couldn't be converted to numeric")
        X = X.drop(columns=cols_with_issues, errors='ignore')
    
    # Optimize memory usage by using appropriate dtypes
    for col in X.columns:
        col_data = X[col]
        
        # For float columns with low precision needs, downcast to float32
        if pd.api.types.is_float_dtype(col_data):
            X[col] = pd.to_numeric(col_data, downcast='float')
        
        # For integer columns, downcast to smallest suitable integer type
        elif pd.api.types.is_integer_dtype(col_data):
            X[col] = pd.to_numeric(col_data, downcast='integer')
    
    # Check memory usage after optimization
    memory_usage_after = X.memory_usage(deep=True).sum() / 1024**2  # in MB
    print(f"Memory usage after optimization: {memory_usage_after:.2f} MB")
    print(f"Memory reduction: {(1 - memory_usage_after/memory_usage_before)*100:.1f}%")
    
    # Time-based train-test split
    test_size = 0.2
    split_idx = int(len(X) * (1 - test_size))
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")
    print(f"Positive samples in training: {sum(y_train == 1)} ({sum(y_train == 1)/len(y_train)*100:.2f}%)")
    print(f"Positive samples in testing: {sum(y_test == 1)} ({sum(y_test == 1)/len(y_test)*100:.2f}%)")
    
    # Initialize the predictor
    predictor = KilnAccretionPredictor(model_type=model_type)
      # Train the model with MLflow tracking
    with mlflow.start_run(run_id=run_id) as run:
        # Log parameters
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_param("positive_train_pct", sum(y_train == 1)/len(y_train)*100)
        mlflow.log_param("feature_count", X_train.shape[1])
        mlflow.log_param("memory_optimization", "True")
        mlflow.log_param("batch_processing", "True")
        
        # Log dataset statistics
        mlflow.log_param("dataset_shape", f"{df.shape[0]}x{df.shape[1]}")
        mlflow.log_param("class_balance", f"{sum(y_train == 1)}:{len(y_train)-sum(y_train == 1)}")
        
        # Log feature names and importances for reference
        mlflow.log_dict({"features": X_train.columns.tolist()}, "feature_names.json")
        
        # Train the model
        start_time = time.time()
        predictor.fit(X_train, y_train)
        train_time = time.time() - start_time
        logger.info(f"Model training completed in {train_time:.2f} seconds")
        
        # Evaluate on test set
        y_pred = predictor.predict(X_test)
        metrics = predictor.evaluate(X_test, y_test)
        
        # Log metrics with better error handling
        logger.info(f"Model evaluation metrics: {metrics}")
        
        # Check if metrics is a dict and contains binary key
        if isinstance(metrics, dict) and 'binary' in metrics:
            # Extract just the binary metrics
            binary_metrics = metrics.get('binary', {})
            
            if isinstance(binary_metrics, dict):
                # Log each binary metric separately
                for metric_name, metric_value in binary_metrics.items():
                    try:
                        mlflow.log_metric(f"binary_{metric_name}", float(metric_value))
                        logger.info(f"Logged metric: binary_{metric_name} = {metric_value}")
                    except (ValueError, TypeError) as e:
                        logger.error(f"Error logging binary_{metric_name}: {e}")
            else:
                logger.warning(f"Unexpected binary metrics format: {binary_metrics}")
                
        elif isinstance(metrics, dict):
            # Handle flat metrics dictionary
            logger.info("No 'binary' key found in metrics, attempting to log directly...")
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    try:
                        mlflow.log_metric(metric_name, float(metric_value))
                        logger.info(f"Logged metric: {metric_name} = {metric_value}")
                    except (ValueError, TypeError) as e:
                        logger.error(f"Error logging {metric_name}: {e}")
                else:
                    logger.warning(f"Skipping non-numeric metric: {metric_name} = {metric_value}")
        else:
            logger.warning(f"Unexpected metrics format: {metrics}")
        
        mlflow.log_metric("training_time", train_time)
        
        # Generate and log confusion matrix and ROC curve
        try:
            import matplotlib.pyplot as plt
            from sklearn.metrics import confusion_matrix, roc_curve, auc
            import io
            
            # Create confusion matrix plot
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            # Save confusion matrix to a buffer and log it
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            mlflow.log_figure(plt.gcf(), "confusion_matrix.png")
            plt.close()
              # Create ROC curve
            if hasattr(predictor.models.get('binary_classifier', {}), 'predict_proba'):
                # Use X_test instead of X_scaled which doesn't exist in this context
                y_prob = predictor.models['binary_classifier'].predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC Curve')
                plt.legend(loc='lower right')
                
                # Save and log ROC curve
                mlflow.log_figure(plt.gcf(), "roc_curve.png")
                plt.close()
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
        
        # Log feature importances if available
        try:
            if 'binary_classifier' in predictor.models and hasattr(predictor.models['binary_classifier'], 'feature_importances_'):
                feature_imp = predictor.models['binary_classifier'].feature_importances_
                feature_names = X_train.columns.tolist()
                
                # Sort features by importance
                indices = np.argsort(feature_imp)[::-1]
                top_features = [(feature_names[i], float(feature_imp[i])) for i in indices[:30]]  # Top 30 features
                
                # Log feature importances as parameters
                for i, (name, importance) in enumerate(top_features[:10]):  # Log top 10
                    mlflow.log_param(f"top_feature_{i+1}", f"{name} ({importance:.4f})")
                
                # Log all feature importances as a dict
                mlflow.log_dict(
                    {name: float(importance) for name, importance in zip(feature_names, feature_imp)},
                    "feature_importances.json"
                )
                
                # Create feature importance plot
                plt.figure(figsize=(10, 8))
                top_n = min(20, len(feature_names))  # Limit to 20 features for readability
                plt.barh(range(top_n), feature_imp[indices[:top_n]], align='center')
                plt.yticks(range(top_n), [feature_names[i] for i in indices[:top_n]])
                plt.xlabel('Feature Importance')
                plt.title('Top 20 Important Features')
                plt.tight_layout()
                
                # Log the plot
                mlflow.log_figure(plt.gcf(), "feature_importance.png")
                plt.close()
        except Exception as e:
            logger.error(f"Error logging feature importances: {e}")
          # Log the model with proper signature and input example
        if model_type != 'lstm':  # LSTM models need special handling
            try:
                # Clean up before we log the model
                gc.collect()
                
                # Log model with signature and input example
                log_model_with_signature(
                    run_id=run_id, 
                    model=predictor.models['binary_classifier'], 
                    model_name='accretion_prediction', 
                    sample_data=(X_test, y_test),
                    signature_name='classifier'
                )
            except Exception as e:
                logger.error(f"Error logging model to MLflow: {e}")
            
            # Log the binary classifier model if it exists
            if 'binary_classifier' in predictor.models:
                model = predictor.models['binary_classifier']
                
                # Infer the model signature with careful handling of dtypes
                try:
                    # First make sure we have consistent dtypes that MLflow can handle
                    sample_inputs_clean = sample_inputs.copy()
                    for col in sample_inputs_clean.columns:
                        if sample_inputs_clean[col].dtype.name == 'int8':
                            sample_inputs_clean[col] = sample_inputs_clean[col].astype('int32')
                        elif sample_inputs_clean[col].dtype.name == 'int16':
                            sample_inputs_clean[col] = sample_inputs_clean[col].astype('int32')
                        elif sample_inputs_clean[col].dtype.name == 'float16':
                            sample_inputs_clean[col] = sample_inputs_clean[col].astype('float32')
                            
                    # Get predictions with the model for signature
                    sample_preds = model.predict(sample_inputs_clean)
                    
                    # Create signature
                    signature = infer_signature(sample_inputs_clean, sample_preds)
                    
                    # Log model with cleaned sample inputs
                    mlflow.sklearn.log_model(
                        model, 
                        "binary_classifier_model",
                        signature=signature,
                        input_example=sample_inputs_clean
                    )
                    print("Logged binary classifier model to MLflow with signature and input example")
                except Exception as e:
                    print(f"Error creating model signature: {e}")
                    # Fall back to logging without signature 
                    mlflow.sklearn.log_model(model, "binary_classifier_model")
                    print("Logged binary classifier model to MLflow without signature due to error")
                
            # Log the days regressor model if it exists
            if 'days_regressor' in predictor.models:
                model = predictor.models['days_regressor']
                
                # For the regressor, we need to adjust the signature if we used transformed target values
                try:
                    sample_inputs_clean = sample_inputs.copy()
                    for col in sample_inputs_clean.columns:
                        if sample_inputs_clean[col].dtype.name in ['int8', 'int16']:
                            sample_inputs_clean[col] = sample_inputs_clean[col].astype('int32')
                        elif sample_inputs_clean[col].dtype.name == 'float16':
                            sample_inputs_clean[col] = sample_inputs_clean[col].astype('float32')
                            
                    # Get predictions with the model for signature
                    sample_preds = model.predict(sample_inputs_clean)
                    
                    # Create signature
                    signature = infer_signature(sample_inputs_clean, sample_preds)
                    
                    # Log model with cleaned sample inputs
                    mlflow.sklearn.log_model(
                        model, 
                        "days_regressor_model",
                        signature=signature,
                        input_example=sample_inputs_clean
                    )
                    print("Logged days regressor model to MLflow with signature and input example")
                except Exception as e:
                    print(f"Error creating days regressor signature: {e}")
                    mlflow.sklearn.log_model(model, "days_regressor_model")
                    print("Logged days regressor model without signature due to error")
                
            # Log the zone classifier model if it exists
            if 'zone_classifier' in predictor.models:
                model = predictor.models['zone_classifier']
                
                try:
                    sample_inputs_clean = sample_inputs.copy()
                    for col in sample_inputs_clean.columns:
                        if sample_inputs_clean[col].dtype.name in ['int8', 'int16']:
                            sample_inputs_clean[col] = sample_inputs_clean[col].astype('int32')
                        elif sample_inputs_clean[col].dtype.name == 'float16':
                            sample_inputs_clean[col] = sample_inputs_clean[col].astype('float32')
                    
                    if 'accretion_zone' in df.columns:
                        zone_labels = df.loc[y_train.index, 'accretion_zone']
                        signature = infer_signature(sample_inputs_clean, zone_labels)
                    else:
                        # Fall back to generic signature
                        sample_zones = model.predict(sample_inputs_clean)
                        signature = infer_signature(sample_inputs_clean, sample_zones)
                    
                    mlflow.sklearn.log_model(
                        model, 
                        "zone_classifier_model",
                        signature=signature,
                        input_example=sample_inputs_clean
                    )
                    print("Logged zone classifier model to MLflow with signature and input example")
                except Exception as e:
                    print(f"Error creating zone classifier signature: {e}")
                    mlflow.sklearn.log_model(model, "zone_classifier_model")
                    print("Logged zone classifier model without signature due to error")
        else:
            # For LSTM models (if implemented)
            for model_name, model in predictor.models.items():
                if hasattr(model, 'save'):  # Check if it's a Keras model
                    # Create signature for Keras model
                    try:
                        # Try to infer signature without importing tensorflow explicitly
                        sample_inputs_clean = sample_inputs.copy()
                        for col in sample_inputs_clean.columns:
                            if sample_inputs_clean[col].dtype.name in ['int8', 'int16']:
                                sample_inputs_clean[col] = sample_inputs_clean[col].astype('int32')
                            elif sample_inputs_clean[col].dtype.name == 'float16':
                                sample_inputs_clean[col] = sample_inputs_clean[col].astype('float32')
                        
                        signature = infer_signature(sample_inputs_clean.values, model.predict(sample_inputs_clean.values))
                        mlflow.keras.log_model(
                            model, 
                            f"{model_name}_model",
                            signature=signature,
                            input_example=sample_inputs_clean.values
                        )
                    except Exception as e:
                        print(f"Could not create signature for {model_name} model: {e}")
                        # Fall back without signature if there's an issue
                        mlflow.keras.log_model(model, f"{model_name}_model")
                    print(f"Logged {model_name} LSTM model to MLflow")
        
        # Save model locally
        model_path = os.path.join(model_dir, f'accretion_predictor_{model_type}.pkl')
        joblib.dump(predictor, model_path)
        print(f"Model saved to {model_path}")
        
        # Create feature importance plot if available - using the binary_classifier model
        if 'binary_classifier' in predictor.models and hasattr(predictor.models['binary_classifier'], 'feature_importances_'):
            try:
                # Get feature importances from the binary classifier
                model = predictor.models['binary_classifier']
                importances = model.feature_importances_
                feature_names = X_train.columns
                
                # Create importance dataframe
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                })
                importance_df = importance_df.sort_values('importance', ascending=False).head(30)
                
                # Plot
                plt.figure(figsize=(12, 8))
                sns.barplot(x='importance', y='feature', data=importance_df)
                plt.title(f'Top 30 Feature Importances - {model_type.upper()}')
                plt.tight_layout()
                
                # Save locally and log to MLflow
                importance_plot_path = os.path.join(model_dir, f'feature_importance_{model_type}.png')
                plt.savefig(importance_plot_path)
                mlflow.log_artifact(importance_plot_path)
                plt.close()
            except Exception as e:
                print(f"Error creating feature importance plot: {e}")
    
    print(f"Prediction model training complete. Test metrics: {metrics}")
    return predictor

def train_prescription_model(df, predictor, model_dir='models'):
    """Train a model to prescribe optimal control actions"""
    print("Training prescription model...")
    
    # Check if we have required columns
    required_cols = ['accretion_next_24h', 'mis_PRODUCTION ACTUAL']
    if not all(col in df.columns for col in required_cols):
        print("Required columns not found. Cannot train prescription model.")
        return None
    
    # Use MLflow to track experiments
    mlflow.set_experiment("kiln_accretion_prescription")
    
    # Create model save directory
    os.makedirs(model_dir, exist_ok=True)
    
    # Identify control variables
    control_vars = [col for col in df.columns if any(
        keyword in col for keyword in ['DAMPER', 'AIR_FLOW', 'FAN', 'VELOCITY', 'SPEED']
    )]
    
    # Ensure we have enough control variables
    if len(control_vars) < 2:
        print("Not enough control variables found. Cannot train prescription model.")
        return None
    
    # Filter for periods leading up to accretion events
    events_window = df['accretion_next_24h'] == 1
    if sum(events_window) >= 100:  # Need sufficient samples for training
        # Use production as reward function goal (maximizing while avoiding accretion)
        reward_col = 'mis_PRODUCTION ACTUAL'
        
        # Prepare training data for prescriptor
        X_prescription = df[events_window].copy()
        
        # Initialize prescriptor
        prescriptor = KilnAccretionPrescriptor()
        
        # Define control variables and other parameters
        action_space = {}
        valid_control_vars = []
        
        # Check that each control variable has variation and is usable
        for var in control_vars:
            # Check for zero variance - can't use these for adjustment
            if X_prescription[var].std() == 0:
                print(f"Warning: Control variable {var} has no variation. Excluding from prescription model.")
                continue
                
            # Check if all values are NaN - can't use these either
            if X_prescription[var].isna().all():
                print(f"Warning: Control variable {var} has all NaN values. Excluding from prescription model.")
                continue
                
            # Create a default action space with reasonable limits
            action_space[var] = {
                'min': -0.1,  # Allow 10% decrease
                'max': 0.1,   # Allow 10% increase
                'step': 0.01  # Step size of 1%
            }
            
            # Add to valid control variables
            valid_control_vars.append(var)
        
        # Check if we have any valid control variables
        if not valid_control_vars:
            print("No valid control variables found. Cannot train prescription model.")
            return None
            
        print(f"Using {len(valid_control_vars)} valid control variables for prescription model")
        
        # Update control variables list to only include valid ones
        control_vars = valid_control_vars
        
        # Define action space with only valid variables
        prescriptor.define_action_space({var: action_space[var] for var in valid_control_vars})
        
        # Train the model with MLflow tracking
        with mlflow.start_run() as run:
            # Log parameters
            mlflow.log_param("num_control_vars", len(control_vars))
            mlflow.log_param("reward_variable", reward_col)
            mlflow.log_param("training_samples", len(X_prescription))
            
            # Train the prescriptor
            start_time = time.time()
            
            # Create target adjustments for each control variable
            # For now, we'll use a simple strategy: adjust each parameter to minimize risk
            # This is a placeholder and would need domain-specific logic in a real system
            Y_adjustments = pd.DataFrame(index=X_prescription.index)
            
            for var in control_vars:
                # Calculate correlation with accretion events safely - handle edge cases
                try:
                    # Safe calculation of correlation to avoid division by zero warnings
                    var_std = X_prescription[var].std()
                    target_std = X_prescription['accretion_next_24h'].std()
                    
                    # Can only calculate meaningful correlation if both have variation
                    if var_std > 0 and target_std > 0:
                        corr = X_prescription[var].corr(X_prescription['accretion_next_24h'])
                        
                        # Handle NaN correlations (can happen with constant values or all NaNs)
                        if pd.isna(corr):
                            print(f"Warning: Correlation for {var} is NaN. Setting to 0.")
                            corr = 0
                    else:
                        print(f"Warning: Variable {var} or target has no variation. Setting correlation to 0.")
                        corr = 0
                        
                    # If positively correlated with accretion, decrease. If negatively correlated, increase
                    adjustment = -0.05 * np.sign(corr)
                    
                    # Apply adjustments safely:
                    # 1. For non-zero values, use percentage adjustment
                    # 2. For zero values, use small fixed adjustment
                    # 3. For NaN values, use small fixed adjustment
                    
                    # Create a mask for non-zero, non-NaN values
                    non_zero_mask = (X_prescription[var] != 0) & (~X_prescription[var].isna())
                    nan_mask = X_prescription[var].isna()
                    
                    # Apply adjustments based on masks
                    if non_zero_mask.any():
                        Y_adjustments.loc[non_zero_mask, var] = adjustment * X_prescription.loc[non_zero_mask, var]
                    
                    # For zero values, use the adjustment directly (small fixed value)
                    zero_mask = (X_prescription[var] == 0) & (~X_prescription[var].isna())
                    if zero_mask.any():
                        Y_adjustments.loc[zero_mask, var] = adjustment
                        
                    # For NaN values, use 0 (no adjustment)
                    if nan_mask.any():
                        Y_adjustments.loc[nan_mask, var] = 0
                        
                except Exception as e:
                    print(f"Error calculating adjustment for {var}: {e}")
                    # In case of error, set all to small fixed value
                    Y_adjustments[var] = 0.01
            
            # Train the prescription model with the calculated adjustments
            try:
                prescriptor.fit(X_prescription, Y_adjustments)
                train_time = time.time() - start_time
                
                # Log metrics
                mlflow.log_metric("training_time", train_time)
                
                # Log feature importances if available
                for param, importances in prescriptor.feature_importances.items():
                    # Convert importance dict to series and log as json
                    importance_series = pd.Series(importances).sort_values(ascending=False)
                    top_features = importance_series.head(10).to_dict()
                    mlflow.log_dict(top_features, f"feature_importance_{param}.json")
                
                # Save model locally and log to MLflow with signature and input example
                model_path = os.path.join(model_dir, 'accretion_prescriptor.pkl')
                joblib.dump(prescriptor, model_path)
                
                # Get a small sample of input data for example
                sample_inputs = X_prescription.iloc[:5] if len(X_prescription) > 5 else X_prescription
                
                # Clean sample inputs to avoid MLflow signature issues
                sample_inputs_clean = sample_inputs.copy()
                for col in sample_inputs_clean.columns:
                    if sample_inputs_clean[col].dtype.name in ['int8', 'int16']:
                        sample_inputs_clean[col] = sample_inputs_clean[col].astype('int32')
                    elif sample_inputs_clean[col].dtype.name == 'float16':
                        sample_inputs_clean[col] = sample_inputs_clean[col].astype('float32')
                
                # Get sample outputs for signature
                try:
                    sample_outputs = {}
                    for param, model in prescriptor.models.items():
                        if hasattr(model, 'predict'):
                            sample_outputs[param] = model.predict(sample_inputs_clean)
                    
                    # Create a combined output dataframe for signature
                    sample_output_df = pd.DataFrame(sample_outputs)
                    
                    # Create model signature
                    from mlflow.models.signature import infer_signature
                    signature = infer_signature(sample_inputs_clean, sample_output_df)
                    
                    # Log model with signature and example
                    mlflow.sklearn.log_model(
                        prescriptor, 
                        "prescription_model",
                        signature=signature,
                        input_example=sample_inputs_clean
                    )
                except Exception as e:
                    print(f"Could not create signature for prescription model: {e}")
                    # Fall back to simple logging
                    mlflow.sklearn.log_model(prescriptor, "prescription_model")
                    
                print(f"Prescription model saved to {model_path}")
                print("Prescription model training complete.")
                return prescriptor
                
            except Exception as e:
                print(f"Error training prescription model: {e}")
                print("Prescription model training failed.")
                return None
    else:
        print(f"Not enough training data for prescription model. Found only {sum(events_window)} samples, need at least 100.")
        return None

def diagnose_dataframe(df, name="DataFrame"):
    """Diagnose issues in a dataframe and provide detailed information"""
    print(f"\n--- Diagnosing {name} ---")
    print(f"Shape: {df.shape}")
    
    # Check for missing values
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    if len(missing_cols) > 0:
        print(f"Columns with missing values ({len(missing_cols)}):")
        for col, count in missing_cols.items():
            print(f"  {col}: {count} missing values ({count/len(df)*100:.1f}%)")
    else:
        print("No missing values found.")
    
    # Check for non-numeric columns
    non_numeric = df.select_dtypes(exclude=['number']).columns
    if len(non_numeric) > 0:
        print(f"Non-numeric columns ({len(non_numeric)}):")
        for col in non_numeric:
            unique_vals = df[col].nunique()
            print(f"  {col}: {unique_vals} unique values")
            # Show sample values
            if unique_vals < 10:
                print(f"    Values: {df[col].dropna().unique()}")
            else:
                print(f"    Sample values: {df[col].dropna().sample(5).values}")
    else:
        print("All columns are numeric.")
    
    # Check for infinite values
    inf_cols = []
    for col in df.select_dtypes(include=['number']).columns:
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            inf_cols.append((col, inf_count))
    
    if inf_cols:
        print(f"Columns with infinite values ({len(inf_cols)}):")
        for col, count in inf_cols:
            print(f"  {col}: {count} infinite values ({count/len(df)*100:.1f}%)")
    else:
        print("No infinite values found.")
    
    # Check for constant columns
    constant_cols = []
    for col in df.columns:
        if df[col].nunique() <= 1:
            constant_cols.append(col)
    
    if constant_cols:
        print(f"Constant columns ({len(constant_cols)}):")
        for col in constant_cols:
            print(f"  {col}: value = {df[col].iloc[0]}")
    else:
        print("No constant columns found.")
    
    # Check memory usage
    memory_usage = df.memory_usage(deep=True).sum() / 1024**2  # in MB
    print(f"Memory usage: {memory_usage:.2f} MB")
    
    print(f"--- End of {name} diagnosis ---\n")
    
    return {
        'missing_columns': missing_cols,
        'non_numeric_columns': non_numeric,
        'infinite_values': inf_cols,
        'constant_columns': constant_cols,
        'memory_usage_mb': memory_usage
    }

def setup_mlflow(experiment_name="kiln_accretion_prediction"):
    """
    Set up MLflow tracking with proper experiment naming and versioning
    
    Args:
        experiment_name (str): Name of the MLflow experiment
        
    Returns:
        str: The run ID of the created MLflow run
    """
    # Set experiment
    mlflow.set_experiment(experiment_name)
    
    # Create and configure a run
    run = mlflow.start_run()
    run_id = run.info.run_id
    
    # Set tags for better organization
    mlflow.set_tag("version", "v2.0")
    mlflow.set_tag("model_type", "accretion_prediction")
    mlflow.set_tag("data_version", datetime.now().strftime("%Y%m%d"))
    
    logger.info(f"Started MLflow run: {run_id}")
    
    return run_id

def log_model_with_signature(run_id, model, model_name, sample_data, signature_name="classifier"):
    """
    Log a model with proper input/output signature to MLflow
    
    Args:
        run_id (str): The MLflow run ID
        model: The model to log
        model_name (str): Name of the model artifact
        sample_data (tuple or pd.DataFrame): Sample data for signature inference
                                           If tuple: (X_sample, y_sample)
                                           If DataFrame: just X_sample
        signature_name (str): Name for the signature in the artifact path
    """
    try:
        # Check what type of sample data we're dealing with
        if isinstance(sample_data, tuple) and len(sample_data) == 2:
            # Assume X, y tuple
            X_sample, y_sample = sample_data
            
            # Get a small subset for the input example
            input_example = X_sample.iloc[:5] if len(X_sample) >= 5 else X_sample
            
            # Infer signature from input and output data
            signature = infer_signature(X_sample, y_sample)
        else:
            # Assume just X
            X_sample = sample_data
            input_example = X_sample.iloc[:5] if len(X_sample) >= 5 else X_sample
            signature = infer_signature(X_sample)
            
        # Log the model with signature and input example
        with mlflow.start_run(run_id=run_id):
            mlflow.sklearn.log_model(
                model,
                artifact_path=f"{model_name}_{signature_name}",
                signature=signature,
                input_example=input_example
            )
        
        logger.info(f"Successfully logged {model_name} with signature to MLflow")
    except Exception as e:
        logger.error(f"Error logging model with signature: {e}")

def main():
    """Main function to train models"""
    # Parse arguments
    args = parse_arguments()
    
    # Create directories
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    vis_dir = os.path.join(args.data_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Generate or load data
    if args.generate_data:
        data = generate_synthetic_data(args)
        visualize_data(data, vis_dir)
    else:
        # Check if data files exist
        data_files = os.listdir(args.data_dir)
        if not any(f.endswith('.csv') for f in data_files):
            print("No data found. Generating synthetic data...")
            data = generate_synthetic_data(args)
            visualize_data(data, vis_dir)
        else:
            print("Using existing data files.")
            data = {}  # We'll load from files
    
    # Preprocess data using simple pipeline
    processed_df, preprocessor = preprocess_data_simple(
        args.data_dir, 
        os.path.join(args.model_dir, 'preprocessing_simple')
    )
    
    # Diagnose the processed dataframe to identify potential issues
    diagnosis_results = diagnose_dataframe(processed_df, "Processed DataFrame")
    
    # Save diagnosis results to a file for reference
    diagnosis_df = pd.DataFrame({
        'Category': [
            'Missing Columns Count', 
            'Non-numeric Columns Count',
            'Infinite Values Count',
            'Constant Columns Count',
            'Memory Usage (MB)'
        ],
        'Count': [
            len(diagnosis_results['missing_columns']),
            len(diagnosis_results['non_numeric_columns']),
            len(diagnosis_results['infinite_values']),
            len(diagnosis_results['constant_columns']),
            diagnosis_results['memory_usage_mb']
        ]
    })
    diagnosis_path = os.path.join(args.model_dir, 'dataframe_diagnosis.csv')
    diagnosis_df.to_csv(diagnosis_path, index=False)
    print(f"DataFrame diagnosis saved to {diagnosis_path}")
    
    # Log the non-numeric columns in detail if any exist
    if diagnosis_results['non_numeric_columns'].any():
        non_numeric_df = pd.DataFrame({
            'Column': diagnosis_results['non_numeric_columns'].index,
            'Missing Count': [processed_df[col].isna().sum() for col in diagnosis_results['non_numeric_columns'].index],
            'Unique Count': [processed_df[col].nunique() for col in diagnosis_results['non_numeric_columns'].index]
        })
        non_numeric_path = os.path.join(args.model_dir, 'non_numeric_columns.csv')
        non_numeric_df.to_csv(non_numeric_path, index=False)
        print(f"Non-numeric column details saved to {non_numeric_path}")
    
    # Train prediction model
    predictor = train_prediction_model(processed_df, args.model_type, args.model_dir, verbose=True)
    
    # Train prescription model
    if predictor is not None:
        prescriptor = train_prescription_model(processed_df, predictor, args.model_dir)
        
        if prescriptor is not None:
            print("ML pipeline complete. Both prediction and prescription models trained successfully.")
        else:
            print("ML pipeline partially complete. Prediction model trained but prescription model failed.")
    else:
        print("ML pipeline failed. Could not train prediction model.")
