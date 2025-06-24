import os
import argparse
import numpy as np
import pandas as pd
import torch
from models import KilnAccretionPredictor
from pre_processing import KilnDataPreprocessor

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train kiln accretion prediction models (LSTM/Transformer, PyTorch)')
    parser.add_argument('--data_dir', type=str, default='data', help='Directory to load data')
    parser.add_argument('--model_dir', type=str, default='models', help='Directory to save trained models')
    parser.add_argument('--model_type', type=str, default='lstm', choices=['lstm', 'transformer'], help='Model type')
    parser.add_argument('--seq_len', type=int, default=10, help='Sequence length for time series')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--generate_data', action='store_true', help='Generate synthetic data')
    parser.add_argument('--start_date', type=str, default='2024-06-01', help='Start date for synthetic data')
    parser.add_argument('--end_date', type=str, default='2025-06-01', help='End date for synthetic data')
    return parser.parse_args()

def preprocess_data(data_dir, save_dir):
    print('Preprocessing data...')
    preprocessor = KilnDataPreprocessor()
    processed_df = preprocessor.process(
        data_dir=data_dir,
        save_dir=save_dir,
        use_temp_files=True,
        batch_size=50,
        max_memory_gb=12
    )
    print('Data preprocessing complete.')
    return processed_df, preprocessor

def train_prediction_model(df, args):
    print(f"Training {args.model_type} model on device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    target_cols = [col for col in df.columns if col.startswith('target_') or col == 'days_to_critical' or col == 'accretion_zone']
    feature_cols = [col for col in df.columns if col not in target_cols]
    X = df[feature_cols]
    y_binary = df['target_accretion_forming'] if 'target_accretion_forming' in df.columns else None
    y_days = df['days_to_critical'] if 'days_to_critical' in df.columns else None
    y_zone = df['accretion_zone'] if 'accretion_zone' in df.columns else None
    valid_idx = ~y_binary.isnull() if y_binary is not None else []
    X = X.loc[valid_idx] if len(valid_idx) > 0 else X
    y_binary = y_binary.loc[valid_idx] if y_binary is not None and len(valid_idx) > 0 else y_binary
    # Time-based split: last 20% for test
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_binary_train = y_binary.iloc[:split_idx] if y_binary is not None else None
    y_binary_test = y_binary.iloc[split_idx:] if y_binary is not None else None
    y_days_train = y_days.iloc[:split_idx] if y_days is not None else None
    y_days_test = y_days.iloc[split_idx:] if y_days is not None else None
    y_zone_train = y_zone.iloc[:split_idx] if y_zone is not None else None
    y_zone_test = y_zone.iloc[split_idx:] if y_zone is not None else None
    predictor = KilnAccretionPredictor(
        model_type=args.model_type,
        seq_len=args.seq_len
    )
    # Set hyperparameters for PyTorch models
    for name in ['binary_classifier', 'days_regressor', 'zone_classifier']:
        predictor.models[name] = None  # Will be created in fit
    predictor.fit(
        X_train,
        y_binary_train,
        y_days=y_days_train,
        y_zone=y_zone_train
    )
    eval_results = predictor.evaluate(
        X_test,
        y_binary_test,
        y_days=y_days_test,
        y_zone=y_zone_test
    )
    print('Evaluation Results:')
    print(eval_results)
    predictor.save(os.path.join(args.model_dir, 'predictor'))
    print('Model saved.')
    return predictor

def generate_synthetic_data(args):
    """Generate synthetic data for kiln monitoring"""
    try:
        from data_generator import KilnDataGenerator
        print(f"Generating synthetic kiln data from {args.start_date} to {args.end_date}")
        generator = KilnDataGenerator(
            start_date=args.start_date,
            end_date=args.end_date,
            save_dir=args.data_dir
        )
        generator.generate_all()
        print(f"Synthetic data generated successfully in {args.data_dir}")
        return True
    except ImportError:
        print("WARNING: data_generator.py module not found. Using dummy data generation.")
        # Create minimal dummy files if generator isn't available
        import pandas as pd
        import numpy as np
        
        # Create date range
        start = pd.to_datetime(args.start_date)
        end = pd.to_datetime(args.end_date)
        dates = pd.date_range(start=start, end=end, freq='15T')
          # Create temperature data
        print("Generating dummy zone temperature data...")
        temps = pd.DataFrame({
            'DATETIME': dates,  # Use uppercase DATETIME to match what pre_processing.py expects
            'zone1_temp': np.random.normal(900, 50, size=len(dates)),
            'zone2_temp': np.random.normal(950, 60, size=len(dates)),
            'zone3_temp': np.random.normal(1000, 70, size=len(dates)),
            'zone4_temp': np.random.normal(1050, 80, size=len(dates)),
        })
        temps.to_csv(os.path.join(args.data_dir, 'zone_temperature.csv'), index=False)
        
        # Create event data
        print("Generating dummy accretion events data...")
        events = pd.DataFrame({
            'EVENT_ID': [1, 2, 3, 4],
            'START_DATE': [
                start + pd.Timedelta(days=30),
                start + pd.Timedelta(days=90),
                start + pd.Timedelta(days=150),
                start + pd.Timedelta(days=210)
            ],
            'CRITICAL_DATE': [
                start + pd.Timedelta(days=45),
                start + pd.Timedelta(days=105),
                start + pd.Timedelta(days=165),
                start + pd.Timedelta(days=225)
            ],
            'ZONE': [3, 4, 5, 6],
            'CLEARED_DATE': [
                start + pd.Timedelta(days=50),
                start + pd.Timedelta(days=110),
                start + pd.Timedelta(days=170),
                start + pd.Timedelta(days=230)
            ],
            'DURATION_DAYS': [20, 20, 20, 20]
        })
        events.to_csv(os.path.join(args.data_dir, 'accretion_events.csv'), index=False)
          # Create truth data
        print("Generating dummy accretion truth data...")
        truth = pd.DataFrame({
            'DATETIME': dates,  # Use uppercase DATETIME to match what pre_processing.py expects
            'accretion_forming': np.random.choice([0, 1], size=len(dates), p=[0.9, 0.1]),
        })
        truth.to_csv(os.path.join(args.data_dir, 'accretion_truth.csv'), index=False)
        
        print(f"Dummy data generated successfully in {args.data_dir}")
        return True
    
def main():
    args = parse_arguments()
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Handle --generate_data flag explicitly
    if args.generate_data:
        print("Generating synthetic data...")
        generate_synthetic_data(args)
    else:
        # Proceed with model training
        print("Preprocessing data for model training...")
        processed_df, preprocessor = preprocess_data(args.data_dir, os.path.join(args.model_dir, 'preprocessing'))
        predictor = train_prediction_model(processed_df, args)

if __name__ == '__main__':
    main()