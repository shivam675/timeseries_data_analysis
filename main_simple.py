import os
import argparse
import logging
import subprocess
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("kiln_system_simple.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('kiln_system')

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Kiln Accretion Monitoring and Prevention System (Simplified Pipeline)')
    parser.add_argument('--generate_data', action='store_true', help='Generate new synthetic data')
    parser.add_argument('--train_models', action='store_true', help='Train prediction and prescription models')
    parser.add_argument('--dashboard', action='store_true', help='Start the dashboard')
    parser.add_argument('--all', action='store_true', help='Run the complete pipeline')
    parser.add_argument('--model_type', type=str, default='xgb', choices=['rf', 'xgb', 'lgbm', 'lstm'],
                      help='Type of model to train')
    return parser.parse_args()

def generate_data():
    """Generate synthetic data"""
    logger.info("Generating synthetic data...")
    try:
        cmd = ["python", "train_models_simple.py", "--generate_data"]
        subprocess.run(cmd, check=True)
        logger.info("Data generation complete")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error generating data: {e}")
        return False

def train_models(model_type):
    """Train prediction and prescription models"""
    logger.info(f"Training {model_type} models with simplified preprocessing...")
    try:
        cmd = ["python", "train_models_simple.py", "--model_type", model_type]
        subprocess.run(cmd, check=True)
        logger.info("Model training complete")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error training models: {e}")
        return False

def start_dashboard():
    """Start the dashboard"""
    logger.info("Starting dashboard...")
    try:
        cmd = ["python", "dashboard.py"]
        dashboard_process = subprocess.Popen(cmd)
        logger.info("Dashboard started at http://localhost:8050")
        return dashboard_process
    except Exception as e:
        logger.error(f"Error starting dashboard: {e}")
        return None

def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Record start time
    start_time = time.time()
    
    # Run the requested components or the complete pipeline
    if args.all or args.generate_data:
        success = generate_data()
        if not success and args.all:
            logger.error("Data generation failed, stopping pipeline")
            return
    
    if args.all or args.train_models:
        success = train_models(args.model_type)
        if not success and args.all:
            logger.error("Model training failed, stopping pipeline")
            return
    
    if args.all or args.dashboard:
        dashboard_process = start_dashboard()
        
        # Keep the main process alive while dashboard is running
        if dashboard_process:
            try:
                while dashboard_process.poll() is None:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Shutting down dashboard...")
                dashboard_process.terminate()
                dashboard_process.wait()
    
    # Record elapsed time
    elapsed_time = time.time() - start_time
    logger.info(f"Total execution time: {elapsed_time:.2f} seconds")
    logger.info("Execution complete")

if __name__ == "__main__":
    main()
