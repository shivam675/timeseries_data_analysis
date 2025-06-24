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
        logging.FileHandler("kiln_system.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('kiln_system')
env_path = r"e:/Code/timeseries_ml_solution/env/Scripts/python.exe"
# env_path = 'python'

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Kiln Accretion Monitoring and Prevention System')
    parser.add_argument('--generate_data', action='store_true', help='Generate new synthetic data')
    parser.add_argument('--train_models', action='store_true', help='Train prediction and prescription models')
    parser.add_argument('--dashboard', action='store_true', help='Start the dashboard')
    parser.add_argument('--all', action='store_true', help='Run the complete pipeline')
    parser.add_argument('--model_type', type=str, default='lstm', choices=['lstm', 'transformer'],
                      help='Type of model to train')
    return parser.parse_args()

def generate_data():
    """Generate synthetic data"""
    logger.info("Generating synthetic data...")
    try:
        # Use absolute path to ensure script is found
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_models.py")
        cmd = [env_path, script_path, "--generate_data"]
        logger.info(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        logger.info("Data generation complete")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error generating data: {e}")
        return False

def train_models(model_type):
    """Train prediction and prescription models"""
    logger.info(f"Training {model_type} models...")
    try:
        # Use absolute path to ensure script is found
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_models.py")
        cmd = [env_path, script_path, "--model_type", model_type]
        logger.info(f"Running command: {' '.join(cmd)}")
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
        # Use absolute path to ensure script is found
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.py")
        cmd = [env_path, script_path]
        logger.info(f"Running command: {' '.join(cmd)}")
        dashboard_process = subprocess.Popen(cmd)
        logger.info("Dashboard started at http://localhost:8050")
        return dashboard_process
    except Exception as e:
        logger.error(f"Error starting dashboard: {e}")
        return None

def main():
    """Main entry point"""
    try:
        from tqdm import tqdm
        logger.info("TQDM library available for progress tracking")
    except ImportError:
        logger.info("TQDM library not available, using basic progress tracking")
    
    args = parse_arguments()
    logger.info("Starting Kiln Accretion Monitoring and Prevention System")
    logger.info(f"Arguments: {args}")
    
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    logger.info("Directories checked/created")
    
    # Define pipeline steps
    steps = []
    if args.all or args.generate_data:
        steps.append("data_generation")
    if args.all or args.train_models:
        steps.append("model_training")
    if args.all or args.dashboard:
        steps.append("dashboard")
    
    logger.info(f"Pipeline steps to execute: {steps}")
    
    # Run the requested components or the complete pipeline
    step_count = 0
    total_steps = len(steps)
    
    logger.info(f"====== STARTING PIPELINE: {total_steps} STEPS ======")
    
    # Data Generation
    if "data_generation" in steps:
        step_count += 1
        logger.info(f"STEP {step_count}/{total_steps}: DATA GENERATION")
        success = generate_data()
        if not success and args.all:
            logger.error("Data generation failed, stopping pipeline")
            return
        logger.info(f"Data generation {'succeeded' if success else 'failed'}")
    
    # Model Training
    if "model_training" in steps:
        step_count += 1
        logger.info(f"STEP {step_count}/{total_steps}: MODEL TRAINING ({args.model_type})")
        success = train_models(args.model_type)
        if not success and args.all:
            logger.error("Model training failed, stopping pipeline")
            return
        logger.info(f"Model training {'succeeded' if success else 'failed'}")
    
    # Dashboard
    if "dashboard" in steps:
        step_count += 1
        logger.info(f"STEP {step_count}/{total_steps}: DASHBOARD")
        dashboard_process = start_dashboard()
        
        # Keep the main process alive while dashboard is running
        if dashboard_process:
            try:
                logger.info("Dashboard running. Press Ctrl+C to stop...")
                while dashboard_process.poll() is None:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Shutting down dashboard...")
                dashboard_process.terminate()
                dashboard_process.wait()
                
    logger.info("====== PIPELINE EXECUTION COMPLETE ======")
        
    logger.info("Execution complete")

if __name__ == "__main__":
    main()