import os
import sys
import dash
from dash import dcc, html
import plotly.graph_objs as go
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import time
import json
from plotly.subplots import make_subplots
import plotly.express as px
import uuid
import threading
import queue
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path so we can import our modules
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import KilnAccretionPredictor
from pre_processing import KilnDataPreprocessor

# Define a simple prescriptor class to use as a fallback
class KilnAccretionPrescriptor:
    """
    Simple prescriptor class to provide basic recommendations
    """
    def __init__(self):
        self.initialized = True
        
    def load(self, path):
        # Placeholder for loading model parameters
        pass
    
    def prescribe(self, data, predictions):
        # Basic prescriptions based on predictions
        recommendations = []
        is_forming = predictions.get('is_forming', False)
        probability = predictions.get('probability', 0)
        days_to_critical = predictions.get('days_to_critical', 365)
        
        if is_forming:
            if days_to_critical < 7:
                recommendations.append({"action": "Immediate kiln shutdown recommended", "priority": "high"})
                recommendations.append({"action": "Prepare for accretion cleaning procedure", "priority": "high"})
            elif days_to_critical < 14:
                recommendations.append({"action": "Reduce feed rate by 15%", "priority": "medium"})
                recommendations.append({"action": "Adjust coal/ore ratio", "priority": "medium"})
            else:
                recommendations.append({"action": "Monitor temperature drop in affected zones", "priority": "low"})
                recommendations.append({"action": "Check material quality", "priority": "low"})
        elif probability > 0.3:
            recommendations.append({"action": "Monitor temperature patterns closely", "priority": "low"})
        else:
            recommendations.append({"action": "Normal operation - no action required", "priority": "info"})
            
        return recommendations

# --- Modern UI Theme and Styles ---
# Switch to a more modern theme (e.g., CYBORG)
DASH_THEME = dbc.themes.CYBORG

# Modern card style
CARD_STYLE = {
    'borderRadius': '18px',
    'boxShadow': '0 4px 24px rgba(0,0,0,0.12)',
    'marginBottom': '18px',
    'background': 'rgba(30, 32, 34, 0.98)',
    'border': '1px solid #23272b',
}

CARD_HEADER_STYLE = {
    'backgroundColor': '#222b45',
    'color': 'white',
    'fontWeight': 'bold',
    'fontSize': '1.1rem',
    'borderTopLeftRadius': '18px',
    'borderTopRightRadius': '18px',
    'display': 'flex',
    'alignItems': 'center',
    'gap': '10px',
}

# Add icons for card headers
ICON_MAP = {
    'Accretion Status': 'fa-solid fa-fire',
    'Current Kiln Parameters': 'fa-solid fa-gauge-high',
    'Recommended Actions': 'fa-solid fa-lightbulb',
    '3D Kiln Visualization': 'fa-solid fa-cube',
    'Zone Temperatures': 'fa-solid fa-temperature-three-quarters',
    'Production Quality': 'fa-solid fa-industry',
    'Material Consumption': 'fa-solid fa-flask',
    'Historical Accretion Events': 'fa-solid fa-clock-rotate-left',
    'Material Quality Impact Analysis': 'fa-solid fa-chart-scatter',
    'Model Feature Importance': 'fa-solid fa-brain',
}

def card_header(title):
    icon = ICON_MAP.get(title, '')
    return html.Span([
        html.I(className=icon, style={'marginRight': '8px'}) if icon else None,
        title
    ], style=CARD_HEADER_STYLE)

# Global variables
REFRESH_INTERVAL = 60  # seconds
DATA_DIR = 'data'
MODEL_DIR = 'models'
HISTORY_HOURS = 72
PREDICTION_HOURS = 72
kiln_length = 80  # meters
kiln_diameter = 4  # meters

# Create a queue for real-time data updates
data_queue = queue.Queue()

# Helper function to create empty charts with custom messages
def empty_chart(title="No Data Available"):
    """Create an empty chart with a message"""
    fig = go.Figure()
    fig.add_annotation(
        text=title,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=16)
    )
    fig.update_layout(
        template='plotly_dark',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

# Load models
def load_models():
    """Load prediction and prescription models"""
    models = {}
    
    predictor_path = os.path.join(MODEL_DIR, 'predictor')
    if os.path.exists(predictor_path):
        try:
            # Create instance first, then load
            predictor = KilnAccretionPredictor()
            predictor.load(predictor_path)
            models['predictor'] = predictor
            print("Loaded prediction model")
        except Exception as e:
            print(f"Error loading predictor: {e}")
    
    prescriptor_path = os.path.join(MODEL_DIR, 'prescriptor')
    # Always create a prescriptor instance (either it will load existing model or use default logic)
    try:
        # Create instance first, then try to load if path exists
        prescriptor = KilnAccretionPrescriptor()
        if os.path.exists(prescriptor_path):
            prescriptor.load(prescriptor_path)
            print("Loaded prescription model")
        else:
            print("Using default prescriptor (no model loaded)")
        models['prescriptor'] = prescriptor
    except Exception as e:
        print(f"Error setting up prescriptor: {e}")
            
    return models

# Load preprocessor
def load_preprocessor():
    """Load data preprocessor"""
    preproc_path = os.path.join(MODEL_DIR, 'preprocessing')
    if os.path.exists(preproc_path):
        try:
            scalers = joblib.load(os.path.join(preproc_path, 'scalers.joblib'))
            imputers = joblib.load(os.path.join(preproc_path, 'imputers.joblib'))
            preprocessor = KilnDataPreprocessor()
            preprocessor.scalers = scalers
            preprocessor.imputers = imputers
            print("Loaded preprocessor")
            return preprocessor
        except Exception as e:
            print(f"Error loading preprocessor: {e}")
    
    # If we can't load, create a new one
    return KilnDataPreprocessor()

# Load the most recent data
def load_data():
    """Load recent data from CSV files"""
    data = {}
      # Get current timestamp
    now = datetime.now()
    
    # Calculate start time for filtering - but make sure we include some data!
    # Use a much wider time window (1 year) to ensure we get data
    start_time = now - timedelta(days=365)
    
    print(f"Loading data with start_time: {start_time}")
    print(f"DATA_DIR: {DATA_DIR}, absolute path: {os.path.abspath(DATA_DIR)}")
    
    # Load MIS report (daily)
    mis_path = os.path.join(DATA_DIR, 'mis_report.csv')
    print(f"Checking MIS path: {mis_path}, exists: {os.path.exists(mis_path)}")
    if os.path.exists(mis_path):
        mis_df = pd.read_csv(mis_path, parse_dates=['DATE'])
        print(f"Loaded MIS data, shape before filtering: {mis_df.shape}")
        mis_df = mis_df[mis_df['DATE'] >= start_time]
        print(f"MIS data after filtering: {mis_df.shape}")
        data['mis'] = mis_df
    
    # Load Air Calibration (daily)
    air_path = os.path.join(DATA_DIR, 'air_calibration.csv')
    if os.path.exists(air_path):
        air_df = pd.read_csv(air_path, parse_dates=['DATE'])
        air_df = air_df[air_df['DATE'] >= start_time]
        data['air'] = air_df
    
    # Load Shell Temperature (daily)
    shell_path = os.path.join(DATA_DIR, 'shell_temperature.csv')
    if os.path.exists(shell_path):
        shell_df = pd.read_csv(shell_path, parse_dates=['DATE'])
        shell_df = shell_df[shell_df['DATE'] >= start_time]
        data['shell'] = shell_df
    
    # Load QRT Temperature (2 hourly)
    qrt_path = os.path.join(DATA_DIR, 'qrt_temperature.csv')
    if os.path.exists(qrt_path):
        qrt_df = pd.read_csv(qrt_path, parse_dates=['DATETIME'])
        qrt_df = qrt_df[qrt_df['DATETIME'] >= start_time]
        data['qrt'] = qrt_df
    
    # Load Zone Temperature (1-2 min)
    zone_path = os.path.join(DATA_DIR, 'zone_temperature.csv')
    if os.path.exists(zone_path):
        zone_df = pd.read_csv(zone_path, parse_dates=['DATETIME'])
        zone_df = zone_df[zone_df['DATETIME'] >= start_time]
        data['zone'] = zone_df
      # Load processed data if available
    processed_path = os.path.join(MODEL_DIR, 'preprocessing', 'preprocessed_data.csv')
    if os.path.exists(processed_path):
        processed_df = pd.read_csv(processed_path, parse_dates=[0])
        # Set the datetime column as the index if it's not already
        if not isinstance(processed_df.index, pd.DatetimeIndex):
            processed_df.set_index(processed_df.columns[0], inplace=True)
        # Now filter by the datetime index
        processed_df = processed_df[processed_df.index >= start_time]
        data['processed'] = processed_df
    print(f"Loaded data from {len(data)} datasets")
    return data

# Process data for prediction
def process_latest_data(data, preprocessor):
    """Process the latest data for model predictions"""
    # If we already have processed data, use it
    if 'processed' in data and not data['processed'].empty:
        return data['processed'].iloc[-1:].copy()
      
    # Otherwise, process the latest data points from each dataset
    latest_data = {}
    
    # Check if we have any data at all
    has_data = bool(data) and any(isinstance(df, pd.DataFrame) and not df.empty for key, df in data.items())
    
    if not has_data:
        print("No input data available - creating dummy dataset for testing")
        # Create dummy data with datetime index
        current_time = pd.Timestamp.now()
        # Create a single row dataframe with current time index
        dummy_df = pd.DataFrame(index=[current_time])
        # Add some dummy columns that would typically be found in zone_temperature.csv
        for i in range(11):
            dummy_df[f'ZONE_{i}'] = 900.0 + i * 10  # Dummy temperatures
        latest_data['zone'] = dummy_df
    
    if 'zone' in data and not data['zone'].empty:
        latest_df = data['zone'].iloc[-1:].copy()
        if 'DATETIME' in latest_df.columns:
            latest_df['DATETIME'] = pd.to_datetime(latest_df['DATETIME'])
            latest_df.set_index('DATETIME', inplace=True)
        latest_data['zone'] = latest_df
        
    if 'qrt' in data and not data['qrt'].empty:
        latest_df = data['qrt'].iloc[-1:].copy()
        if 'DATETIME' in latest_df.columns:
            latest_df['DATETIME'] = pd.to_datetime(latest_df['DATETIME'])
            latest_df.set_index('DATETIME', inplace=True)
        latest_data['qrt'] = latest_df
        
    if 'mis' in data and not data['mis'].empty:
        latest_df = data['mis'].iloc[-1:].copy()
        if 'DATE' in latest_df.columns:
            # Ensure DATE is datetime type with time component
            latest_df['DATE'] = pd.to_datetime(latest_df['DATE'])
            latest_df.set_index('DATE', inplace=True)
        latest_data['mis'] = latest_df
        
    if 'air' in data and not data['air'].empty:
        latest_df = data['air'].iloc[-1:].copy()
        if 'DATE' in latest_df.columns:
            # Ensure DATE is datetime type with time component
            latest_df['DATE'] = pd.to_datetime(latest_df['DATE'])
            latest_df.set_index('DATE', inplace=True)
        latest_data['air'] = latest_df
        
    if 'shell' in data and not data['shell'].empty:
        latest_df = data['shell'].iloc[-1:].copy()
        if 'DATE' in latest_df.columns:
            # Ensure DATE is datetime type with time component
            latest_df['DATE'] = pd.to_datetime(latest_df['DATE'])
            latest_df.set_index('DATE', inplace=True)
        latest_data['shell'] = latest_df
      # Process this small dataset
    try:
        # Debug info
        print("Latest data keys:", latest_data.keys())
        empty_dfs = []
        for k, df in latest_data.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                print(f"Dataset {k} shape: {df.shape}")
                print(f"Dataset {k} index type: {df.index.dtype}, kind: {df.index.dtype.kind}")
                print(f"Dataset {k} index sample: {df.index[0]}")
                # Force conversion to datetime
                try:
                    if df.index.dtype.kind != 'M':  # Not already datetime
                        print(f"Converting {k} index to datetime...")
                        df.index = pd.to_datetime(df.index)
                        latest_data[k] = df
                        print(f"After conversion - {k} index type: {df.index.dtype}, kind: {df.index.dtype.kind}")
                except Exception as e:
                    print(f"Error converting {k} index: {e}")
            else:
                empty_dfs.append(k)
                print(f"Dataset {k} is empty or not a DataFrame")
        
        if empty_dfs:
            print(f"Empty dataframes: {empty_dfs}")
            
        if not latest_data or all(not isinstance(df, pd.DataFrame) or df.empty for df in latest_data.values()):
            raise ValueError("No valid dataframes in latest_data")
            
        # Check if any dataframes have datetime indexes
        time_indexed = [k for k, df in latest_data.items() 
                      if isinstance(df, pd.DataFrame) and hasattr(df.index, 'dtype') 
                      and hasattr(df.index.dtype, 'kind') and df.index.dtype.kind == 'M']
        
        if not time_indexed:
            print("No time-indexed dataframes found after conversion attempts.")
            raise ValueError("No time-indexed dataframes found after conversion attempts.")
        
        try:
            aligned_df = preprocessor.align_time_series(latest_data, target_freq='1H')
            imputed_df = preprocessor.impute_missing_values(aligned_df, method='knn')
            return imputed_df
        except ValueError as e:
            print(f"Error in align_time_series: {e}")
            # If alignment fails, create a simple DataFrame directly from the dummy data
            if 'zone' in latest_data:
                df = latest_data['zone'].copy()
                # Make a copy to avoid mutation warnings
                processed_df = df.copy()
                # This is a temporary workaround
                print("Using direct dummy data as fallback")
                return processed_df
            raise
    except Exception as e:
        print(f"Error processing latest data: {e}")
        # Print more details about each dataset to help debug
        for k, df in latest_data.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                print(f"Dataset {k} shape: {df.shape}, columns: {df.columns.tolist()}")
                print(f"Dataset {k} index: {df.index}")
        return pd.DataFrame()

# Make predictions
def predict_accretion(df, models):
    """Make predictions using the loaded models and direct temperature analysis"""
    results = {}
    
    # First check for direct temperature indicators of accretion
    # If temperatures in multiple zones are below expected thresholds, this is a strong indicator
    zone_temps = {}
    low_temp_zones = []
    
    # Expected normal temperature ranges for each zone
    zone_temp_thresholds = {
        0: 750,  # Zone 0 should be above 750°C
        1: 775,  # Zone 1 should be above 775°C
        2: 800,  # Zone 2 should be above 800°C
        3: 825,  # Zone 3 should be above 825°C
        4: 850,  # Zone 4 should be above 850°C
        5: 875,  # Zone 5 should be above 875°C
        6: 875,  # Zone 6 should be above 875°C
        7: 875,  # Zone 7 should be above 875°C
        8: 850,  # Zone 8 should be above 850°C
        9: 825,  # Zone 9 should be above 825°C
        10: 800, # Zone 10 should be above 800°C
    }
    
    # Extract zone temperatures if available
    for i in range(11):
        zone_col = f'ZONE_{i}'
        if zone_col in df.columns:
            temp = df[zone_col].iloc[-1] if not df.empty else None
            if temp is not None:
                zone_temps[i] = float(temp)
                # Check if temperature is below threshold
                if float(temp) < zone_temp_thresholds.get(i, 850):
                    low_temp_zones.append(i)
    
    # If multiple zones (3+) have low temperatures, this is a strong indicator of accretion
    direct_detection = False
    most_affected_zone = None
    
    if len(low_temp_zones) >= 3:
        direct_detection = True
        # Find the zone with the largest temperature drop relative to its threshold
        if zone_temps:
            temp_drops = {zone: zone_temp_thresholds.get(zone, 850) - zone_temps[zone] 
                         for zone in low_temp_zones}
            most_affected_zone = max(temp_drops, key=temp_drops.get) if temp_drops else low_temp_zones[0]
    
    # Then try using the ML model if available
    if 'predictor' in models:
        predictor = models['predictor']
        
        try:
            # Filter out target columns that are causing model issues
            input_df = df.copy()
            target_columns = ['accretion_zone', 'days_to_critical', 'target_accretion_critical', 'target_accretion_forming']
            for col in target_columns:
                if col in input_df.columns:
                    input_df = input_df.drop(columns=[col])
            
            # Make model predictions
            predictions = predictor.predict(input_df)
            
            # Format results
            if 'binary' in predictions:
                results['is_forming'] = bool(predictions['binary'][-1]) or direct_detection
            else:
                results['is_forming'] = direct_detection
            
            if 'binary_proba' in predictions:
                # If we have direct temperature evidence, boost probability
                model_prob = float(predictions['binary_proba'][-1])
                results['probability'] = max(model_prob, 0.7 if direct_detection else 0.0)
            else:
                results['probability'] = 0.7 if direct_detection else 0.1
            
            if 'days' in predictions:
                results['days_to_critical'] = float(predictions['days'][-1])
            else:
                results['days_to_critical'] = 15 if direct_detection else 30
            
            if 'zone' in predictions and not direct_detection:
                results['zone'] = int(predictions['zone'][-1])
            elif most_affected_zone is not None:
                # Use zone with largest temperature drop if directly detected
                results['zone'] = most_affected_zone
            
            return results
            
        except Exception as e:
            print(f"Error making predictions with model: {e}")
    
    # If model prediction failed or no model available, use direct temperature detection
    return {
        'is_forming': direct_detection,
        'probability': 0.7 if direct_detection else 0.1,
        'days_to_critical': 15 if direct_detection else 30,
        'zone': most_affected_zone if most_affected_zone is not None else 5
    }

# Get prescriptions
def get_prescriptions(df, models):
    """Get parameter adjustment prescriptions"""
    if 'prescriptor' not in models:
        return {}
    
    prescriptor = models['prescriptor']
    
    try:
        # Get predictions for the current data
        predictions = predict_accretion(df, models)
        
        # Use the prescribe method on our KilnAccretionPrescriptor
        if hasattr(prescriptor, 'prescribe'):
            recommendations = prescriptor.prescribe(df, predictions)
            
            # Convert recommendations to format expected by UI
            result = {}
            # Convert the recommendations to the expected format for the UI
            for rec in recommendations:
                action = rec.get('action', '')
                priority = rec.get('priority', 'medium')
                
                # Format recommendations as adjustments to match expected UI format
                if "reduce feed" in action.lower():
                    result['mis_IRON ORE CONSUMPTION'] = -0.15
                elif "adjust coal/ore" in action.lower():
                    result['mis_GROSS COAL CONSUMPTION'] = -0.10
                    result['mis_IRON ORE CONSUMPTION'] = 0.05
                elif "monitor temperature" in action.lower():
                    # Add a small damper adjustment as a monitoring action
                    result['air_DAMPER 1'] = 2.0
                    result['air_DAMPER 2'] = -2.0
                
                # Add the original recommendation text to the result
                result[f"recommendation_{len(result)}"] = f"{action} ({priority})"
            
            return result
        else:
            # Fallback to old method or empty result
            print("Prescriptor has no prescribe method")
            return {}
    except Exception as e:
        print(f"Error generating prescriptions: {e}")
        return {}

# Create a function to simulate real-time data (in a real system, this would be replaced with actual data acquisition)
def simulate_real_time_data():
    """Simulate real-time data flow"""
    while True:
        # Load the latest data
        data = load_data()
        
        # Push to queue
        if data:
            data_queue.put(data)
            
        # Wait for next update
        time.sleep(REFRESH_INTERVAL)

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[DASH_THEME, "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"], suppress_callback_exceptions=True)
app.title = "Kiln Accretion Monitoring System"

colors = {
    'background': '#181a1b',
    'text': '#FFFFFF',
    'primary': '#00bfff',
    'success': '#28A745',
    'danger': '#DC3545',
    'warning': '#FFC107',
    'info': '#17A2B8',
    'card': '#23272b',
}

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Kiln Accretion Monitoring System", 
                   style={'textAlign': 'center', 'color': colors['primary'], 'marginTop': 20, 'fontWeight': 'bold', 'fontSize': '2.5rem', 'letterSpacing': '1px'}),
            html.Div(id='last-update-time', 
                    style={'textAlign': 'center', 'color': colors['info'], 'fontSize': 16, 'marginBottom': 10})
        ], width=12)
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(card_header("Accretion Status")),
                dbc.CardBody([
                    html.Div([
                        html.H3(id='accretion-status', style={'textAlign': 'center', 'fontWeight': 'bold', 'fontSize': '2rem'}),
                        html.Div(id='accretion-details', style={'textAlign': 'center', 'marginTop': 10, 'fontSize': '1.1rem'}),
                        dbc.Progress(id='probability-bar', style={'marginTop': 20, 'height': '2.2rem', 'transition': 'width 0.6s cubic-bezier(.4,0,.2,1)'}),
                    ])
                ])
            ], style=CARD_STYLE)
        ], width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(card_header("Current Kiln Parameters")),
                dbc.CardBody([
                    html.Div(id='current-parameters', style={'overflowY': 'auto', 'height': '200px', 'fontSize': '1.05rem'})
                ])
            ], style=CARD_STYLE)
        ], width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(card_header("Recommended Actions")),
                dbc.CardBody([
                    html.Div(id='recommended-actions', style={'overflowY': 'auto', 'height': '200px', 'fontSize': '1.05rem'})
                ])
            ], style=CARD_STYLE)
        ], width=4)
    ], className='mb-4'),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(card_header("3D Kiln Visualization")),
                dbc.CardBody([
                    dcc.Graph(id='kiln-3d-visualization', style={'height': '450px'}),
                    html.Div([
                        html.Label("Time Slider:", style={'fontWeight': 'bold', 'marginTop': '10px'}),
                        dcc.Slider(
                            id='time-slider',
                            min=0,
                            max=1,
                            step=None,
                            marks={},
                            value=1,
                            tooltip={'placement': 'bottom', 'always_visible': True}
                        ),
                        html.Div(id='slider-date-display', style={'textAlign': 'center', 'marginTop': '5px'})
                    ], style={'padding': '10px 20px'})
                ])
            ], style=CARD_STYLE)
        ], width=8),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(card_header("Zone Temperatures")),
                dbc.CardBody([
                    dcc.Graph(id='zone-temperatures', style={'height': '240px'})
                ])
            ], style=CARD_STYLE, className='mb-4'),
            dbc.Card([
                dbc.CardHeader(card_header("Production Quality")),
                dbc.CardBody([
                    dcc.Graph(id='production-quality', style={'height': '240px'})
                ])
            ], style=CARD_STYLE)
        ], width=4)
    ], className='mb-4'),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(card_header("Material Consumption")),
                dbc.CardBody([
                    dcc.Graph(id='material-consumption', style={'height': '300px'})
                ])
            ], style=CARD_STYLE)
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(card_header("Historical Accretion Events")),
                dbc.CardBody([
                    dcc.Graph(id='historical-events', style={'height': '300px'})
                ])
            ], style=CARD_STYLE)
        ], width=6)
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(card_header("Material Quality Impact Analysis")),
                dbc.CardBody([
                    dbc.Tabs([
                        dbc.Tab(dcc.Graph(id="material-quality-grades", style={'height': '350px'}), label="Quality vs Grades"),
                        dbc.Tab(dcc.Graph(id="material-quality-accretion", style={'height': '350px'}), label="Quality vs Accretion"),
                    ])
                ])
            ], style=CARD_STYLE)
        ], width=12)
    ], className='mb-4'),
    dbc.Row([
        dbc.Col([            dbc.Card([
                dbc.CardHeader(card_header("Model Feature Importance")),
                dbc.CardBody([                    dbc.Tabs([
                        dbc.Tab(dcc.Graph(id="feature-importance-accretion", style={'height': '300px'}), label="Accretion Features"),
                        dbc.Tab(dcc.Graph(id="feature-importance-days", style={'height': '300px'}), label="Days to Critical Features"),
                        dbc.Tab(dcc.Graph(id="feature-importance-zone", style={'height': '300px'}), label="Zone Features"),
                        dbc.Tab(dcc.Graph(id="incremental-learning-metrics", style={'height': '300px'}), label="Learning Progress"),
                    ]),
                    html.Hr(),
                    html.Div([
                        dbc.Button(
                            "Update Model with Latest Data", 
                            id="incremental-update-button", 
                            color="primary", 
                            className="mt-3"
                        ),
                        html.Div(id="incremental-update-status", className="mt-2")
                    ], style={'textAlign': 'center'})
                ])
            ], style=CARD_STYLE)
        ], width=12)
    ]),
    # Hidden div for storing intermediate data
    html.Div(id='intermediate-data', style={'display': 'none'}),
    
    # Data refresh interval
    dcc.Interval(
        id='data-refresh',
        interval=REFRESH_INTERVAL * 1000,  # Convert to milliseconds
        n_intervals=0
    )
], fluid=True, style={'backgroundColor': colors['background'], 'minHeight': '100vh', 'padding': '24px'})

# Callback to update the time slider based on available data
@app.callback(
    Output('time-slider', 'min'),
    Output('time-slider', 'max'),
    Output('time-slider', 'marks'),
    Output('time-slider', 'value'),
    Output('time-slider', 'step'),
    Input('intermediate-data', 'children')
)
def update_time_slider(json_data):
    """Update the time slider based on available data"""
    if not json_data:
        # Default empty slider
        return 0, 1, {0: 'No data', 1: 'No data'}, 1, 1
    
    try:
        # Parse the JSON data
        data_dict = json.loads(json_data)
        
        # Use zone temperature data for time range if available
        if 'zone' in data_dict:
            df_zone = pd.read_json(data_dict['zone'], orient='split')
            if 'DATETIME' in df_zone.columns:
                df_zone['DATETIME'] = pd.to_datetime(df_zone['DATETIME'])
                
                # Get unique dates (not times) for the slider
                unique_dates = df_zone['DATETIME'].dt.date.unique()
                dates_list = sorted(unique_dates)
                
                # Create slider marks - show one mark every 7 days to avoid overcrowding
                marks = {}
                for i, date in enumerate(dates_list):
                    if i % 7 == 0 or i == len(dates_list) - 1:
                        date_str = date.strftime('%Y-%m-%d')
                        marks[i] = date_str
                
                # Return slider properties
                min_val = 0
                max_val = len(dates_list) - 1
                current_val = max_val  # Default to most recent
                step = 1
                
                return min_val, max_val, marks, current_val, step
        
        # Fallback if no zone data
        return 0, 1, {0: 'Start', 1: 'End'}, 1, 1
        
    except Exception as e:
        print(f"Error updating time slider: {e}")
        return 0, 1, {0: 'Error', 1: 'Error'}, 1, 1

# Update the display of the selected date
@app.callback(
    Output('slider-date-display', 'children'),
    Input('time-slider', 'value'),
    Input('intermediate-data', 'children')
)
def update_date_display(slider_value, json_data):
    """Show the selected date based on slider value"""
    if not json_data or slider_value is None:
        return "No date selected"
    
    try:
        # Parse the JSON data
        data_dict = json.loads(json_data)
        
        # Get dates from zone data
        if 'zone' in data_dict:
            df_zone = pd.read_json(data_dict['zone'], orient='split')
            if 'DATETIME' in df_zone.columns:
                df_zone['DATETIME'] = pd.to_datetime(df_zone['DATETIME'])
                unique_dates = df_zone['DATETIME'].dt.date.unique()
                dates_list = sorted(unique_dates)
                
                if 0 <= slider_value < len(dates_list):
                    selected_date = dates_list[slider_value]
                    return f"Selected Date: {selected_date.strftime('%Y-%m-%d')}"
        
        return f"Position: {slider_value}"
        
    except Exception as e:
        print(f"Error updating date display: {e}")
        return "Date display error"

# Callback to update data
@app.callback(
    Output('intermediate-data', 'children'),
    Output('last-update-time', 'children'),
    Input('data-refresh', 'n_intervals')
)
def update_data(n_intervals):
    """Update data from queue or load data"""
    try:
        # Try to get data from queue (non-blocking)
        try:
            data = data_queue.get(block=False)
        except queue.Empty:
            # If queue is empty, load data from files
            data = load_data()
        
        # Store data as JSON for intermediate use
        json_data = {
            key: value.to_json(date_format='iso', orient='split') 
            for key, value in data.items() 
            if isinstance(value, pd.DataFrame)
        }
        
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        return json.dumps(json_data), f"Last Updated: {current_time}"
    except Exception as e:
        print(f"Error updating data: {e}")
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return dash.no_update, f"Last Updated (Error): {current_time}"

# Callback to update accretion status
@app.callback(
    Output('accretion-status', 'children'),
    Output('accretion-status', 'style'),
    Output('accretion-details', 'children'),
    Output('probability-bar', 'value'),
    Output('probability-bar', 'color'),
    Input('intermediate-data', 'children')
)
def update_accretion_status(json_data):
    """Update accretion status display"""
    if not json_data:
        return "No Data", {}, "No prediction data available", 0, colors['info']
    
    try:
        # Parse the JSON data
        data_dict = json.loads(json_data)
        # Convert each JSON to DataFrame and ensure it has proper datetime index
        data = {}
        for key, value in data_dict.items():
            df = pd.read_json(value, orient='split')
            # Ensure index is datetime if it has date-like values
            if not isinstance(df.index, pd.DatetimeIndex) and df.index.dtype == 'object':
                try:
                    df.index = pd.to_datetime(df.index)
                except:
                    pass  # If conversion fails, keep original index
            data[key] = df
        
        # Load models and preprocessor
        models = load_models()
        preprocessor = load_preprocessor()
        
        # Process latest data
        latest_df = process_latest_data(data, preprocessor)
        
        if latest_df.empty:
            return "No Data", {}, "No prediction data available", 0, colors['info']
        
        # Make predictions
        predictions = predict_accretion(latest_df, models)
        
        if not predictions:
            return "No Predictions", {}, "Prediction model not loaded or error", 0, colors['info']
        
        # Format the output
        is_forming = predictions.get('is_forming', False)
        probability = predictions.get('probability', 0.0) * 100  # Convert to percentage
        days_to_critical = predictions.get('days_to_critical', None)
        zone = predictions.get('zone', None)
          # Extract zone temperatures for additional context
        zone_temps = {}
        for i in range(11):
            zone_col = f'ZONE_{i}'
            if zone_col in latest_df.columns:
                zone_temps[i] = float(latest_df[zone_col].iloc[-1])
        
        # Count low temperature zones
        low_temp_zones = []
        zone_temp_thresholds = {
            0: 750, 1: 775, 2: 800, 3: 825, 4: 850, 5: 875, 
            6: 875, 7: 875, 8: 850, 9: 825, 10: 800
        }
        for zone_num, temp in zone_temps.items():
            if temp < zone_temp_thresholds.get(zone_num, 850):
                low_temp_zones.append(zone_num)
                
        if is_forming:
            status_text = "Accretion Forming"
            status_style = {'color': colors['danger'], 'textAlign': 'center'}
            
            details = []
            if days_to_critical is not None:
                details.append(f"Estimated {days_to_critical:.1f} days until critical accretion")
                
            if zone is not None:
                details.append(f"Predicted location: Zone {zone}")
                
            # Add information about low temperature zones
            if low_temp_zones:
                details.append(f"Low temperature detected in {len(low_temp_zones)} zones: {', '.join(map(str, sorted(low_temp_zones)))}")
                
            details_text = html.Div([
                html.P(d, style={'margin': '5px 0'}) for d in details
            ])
            
            progress_color = colors['danger']
        elif len(low_temp_zones) >= 3:
            # Even if model doesn't predict accretion, warn about multiple low temperature zones
            status_text = "Temperature Anomaly"
            status_style = {'color': colors['warning'], 'textAlign': 'center'}
            
            details = [
                f"Low temperature detected in {len(low_temp_zones)} zones: {', '.join(map(str, sorted(low_temp_zones)))}",
                "Temperatures below normal operating range may indicate early accretion formation"
            ]
            
            details_text = html.Div([
                html.P(d, style={'margin': '5px 0'}) for d in details
            ])
            
            progress_color = colors['warning']
        else:
            status_text = "Normal Operation"
            status_style = {'color': colors['success'], 'textAlign': 'center'}
            details_text = "No accretion forming detected"
            progress_color = colors['success']
        
        return status_text, status_style, details_text, probability, progress_color
        
    except Exception as e:
        print(f"Error updating accretion status: {e}")
        return "Error", {'color': colors['warning'], 'textAlign': 'center'}, f"Error: {str(e)}", 0, colors['warning']

# Callback to update current parameters
@app.callback(
    Output('current-parameters', 'children'),
    Input('intermediate-data', 'children')
)
def update_current_parameters(json_data):
    """Update current parameters display"""
    if not json_data:
        return html.P("No data available")
    
    try:
        # Parse the JSON data
        data_dict = json.loads(json_data)
        # Convert each JSON to DataFrame and ensure it has proper datetime index
        data = {}
        for key, value in data_dict.items():
            df = pd.read_json(value, orient='split')
            # Ensure index is datetime if it has date-like values
            if not isinstance(df.index, pd.DatetimeIndex) and df.index.dtype == 'object':
                try:
                    df.index = pd.to_datetime(df.index)
                except:
                    pass  # If conversion fails, keep original index
            data[key] = df
        
        # Prepare parameter display
        parameters = []
        
        # Zone temperatures (latest)
        if 'zone' in data and not data['zone'].empty:
            latest_zone = data['zone'].iloc[-1]
            zone_temps = []
            
            for i in range(11):
                col_name = f'ZONE_{i}'
                if col_name in latest_zone:
                    temp = latest_zone[col_name]
                    zone_temps.append(html.Div([
                        html.Span(f"Zone {i}: ", style={'fontWeight': 'bold'}),
                        f"{temp:.1f}°C"
                    ], style={'margin': '3px 0'}))
            
            if zone_temps:
                parameters.append(html.Div([
                    html.H5("Zone Temperatures:", style={'borderBottom': '1px solid #444', 'paddingBottom': '5px'}),
                    html.Div(zone_temps)
                ], style={'marginBottom': '15px'}))
        
        # Material consumption (latest)
        if 'mis' in data and not data['mis'].empty:
            latest_mis = data['mis'].iloc[-1]
            material_params = []
            
            key_parameters = [
                ('IRON ORE CONSUMPTION', 'Iron Ore'),
                ('GROSS COAL CONSUMPTION', 'Coal'),
                ('DOLO CONSUMPTION', 'Dolomite'),
                ('PRODUCTION ACTUAL', 'Production')
            ]
            
            for key, label in key_parameters:
                if key in latest_mis:
                    value = latest_mis[key]
                    material_params.append(html.Div([
                        html.Span(f"{label}: ", style={'fontWeight': 'bold'}),
                        f"{value:.1f} tons"
                    ], style={'margin': '3px 0'}))
            
            if material_params:
                parameters.append(html.Div([
                    html.H5("Material Parameters:", style={'borderBottom': '1px solid #444', 'paddingBottom': '5px'}),
                    html.Div(material_params)
                ], style={'marginBottom': '15px'}))
        
        # Air flow parameters (latest)
        if 'air' in data and not data['air'].empty:
            latest_air = data['air'].iloc[-1]
            air_params = []
            
            damper_cols = [col for col in latest_air.index if 'DAMPER' in col]
            flow_cols = [col for col in latest_air.index if 'AIR FLOW' in col]
            
            for col in damper_cols:
                if pd.notna(latest_air[col]):
                    air_params.append(html.Div([
                        html.Span(f"{col}: ", style={'fontWeight': 'bold'}),
                        f"{latest_air[col]:.1f}%"
                    ], style={'margin': '3px 0'}))
            
            for col in flow_cols:
                if pd.notna(latest_air[col]):
                    air_params.append(html.Div([
                        html.Span(f"{col}: ", style={'fontWeight': 'bold'}),
                        f"{latest_air[col]:.1f} m³/h"
                    ], style={'margin': '3px 0'}))
            
            if air_params:
                parameters.append(html.Div([
                    html.H5("Air Parameters:", style={'borderBottom': '1px solid #444', 'paddingBottom': '5px'}),
                    html.Div(air_params)
                ]))
        
        if not parameters:
            return html.P("No parameter data available")
            
        return parameters
        
    except Exception as e:
        print(f"Error updating current parameters: {e}")
        return html.P(f"Error loading parameters: {str(e)}")

# Callback to update recommended actions
@app.callback(
    Output('recommended-actions', 'children'),
    Input('intermediate-data', 'children')
)
def update_recommended_actions(json_data):
    """Update recommended actions display"""
    if not json_data:
        return html.P("No data available")
    
    try:
        # Parse the JSON data
        data_dict = json.loads(json_data)
        # Convert each JSON to DataFrame and ensure it has proper datetime index
        data = {}
        for key, value in data_dict.items():
            df = pd.read_json(value, orient='split')
            # Ensure index is datetime if it has date-like values
            if not isinstance(df.index, pd.DatetimeIndex) and df.index.dtype == 'object':
                try:
                    df.index = pd.to_datetime(df.index)
                except:
                    pass  # If conversion fails, keep original index
            data[key] = df
        
        # Load models and preprocessor
        models = load_models()
        preprocessor = load_preprocessor()
        
        # Process latest data
        latest_df = process_latest_data(data, preprocessor)
        
        if latest_df.empty or 'prescriptor' not in models:
            return html.P("No prescription data available")
        
        # Make predictions to check if accretion is forming
        predictions = predict_accretion(latest_df, models)
        
        is_forming = predictions.get('is_forming', False)
        probability = predictions.get('probability', 0.0)
              # Always get prescriptions if accretion is forming or probability is above a threshold
        if is_forming or probability > 0.2:
            # Get prescriptions
            prescriptions = get_prescriptions(latest_df, models)
            
            if not prescriptions:
                return html.P("Recommendations not available - check prescription model")
            
            # Format recommendations
            recommendations = []
            
            # Direct recommendations from prescriptor (new format)
            direct_recommendations = []
            for k, v in prescriptions.items():
                if k.startswith('recommendation_'):
                    direct_recommendations.append(v)
            
            if direct_recommendations:
                # Sort recommendations by priority (high, medium, low, info)
                priority_order = {'high': 0, 'medium': 1, 'low': 2, 'info': 3}
                
                # Parse out priority information and sort
                parsed_recs = []
                for rec in direct_recommendations:
                    priority_match = rec.lower()
                    if 'high' in priority_match:
                        style = {'color': colors['danger'], 'fontWeight': 'bold'}
                        icon = 'fa-solid fa-triangle-exclamation'
                        priority = 'high'
                    elif 'medium' in priority_match:
                        style = {'color': colors['warning']}
                        icon = 'fa-solid fa-exclamation'
                        priority = 'medium'
                    elif 'low' in priority_match:
                        style = {'color': colors['info']}
                        icon = 'fa-solid fa-info-circle'
                        priority = 'low'
                    else:
                        style = {'color': colors['text']}
                        icon = 'fa-solid fa-circle-check'
                        priority = 'info'
                    
                    # Get just the action text
                    text = rec.split('(')[0].strip()
                    
                    parsed_recs.append({
                        'text': text,
                        'style': style,
                        'icon': icon,
                        'priority': priority_order.get(priority, 4)
                    })
                
                # Sort by priority
                parsed_recs.sort(key=lambda x: x['priority'])
                
                # Create recommendation elements
                rec_elements = []
                for rec in parsed_recs:
                    rec_elements.append(html.Div([
                        html.I(className=rec['icon'], style={'marginRight': '10px'}),
                        html.Span(rec['text'])
                    ], style={**rec['style'], 'margin': '8px 0', 'padding': '5px 0'}))
                
                recommendations.append(html.Div([
                    html.H5("Recommended Actions:", style={'borderBottom': '1px solid #444', 'paddingBottom': '5px', 'marginBottom': '10px'}),
                    html.Div(rec_elements)
                ], style={'marginBottom': '15px'}))
            
            # Air damper adjustments - show even small adjustments
            damper_adjustments = {k: v for k, v in prescriptions.items() if 'DAMPER' in k and abs(v) > 0.5}
            if damper_adjustments:
                damper_recs = []
                for param, value in damper_adjustments.items():
                    param_name = param.replace('air_DAMPER ', '')
                    direction = "Increase" if value > 0 else "Decrease"
                    damper_recs.append(html.Div([
                        html.Span(f"{param_name}: ", style={'fontWeight': 'bold'}),
                        f"{direction} by {abs(value):.1f}%"
                    ], style={'margin': '3px 0'}))
                
                recommendations.append(html.Div([
                    html.H5("Adjust Air Dampers:", style={'borderBottom': '1px solid #444', 'paddingBottom': '5px'}),
                    html.Div(damper_recs)
                ], style={'marginBottom': '15px'}))
              # Material adjustments - show even small adjustments 
            material_adjustments = {k: v for k, v in prescriptions.items() if 'CONSUMPTION' in k and abs(v) > 0.01}
            if material_adjustments:
                material_recs = []
                for param, value in material_adjustments.items():
                    param_name = param.replace('mis_', '').replace(' CONSUMPTION', '')
                    direction = "Increase" if value > 0 else "Decrease"
                    material_recs.append(html.Div([
                        html.Span(f"{param_name}: ", style={'fontWeight': 'bold'}),
                        f"{direction} by {abs(value) * 100:.1f}%"
                    ], style={'margin': '3px 0'}))
                
                recommendations.append(html.Div([
                    html.H5("Adjust Materials:", style={'borderBottom': '1px solid #444', 'paddingBottom': '5px'}),
                    html.Div(material_recs)
                ], style={'marginBottom': '15px'}))
            
            if recommendations:
                return recommendations
            else:
                return html.P("No significant parameter adjustments recommended")
        else:
            return html.Div([
                html.H5("System Status", style={'borderBottom': '1px solid #444', 'paddingBottom': '5px'}),
                html.P("Normal operation - no adjustments needed", style={'color': colors['success']})
            ])
        
    except Exception as e:
        print(f"Error updating recommended actions: {e}")
        return html.P(f"Error generating recommendations: {str(e)}")

# Callback to update 3D kiln visualization
@app.callback(
    Output('kiln-3d-visualization', 'figure'),
    Input('intermediate-data', 'children'),
    Input('time-slider', 'value')
)
def update_3d_visualization(json_data, slider_value):
    """Update 3D kiln visualization"""
    if not json_data:
        # Create empty 3D cylinder
        return create_empty_kiln_viz()
    
    try:
        # Parse the JSON data
        data_dict = json.loads(json_data)
        # Convert each JSON to DataFrame and ensure it has proper datetime index
        data = {}
        for key, value in data_dict.items():
            df = pd.read_json(value, orient='split')
            # Ensure index is datetime if it has date-like values
            if not isinstance(df.index, pd.DatetimeIndex) and df.index.dtype == 'object':
                try:
                    df.index = pd.to_datetime(df.index)
                except:
                    pass  # If conversion fails, keep original index
            data[key] = df
            
        # Get zone temperatures for the selected date from the slider
        if 'zone' in data and not data['zone'].empty:
            df_zone = data['zone']
            
            # Process based on the slider value
            if slider_value is not None and 'DATETIME' in df_zone.columns:
                df_zone['DATETIME'] = pd.to_datetime(df_zone['DATETIME'])
                # Get unique dates
                unique_dates = df_zone['DATETIME'].dt.date.unique()
                dates_list = sorted(unique_dates)
                
                if 0 <= slider_value < len(dates_list):
                    # Filter data for the selected date
                    selected_date = dates_list[slider_value]
                    selected_date_rows = df_zone[df_zone['DATETIME'].dt.date == selected_date]
                    
                    if not selected_date_rows.empty:
                        # Use the latest row for that date
                        selected_zone = selected_date_rows.iloc[-1]
                    else:
                        # Fallback to the latest zone data
                        selected_zone = df_zone.iloc[-1]
                else:
                    # Default to the latest data
                    selected_zone = df_zone.iloc[-1]
            else:
                # If no valid slider value, use latest data
                selected_zone = df_zone.iloc[-1]
                
            # Extract zone temperatures
            zone_temps = []
            for i in range(11):
                col_name = f'ZONE_{i}'
                if col_name in selected_zone:
                    zone_temps.append(float(selected_zone[col_name]))
                else:
                    zone_temps.append(900)  # Default temperature if missing
        else:
            zone_temps = [900] * 11  # Default temperatures
        
        # Get predictions if available
        models = load_models()
        preprocessor = load_preprocessor()
        latest_df = process_latest_data(data, preprocessor)
        
        accretion_zone = None
        is_forming = False
        
        if not latest_df.empty and 'predictor' in models:
            predictions = predict_accretion(latest_df, models)
            is_forming = predictions.get('is_forming', False)
            if is_forming and 'zone' in predictions:
                accretion_zone = predictions.get('zone')
        
        # Create 3D visualization
        return create_3d_kiln(zone_temps, accretion_zone)
        
    except Exception as e:
        print(f"Error updating 3D visualization: {e}")
        return create_empty_kiln_viz()

def create_empty_kiln_viz():
    """Create an empty kiln visualization (horizontal orientation)"""
    # Create a cylinder representing the kiln
    fig = go.Figure()
      # Add an empty cylinder - horizontal orientation
    phi = np.linspace(0, 2*np.pi, 100)
    x = np.linspace(0, kiln_length, 50)  # x is now the length
    phi_grid, x_grid = np.meshgrid(phi, x)
    
    y = kiln_diameter/2 * np.cos(phi_grid)
    z = kiln_diameter/2 * np.sin(phi_grid)  # z is vertical
    
    colorscale = [[0, '#000080'], [0.2, '#0000FF'], [0.5, '#00FFFF'], [0.7, '#FFFF00'], [1, '#FF0000']]
    
    fig.add_trace(go.Surface(
        x=x_grid, y=y, z=z,
        colorscale=colorscale,
        surfacecolor=np.ones_like(x_grid) * 0.5,
        showscale=True,
        colorbar=dict(
            title='Temperature (°C)',
            tickvals=[0, 0.2, 0.5, 0.7, 1],
            ticktext=['650', '750', '850', '950', '1100']
        )
    ))
    
    # Add shape markers for zone separations
    for i in range(1, 11):
        x_pos = i * (kiln_length / 11)
        fig.add_trace(go.Scatter3d(
            x=[x_pos, x_pos], 
            y=[0, 0], 
            z=[kiln_diameter/2, kiln_diameter/2],  # z is vertical now
            mode='markers',
            marker=dict(size=3, color='white'),
            showlegend=False
        ))
    
    # Update layout
    fig.update_layout(
        title='3D Kiln Temperature Visualization',
        scene=dict(
            aspectratio=dict(x=3, y=1, z=1),  # Horizontal aspect ratio
            xaxis_title='Length (m)',
            yaxis_title='Width (m)',
            zaxis_title='Height (m)',
            xaxis=dict(range=[0, kiln_length]),
            yaxis=dict(range=[-kiln_diameter, kiln_diameter]),
            zaxis=dict(range=[0, kiln_length])
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    return fig

def create_3d_kiln(zone_temps, accretion_zone=None):
    """Create a 3D visualization of the kiln with temperature data (horizontal orientation)"""
    # Create a cylinder representing the kiln
    fig = go.Figure()
    
    # Interpolate temperatures to create a smooth gradient
    zone_positions = np.linspace(0, kiln_length, 11)
    interp_func = interp1d(zone_positions, zone_temps, kind='cubic', fill_value="extrapolate")
    
    # Create a cylinder mesh - horizontal orientation (x is length, z is vertical)
    phi = np.linspace(0, 2*np.pi, 100)
    x = np.linspace(0, kiln_length, 50)  # x is now the length
    phi_grid, x_grid = np.meshgrid(phi, x)
    
    y = kiln_diameter/2 * np.cos(phi_grid)
    z = kiln_diameter/2 * np.sin(phi_grid)  # z is vertical position
    
    # Get interpolated temperatures for each x position
    temp_grid = np.array([interp_func(xi) for xi in x])
    temp_grid = np.tile(temp_grid[:, np.newaxis], (1, len(phi)))    # Normalize temperatures for color mapping
    # Updated normalization to better highlight cool spots (potential accretion)
    norm_temps = (temp_grid - 650) / (1100 - 650)  # Normalize to 650-1100°C range to capture drops
    norm_temps = np.clip(norm_temps, 0, 1)  # Clip to 0-1
    
    # Create the surface with temperature mapping
    # Updated colorscale: dark blue for cold spots (potential accretion), red for hot
    colorscale = [[0, '#000080'], [0.2, '#0000FF'], [0.5, '#00FFFF'], [0.7, '#FFFF00'], [1, '#FF0000']]
    
    fig.add_trace(go.Surface(
        x=x_grid, y=y, z=z,  # Now x is length, z is vertical
        colorscale=colorscale,
        surfacecolor=norm_temps,
        showscale=True,        
        colorbar=dict(
            title='Temperature (°C)',
            tickvals=[0, 0.2, 0.5, 0.7, 1],
            ticktext=['650', '750', '850', '950', '1100']
        )
    ))
    
    # Add markers for zone positions
    for i in range(11):
        x_pos = i * (kiln_length / 10)
        fig.add_trace(go.Scatter3d(
            x=[x_pos, x_pos],  # x position (along kiln length) 
            y=[0, 0], 
            z=[kiln_diameter/2, kiln_diameter/2],  # z is now vertical
            mode='markers+text',
            marker=dict(size=5, color='white'),
            text=[f'Zone {i}', ''],
            textposition='top center',
            showlegend=False
        ))
      # Add accretion visualization if predicted
    if accretion_zone is not None:
        # Calculate position along kiln
        accr_x = accretion_zone * (kiln_length / 10)  # x is now position along kiln
        
        # Create a ring to represent accretion - make it more prominent
        accr_phi = np.linspace(0, 2*np.pi, 30)
        accr_y = (kiln_diameter/2 - 0.3) * np.cos(accr_phi)  # y is horizontal
        accr_z = (kiln_diameter/2 - 0.3) * np.sin(accr_phi)  # z is vertical
        accr_x = np.ones_like(accr_phi) * accr_x
        
        # Add a second, larger ring to better highlight the cold spot
        accr_y2 = (kiln_diameter/2 - 0.1) * np.cos(accr_phi)
        accr_z2 = (kiln_diameter/2 - 0.1) * np.sin(accr_phi)
          # First add the outer marker to highlight cold zone
        fig.add_trace(go.Scatter3d(
            x=accr_x, y=accr_y2, z=accr_z2,
            mode='lines',
            line=dict(color='darkblue', width=5),
            name='Cold Zone',
            opacity=0.7,
            showlegend=True
        ))
        
        # Then add the accretion ring
        fig.add_trace(go.Scatter3d(
            x=accr_x, y=accr_y, z=accr_z,
            mode='lines',
            line=dict(color='brown', width=10),
            name='Accretion',
            showlegend=True
        ))
    
    # Update layout
    fig.update_layout(
        title='3D Kiln Temperature Visualization',        scene=dict(
            aspectratio=dict(x=3, y=1, z=1),  # Changed aspect ratio to make it horizontal
            xaxis_title='Length (m)',
            yaxis_title='Width (m)',
            zaxis_title='Height (m)',
            xaxis=dict(range=[0, kiln_length]),
            yaxis=dict(range=[-kiln_diameter, kiln_diameter]),
            zaxis=dict(range=[-kiln_diameter, kiln_diameter])
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    return fig

# Callback to update zone temperature chart
@app.callback(
    Output('zone-temperatures', 'figure'),
    Input('intermediate-data', 'children')
)
def update_zone_temperatures(json_data):
    """Update zone temperature trend chart"""
    if not json_data:
        return empty_chart("Zone Temperature Trends")
    
    try:
        # Parse the JSON data
        data_dict = json.loads(json_data)
        # Convert each JSON to DataFrame and ensure it has proper datetime index
        data = {}
        for key, value in data_dict.items():
            df = pd.read_json(value, orient='split')
            # Ensure index is datetime if it has date-like values
            if not isinstance(df.index, pd.DatetimeIndex) and df.index.dtype == 'object':
                try:
                    df.index = pd.to_datetime(df.index)
                except:
                    pass  # If conversion fails, keep original index
            data[key] = df
        
        if 'zone' not in data or data['zone'].empty:
            return empty_chart("Zone Temperature Trends")
        
        zone_df = data['zone']
        
        # Get a sample of points to avoid overplotting
        if len(zone_df) > 500:
            zone_df = zone_df.iloc[::int(len(zone_df)/500)]
            
        fig = go.Figure()
        
        # Add a line for each zone
        for i in range(0, 11, 2):  # Plot every other zone to avoid cluttering
            col = f'ZONE_{i}'
            if col in zone_df.columns:
                fig.add_trace(go.Scatter(
                    x=zone_df['DATETIME'],
                    y=zone_df[col],
                    mode='lines',
                    name=f'Zone {i}'
                ))
        
        fig.update_layout(
            title='Zone Temperature Trends',
            xaxis_title='Time',
            yaxis_title='Temperature (°C)',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
            margin=dict(l=10, r=10, t=40, b=10),
            height=240,
            template='plotly_dark'
        )
        
        return fig
        
    except Exception as e:
        print(f"Error updating zone temperatures chart: {e}")
        return empty_chart("Zone Temperature Trends (Error)")

# Callback to update production quality chart
@app.callback(
    Output('production-quality', 'figure'),
    [Input('intermediate-data', 'children'),
     Input('time-slider', 'value')]
)
def update_production_quality(json_data, time_slider_value):
    """Update production quality chart"""
    if not json_data:
        return empty_chart("Production Quality")
    
    try:
        # Parse the JSON data
        data_dict = json.loads(json_data)
        # Convert each JSON to DataFrame and ensure it has proper datetime index
        data = {}
        for key, value in data_dict.items():
            df = pd.read_json(value, orient='split')
            # Ensure index is datetime if it has date-like values
            if not isinstance(df.index, pd.DatetimeIndex) and df.index.dtype == 'object':
                try:
                    df.index = pd.to_datetime(df.index)
                except:
                    pass  # If conversion fails, keep original index
            data[key] = df
        
        if 'mis' not in data or data['mis'].empty:
            return empty_chart("Production Quality")
        
        mis_df = data['mis']
        
        # Create a figure with product grade metrics
        # Use subplots to show both the absolute values and the ratio over time
        fig = make_subplots(rows=1, cols=2, 
                          subplot_titles=["Product Grades", "Grade Proportions"],
                          specs=[[{"type": "scatter"}, {"type": "scatter"}]])
        
        # Check if Grade A and Grade B columns exist
        grade_a_col = next((col for col in mis_df.columns if 'GRADE A' in col), None)
        grade_b_col = next((col for col in mis_df.columns if 'GRADE B' in col), None)
        
        # Add Grade A production
        if grade_a_col and not mis_df[grade_a_col].isna().all():
            fig.add_trace(go.Scatter(
                x=mis_df['DATE'],
                y=mis_df[grade_a_col],
                mode='lines+markers',
                name='Grade A',
                line=dict(color=colors['success']),
                marker=dict(size=6)
            ), row=1, col=1)
        
        # Add Grade B production
        if grade_b_col and not mis_df[grade_b_col].isna().all():
            fig.add_trace(go.Scatter(
                x=mis_df['DATE'],
                y=mis_df[grade_b_col],
                mode='lines+markers',
                name='Grade B',
                line=dict(color=colors['warning']),
                marker=dict(size=6)
            ), row=1, col=1)
        
        # Add total production if available
        if 'PRODUCTION ACTUAL' in mis_df.columns:
            fig.add_trace(go.Scatter(
                x=mis_df['DATE'],
                y=mis_df['PRODUCTION ACTUAL'],
                mode='lines',
                name='Total Production',
                line=dict(color=colors['info'], dash='dot'),
                opacity=0.6
            ), row=1, col=1)
            
        # Calculate grade proportions for the second subplot
        if grade_a_col and grade_b_col and not mis_df[[grade_a_col, grade_b_col]].isna().all().all():
            # Calculate percentage of each grade
            total_production = mis_df[grade_a_col] + mis_df[grade_b_col]
            total_production = total_production.replace(0, np.nan)  # Avoid division by zero
            
            grade_a_pct = (mis_df[grade_a_col] / total_production * 100).fillna(0)
            grade_b_pct = (mis_df[grade_b_col] / total_production * 100).fillna(0)
            
            # Add grade percentages to second subplot
            fig.add_trace(go.Scatter(
                x=mis_df['DATE'],
                y=grade_a_pct,
                mode='lines',
                name='Grade A %',
                line=dict(color=colors['success']),
                fill='tozeroy',
                opacity=0.7
            ), row=1, col=2)
            
            fig.add_trace(go.Scatter(
                x=mis_df['DATE'],
                y=100 - grade_b_pct,  # Start of Grade B area
                mode='lines',
                showlegend=False,
                line=dict(color='rgba(0,0,0,0)'),
                hoverinfo='skip'
            ), row=1, col=2)
            
            fig.add_trace(go.Scatter(
                x=mis_df['DATE'],
                y=100,  # Top of chart
                mode='lines',
                name='Grade B %',
                line=dict(color='rgba(0,0,0,0)'),
                fill='tonexty',
                fillcolor=colors['warning'],
                opacity=0.7
            ), row=1, col=2)
            
            # Add accretion events if available for context
            accretion_col = next((col for col in mis_df.columns if 'ACCRETION' in col), None)
            if accretion_col:
                accretion_days = mis_df[mis_df[accretion_col] > 0]['DATE']
                if not accretion_days.empty:
                    for date in accretion_days:
                        fig.add_vline(
                            x=date, line_width=1, line_dash="dash", line_color="red",
                            row="all", col="all"
                        )
                        fig.add_annotation(
                            x=date,
                            y=1.05,
                            yref="paper",
                            text="Accretion",
                            showarrow=False,
                            font=dict(size=8, color="red"),
                            row=1, col=2
                        )
        
        # Update layout
        fig.update_layout(
            title='Production Quality Metrics',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            margin=dict(l=10, r=10, t=40, b=10),
            height=240,
            template='plotly_dark'
        )
        
        # Update axes
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_yaxes(title_text="Production (tons)", row=1, col=1)
        fig.update_xaxes(title_text="Date", row=1, col=2)
        fig.update_yaxes(title_text="Proportion (%)", range=[0, 100], row=1, col=2)
        
        return fig
        
    except Exception as e:
        print(f"Error updating production quality chart: {e}")
        return empty_chart("Production Quality (Error)")

# Callback to update material consumption chart
@app.callback(
    Output('material-consumption', 'figure'),
    Input('intermediate-data', 'children')
)
def update_material_consumption(json_data):
    """Update material consumption chart"""
    if not json_data:
        return empty_chart("Material Consumption")
    
    try:
        # Parse the JSON data
        data_dict = json.loads(json_data)
        # Convert each JSON to DataFrame and ensure it has proper datetime index
        data = {}
        for key, value in data_dict.items():
            df = pd.read_json(value, orient='split')
            # Ensure index is datetime if it has date-like values
            if not isinstance(df.index, pd.DatetimeIndex) and df.index.dtype == 'object':
                try:
                    df.index = pd.to_datetime(df.index)
                except:
                    pass  # If conversion fails, keep original index
            data[key] = df
        
        if 'mis' not in data or data['mis'].empty:
            return empty_chart("Material Consumption")
        
        mis_df = data['mis']
        
        # Create subplots: materials and ratios
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            subplot_titles=("Material Consumption", "Material Ratios"),
            row_heights=[0.6, 0.4]
        )
        
        # Add materials to top plot
        materials = [
            ('IRON ORE CONSUMPTION', 'Iron Ore', colors['primary']),
            ('GROSS COAL CONSUMPTION', 'Coal', colors['danger']),
            ('DOLO CONSUMPTION', 'Dolomite', colors['warning'])
        ]
        
        for col, name, color in materials:
            if col in mis_df.columns:
                fig.add_trace(go.Scatter(
                    x=mis_df['DATE'],
                    y=mis_df[col],
                    mode='lines',
                    name=name,
                    line=dict(color=color)
                ), row=1, col=1)
        
        # Add ore/coal ratio to bottom plot
        if all(col in mis_df.columns for col, _, _ in materials[:2]):
            ratio = mis_df['IRON ORE CONSUMPTION'] / mis_df['GROSS COAL CONSUMPTION']
            fig.add_trace(go.Scatter(
                x=mis_df['DATE'],
                y=ratio,
                mode='lines+markers',
                name='Ore/Coal Ratio',
                line=dict(color=colors['info'])
            ), row=2, col=1)
        
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Consumption (tons)',
            xaxis2_title='Date',
            yaxis2_title='Ratio',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
            margin=dict(l=10, r=10, t=60, b=10),
            height=300,
            template='plotly_dark'
        )
        
        return fig
        
    except Exception as e:
        print(f"Error updating material consumption chart: {e}")
        return empty_chart("Material Consumption (Error)")

# Callback for Material Quality vs Grades visualization
@app.callback(
    Output('material-quality-grades', 'figure'),
    Input('intermediate-data', 'children')
)
def update_material_quality_grades(json_data):
    """Update the material quality vs grades visualization"""
    if not json_data:
        return empty_chart("Material Quality vs Grades")
    
    try:
        # Parse the JSON data
        data_dict = json.loads(json_data)
        data = {}
        for key, value in data_dict.items():
            df = pd.read_json(value, orient='split')
            # Ensure proper datetime conversion
            if 'DATE' in df.columns:
                df['DATE'] = pd.to_datetime(df['DATE'])
            if 'DATETIME' in df.columns:
                df['DATETIME'] = pd.to_datetime(df['DATETIME'])
            data[key] = df
        
        if 'mis' not in data or data['mis'].empty:
            return empty_chart("Material Quality vs Grades")
        
        mis_df = data['mis'].copy()
        
        # Extract grade columns and material qualities
        grade_cols = [col for col in mis_df.columns if 'GRADE' in col]
        material_cols = [
            col for col in mis_df.columns if any(
                term in col for term in ['IRON ORE', 'COAL', 'FINES', 'LIMESTONE', 'COKE']
            ) and 'QUALITY' in col
        ]
        
        if not grade_cols or not material_cols:
            return empty_chart("Material Quality vs Grades (No Data)")
        
        # Create scatter plots for each grade vs material quality
        fig = make_subplots(rows=1, cols=len(grade_cols), 
                          subplot_titles=[f"{grade}" for grade in grade_cols],
                          shared_yaxes=True)
        
        # Colors for different material types
        color_map = {
            'IRON ORE': '#1f77b4',
            'COAL': '#ff7f0e',
            'FINES': '#2ca02c',
            'LIMESTONE': '#d62728',
            'COKE': '#9467bd'
        }
        
        for i, grade in enumerate(grade_cols):
            col_idx = i + 1  # 1-based indexing for subplots
            
            for mat_col in material_cols:
                # Determine material type for coloring
                material_type = next((mat for mat in color_map.keys() if mat in mat_col), 'OTHER')
                
                # Skip if no data
                if mis_df[grade].isna().all() or mis_df[mat_col].isna().all():
                    continue
                    
                fig.add_trace(
                    go.Scatter(
                        x=mis_df[mat_col],
                        y=mis_df[grade],
                        mode='markers',
                        marker=dict(
                            size=8,
                            color=color_map.get(material_type, '#777777'),
                            opacity=0.7
                        ),
                        name=mat_col,
                        text=mis_df['DATE'].dt.strftime('%Y-%m-%d'),
                        hovertemplate='%{text}<br>' +
                                     f'{mat_col}: %{{x:.2f}}<br>' +
                                     f'{grade}: %{{y:.2f}}<extra></extra>'
                    ),
                    row=1, col=col_idx
                )
        
        # Update layout
        fig.update_layout(
            title='Material Quality Impact on Product Grades',
            height=400,
            template='plotly_dark',
            showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=-0.2),
            margin=dict(l=50, r=20, t=50, b=100)
        )
        
        # Update axes
        for i in range(1, len(grade_cols) + 1):
            fig.update_xaxes(title_text='Material Quality', row=1, col=i)
            if i == 1:
                fig.update_yaxes(title_text='Grade Output (%)', row=1, col=i)
        return fig
    except Exception as e:
        print(f"Error updating material quality vs grades chart: {e}")
        return empty_chart("Material Quality vs Grades (Error)")

# Callback for Material Quality vs Accretion visualization
@app.callback(
    Output('material-quality-accretion', 'figure'),
    Input('intermediate-data', 'children')
)
def update_material_quality_accretion(json_data):
    """Update the material quality vs accretion visualization"""
    if not json_data:
        return empty_chart("Material Quality vs Accretion")
    try:
        # Parse the JSON data
        data_dict = json.loads(json_data)
        data = {}
        for key, value in data_dict.items():
            df = pd.read_json(value, orient='split')
            # Ensure proper datetime conversion
            if 'DATE' in df.columns:
                df['DATE'] = pd.to_datetime(df['DATE'])
            if 'DATETIME' in df.columns:
                df['DATETIME'] = pd.to_datetime(df['DATETIME'])
            data[key] = df
        # We'll need both MIS data for material quality and accretion truth data
        if ('mis' not in data or data['mis'].empty) or ('shell' not in data or data['shell'].empty):
            return empty_chart("Material Quality vs Accretion (No Data)")
        mis_df = data['mis'].copy()
        shell_df = data['shell'].copy()
        # Join the data on DATE
        if 'DATE' not in shell_df.columns and 'DATETIME' in shell_df.columns:
            shell_df['DATE'] = shell_df['DATETIME'].dt.date
        # Extract material quality columns
        material_cols = [
            col for col in mis_df.columns if any(
                term in col for term in ['IRON ORE', 'COAL', 'FINES', 'LIMESTONE', 'COKE']
            ) and 'QUALITY' in col
        ]
        # Extract accretion indicators
        accretion_cols = [col for col in shell_df.columns if 'ZONE_' in col and col.endswith('_ACCRETION')]
        if not material_cols or not accretion_cols:
            return empty_chart("Material Quality vs Accretion (Missing Columns)")
        # Merge the data frames
        merged_df = pd.merge(mis_df, shell_df, on='DATE', how='inner')
        if merged_df.empty:
            return empty_chart("Material Quality vs Accretion (No Matching Dates)")
        # Create an overall accretion severity metric (sum of accretion indicators)
        merged_df['TOTAL_ACCRETION'] = merged_df[accretion_cols].sum(axis=1)
        # Create scatter plot matrix
        fig = make_subplots(rows=1, cols=len(material_cols), 
                          subplot_titles=[col.replace('QUALITY', '').strip() for col in material_cols],
                          shared_yaxes=True)
        # Colors for different accretion intensities
        colorscale = px.colors.sequential.Plasma
        for i, mat_col in enumerate(material_cols):
            col_idx = i + 1  # 1-based indexing for subplots
            # Skip if no data
            if merged_df[mat_col].isna().all() or merged_df['TOTAL_ACCRETION'].isna().all():
                continue
            fig.add_trace(
                go.Scatter(
                    x=merged_df[mat_col],
                    y=merged_df['TOTAL_ACCRETION'],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=merged_df['TOTAL_ACCRETION'],
                        colorscale=colorscale,
                        colorbar=dict(title="Accretion Severity") if i == len(material_cols) - 1 else None,
                        showscale=i == len(material_cols) - 1,  # Only show colorbar on last subplot
                        opacity=0.8
                    ),
                    name=mat_col,
                    text=merged_df['DATE'].dt.strftime('%Y-%m-%d'),
                    hovertemplate='%{text}<br>' +
                                 f'{mat_col}: %{{x:.2f}}<br>' +
                                 'Accretion Severity: %{y:.2f}<extra></extra>'
                ),
                row=1, col=col_idx
            )
        # Update layout
        fig.update_layout(
            title='Impact of Material Quality on Accretion Formation',
            height=400,
            template='plotly_dark',
            showlegend=False,
            margin=dict(l=50, r=20, t=50, b=50)
        )
        # Update axes
        for i in range(1, len(material_cols) + 1):
            fig.update_xaxes(title_text='Material Quality', row=1, col=i)
            if i == 1:  # Only add y-axis title to first subplot
                fig.update_yaxes(title_text='Accretion Severity', row=1, col=i)
        
        return fig
    except Exception as e:
        print(f"Error updating material quality vs accretion chart: {e}")
        return empty_chart("Material Quality vs Accretion (Error)")

# Callback to update incremental learning performance
@app.callback(
    Output('incremental-learning-metrics', 'figure'),
    Input('data-refresh', 'n_intervals')
)
def update_incremental_learning_metrics(n_intervals):
    """Update the incremental learning performance visualization"""
    try:
        # Load models
        models = load_models()
        if 'predictor' not in models:
            return empty_chart("Incremental Learning (No Model)")
        
        predictor = models['predictor']
        
        # Check if the predictor has a get_training_history method
        if not hasattr(predictor, 'get_training_history'):
            return empty_chart("Model does not support training history tracking")
        
        # Get training history
        try:
            history = predictor.get_training_history()
        except Exception as e:
            print(f"Error retrieving training history: {e}")
            return empty_chart(f"Unable to retrieve training history: {str(e)}")
        
        # Check if there's any history data
        if not history or ('binary' not in history and 'regression' not in history):
            return empty_chart("No training history data available")
            
        if (('binary' in history and not history['binary'].get('timestamps')) and 
            ('regression' in history and not history['regression'].get('timestamps'))):
            return empty_chart("No Incremental Learning History")
          # Create figure with secondary y-axis
        fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])
        
        # Add binary classification metrics
        if ('binary' in history and 
            'timestamps' in history['binary'] and 
            'metrics' in history['binary'] and
            history['binary']['timestamps'] and 
            history['binary']['metrics']):
            
            timestamps = history['binary']['timestamps']
            accuracies = [m.get('accuracy', 0) for m in history['binary']['metrics']]
            f1_scores = [m.get('f1', 0) for m in history['binary']['metrics']]
            
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=accuracies,
                    mode='lines+markers',
                    name='Binary Accuracy',
                    line=dict(color=colors['success'])
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=f1_scores,
                    mode='lines+markers',
                    name='Binary F1 Score',
                    line=dict(color=colors['warning'])
                )
            )
        
        # Add regression metrics
        if ('regression' in history and 
            'timestamps' in history['regression'] and 
            'metrics' in history['regression'] and
            history['regression']['timestamps'] and 
            history['regression']['metrics']):
            
            timestamps = history['regression']['timestamps']
            r2_scores = [m.get('r2', 0) for m in history['regression']['metrics']]
            rmse_values = [m.get('rmse', 0) for m in history['regression']['metrics']]
            
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=r2_scores,
                    mode='lines+markers',
                    name='Regression R² Score',
                    line=dict(color=colors['info'])
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=rmse_values,
                    mode='lines+markers',
                    name='RMSE',
                    line=dict(color=colors['danger']),
                ),
                secondary_y=True
            )
        
        # Update layout
        fig.update_layout(
            title='Model Performance Over Incremental Updates',
            xaxis_title='Update Time',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            margin=dict(l=10, r=10, t=40, b=10),
            template='plotly_dark'
        )
        
        # Update y-axes titles
        fig.update_yaxes(title_text="Accuracy / R² Score", secondary_y=False)
        fig.update_yaxes(title_text="RMSE", secondary_y=True)
        
        return fig
        
    except Exception as e:
        print(f"Error updating incremental learning metrics: {e}")
        return empty_chart(f"Incremental Learning Metrics (Error: {str(e)})")

# Callback to handle incremental model updates
@app.callback(
    Output('incremental-update-status', 'children'),
    Input('incremental-update-button', 'n_clicks'),
    State('intermediate-data', 'children'),
    prevent_initial_call=True
)
def update_model_incrementally(n_clicks, json_data):
    """Handle incremental model updates when button is clicked"""
    if not n_clicks or not json_data:
        return ""
    
    try:
        # Parse the JSON data
        data_dict = json.loads(json_data)
        # Convert each JSON to DataFrame
        data = {}
        for key, value in data_dict.items():
            df = pd.read_json(value, orient='split')
            # Ensure proper datetime conversion
            if 'DATE' in df.columns:
                df['DATE'] = pd.to_datetime(df['DATE'])
            if 'DATETIME' in df.columns:
                df['DATETIME'] = pd.to_datetime(df['DATETIME'])
            data[key] = df
            
        # Get most recent data for incremental update
        # We'll use the last 5 days of data
        if 'processed' in data and not data['processed'].empty:
            preprocessed_df = data['processed']
            
            # Get the latest 5 days of data
            cutoff_date = preprocessed_df.index.max() - pd.Timedelta(days=5)
            recent_data = preprocessed_df[preprocessed_df.index >= cutoff_date]
            
            if recent_data.empty:
                return html.Div("No recent data available for update", className="text-warning")
            
            # Extract features and targets
            feature_cols = [col for col in recent_data.columns if not col.startswith('target_')]
            X = recent_data[feature_cols]
            
            y_binary = recent_data['target_binary'] if 'target_binary' in recent_data.columns else None
            y_days = recent_data['target_days'] if 'target_days' in recent_data.columns else None
            y_zone = recent_data['target_zone'] if 'target_zone' in recent_data.columns else None
            
            if y_binary is None:
                return html.Div("Missing target variables in processed data", className="text-warning")
            
            # Load models
            models = load_models()
            if 'predictor' not in models:
                return html.Div("No prediction model available", className="text-danger")
            
            # Perform incremental update
            predictor = models['predictor']
            metrics = predictor.update_incrementally(X, y_binary, y_days, y_zone)
            
            # Save the updated model
            predictor_path = os.path.join(MODEL_DIR, 'predictor')
            predictor.save(predictor_path)
            
            # Format results message
            msg = [html.H6("Model Updated Successfully", className="text-success mb-3")]
            
            for model_type, model_metrics in metrics.items():
                msg.append(html.Div(f"{model_type.capitalize()} Model Metrics:"))
                for metric_name, value in model_metrics.items():
                    msg.append(html.Div(f"- {metric_name}: {value:.4f}"))
                msg.append(html.Br())
            
            return html.Div(msg)
            
        else:
            return html.Div("No processed data available", className="text-warning")
            
    except Exception as e:
        print(f"Error during incremental model update: {e}")
        return html.Div(f"Error: {str(e)}", className="text-danger")

if __name__ == '__main__':
    import sys
    import logging
    
    # Set up logging
    logging.basicConfig(
        filename='dashboard_debug.log',
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Add console handler for immediate visibility
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    
    def exception_handler(exctype, value, traceback):
        logging.error("Uncaught exception", exc_info=(exctype, value, traceback))
    
    # Set the exception handler
    sys.excepthook = exception_handler
    
    try:
        app.run(debug=True, port=8050, use_reloader=False)
    except Exception as e:
        logging.exception("Exception during dashboard execution: %s", str(e))