import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle
import matplotlib.patheffects as path_effects

def create_system_architecture():
    """Create system architecture diagram"""
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Define colors
    colors = {
        'data': '#3498db',
        'processing': '#2ecc71',
        'model': '#9b59b6',
        'dashboard': '#e74c3c',
        'io': '#f39c12',
        'background': '#f5f5f5',
        'text': '#2c3e50'
    }
    
    # Set background color
    ax.set_facecolor(colors['background'])
    fig.patch.set_facecolor(colors['background'])
    
    # Function to add a box with text
    def add_box(x, y, width, height, label, color, alpha=0.7):
        box = Rectangle((x, y), width, height, facecolor=color, alpha=alpha, edgecolor='black', linewidth=1)
        ax.add_patch(box)
        text = ax.text(x + width/2, y + height/2, label, ha='center', va='center', color=colors['text'],
                     fontsize=10, fontweight='bold', wrap=True)
        text.set_path_effects([path_effects.withStroke(linewidth=2, foreground='white')])
        return box
    
    # Function to add an arrow
    def add_arrow(start, end, label=None, color='black'):
        arrow = FancyArrowPatch(start, end, color=color, arrowstyle='->', linewidth=1.5, connectionstyle='arc3,rad=0.1')
        ax.add_patch(arrow)
        if label:
            midpoint = ((start[0] + end[0])/2, (start[1] + end[1])/2)
            ax.text(midpoint[0], midpoint[1], label, ha='center', va='center', fontsize=8,
                   color=colors['text'], backgroundcolor=colors['background'])
        return arrow
    
    # External data sources
    add_box(1, 9, 2, 0.8, "Kiln\nSensors", colors['io'])
    add_box(1, 8, 2, 0.8, "MIS\nReports", colors['io'])
    add_box(1, 7, 2, 0.8, "QRT\nMeasurements", colors['io'])
    
    # Data collection
    data_collection = add_box(4, 8, 2, 1.5, "Data\nCollection\nSystem", colors['data'])
    
    # Data storage
    data_store = add_box(7, 8, 2, 1.5, "Raw Data\nStorage", colors['data'])
    
    # Preprocessor
    preprocessor = add_box(7, 6, 2, 1.5, "Data\nPreprocessor", colors['processing'])
    
    # Feature engineering
    feature_eng = add_box(7, 4, 2, 1.5, "Feature\nEngineering", colors['processing'])
    
    # Models
    predict_model = add_box(4, 2, 2, 1.5, "Prediction\nModel", colors['model'])
    prescribe_model = add_box(7, 2, 2, 1.5, "Prescription\nModel", colors['model'])
    
    # Dashboard and visualization
    dashboard = add_box(10, 4, 2, 1.5, "Real-time\nDashboard", colors['dashboard'])
    viz_3d = add_box(10, 2, 2, 1.5, "3D Kiln\nVisualization", colors['dashboard'])
    
    # Operators and actions
    operators = add_box(13, 3, 2, 1.5, "Kiln\nOperators", colors['io'])
    
    # Model training components
    model_training = add_box(4, 4, 2, 1.5, "Model\nTraining\nPipeline", colors['model'])
    mlflow = add_box(4, 6, 2, 0.8, "MLflow\nTracking", colors['model'])
    
    # Draw connections
    # Data collection flows
    add_arrow((3, 9), (4, 8.75), "Sensor data")
    add_arrow((3, 8.4), (4, 8.5), "MIS data")
    add_arrow((3, 7.4), (4, 8.25), "QRT data")
    
    # Data pipeline
    add_arrow((6, 8.5), (7, 8.5), "Raw data")
    add_arrow((8, 7.5), (8, 6.5), "")
    add_arrow((8, 5.5), (8, 4.5), "")
    
    # Model training
    add_arrow((7, 4.75), (6, 4.75), "Training\ndata")
    add_arrow((5, 4), (5, 2.75), "")
    add_arrow((6, 4.75), (7, 2.75), "Training\ndata")
    add_arrow((5, 6), (5, 5.5), "")
    
    # Dashboard flows
    add_arrow((8, 4.75), (10, 4.75), "Features")
    add_arrow((6, 2.5), (10, 2.5), "Predictions")
    add_arrow((9, 2.5), (10, 3.5), "Prescriptions")
    
    # Operator interaction
    add_arrow((12, 4.5), (13, 4), "Alerts")
    add_arrow((12, 2.5), (13, 3), "Recommendations")
    add_arrow((13, 3.75), (11, 7), "Parameter\nAdjustments", color='red')
    
    # Set axis limits and remove ticks
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add title
    fig.suptitle('Kiln Accretion Monitoring and Prevention System Architecture', fontsize=16, fontweight='bold')
    
    # Add legend
    handles = [
        Rectangle((0, 0), 1, 1, color=colors['data']),
        Rectangle((0, 0), 1, 1, color=colors['processing']),
        Rectangle((0, 0), 1, 1, color=colors['model']),
        Rectangle((0, 0), 1, 1, color=colors['dashboard']),
        Rectangle((0, 0), 1, 1, color=colors['io'])
    ]
    labels = ['Data Components', 'Processing Components', 'Model Components', 'User Interface', 'External Systems']
    ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=5)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('system_architecture.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("System architecture diagram created successfully.")

if __name__ == "__main__":
    create_system_architecture()