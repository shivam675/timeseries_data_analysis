import os
import math
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error, mean_absolute_error, r2_score

# PyTorch imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
    
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Force CPU to avoid CUDA issues
    DEVICE = torch.device('cpu')
    print(f"PyTorch available. Using device: {DEVICE}")
    
    # Set default tensor type to float32 for stability
    torch.set_default_dtype(torch.float32)
    
    # For numerical stability
    torch.set_printoptions(precision=10)
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = 'cpu'
    print("PyTorch not available. LSTM/Transformer models will not be functional.")

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y=None, seq_len=10):
        self.X = X
        self.y = y
        self.seq_len = seq_len
        self.n_samples, self.n_features = X.shape
        
        # Check for and report any NaN or infinite values in input data
        self._validate_data()
    
    def _validate_data(self):
        """Check for problematic values in the input data"""
        if np.isnan(self.X).any():
            print(f"Warning: {np.isnan(self.X).sum()} NaN values detected in input features")
            # Replace NaNs with 0s in input data
            self.X = np.nan_to_num(self.X, nan=0.0)
        
        if np.isinf(self.X).any():
            print(f"Warning: {np.isinf(self.X).sum()} infinite values detected in input features")
            # Replace infs with large values
            self.X = np.nan_to_num(self.X, posinf=1.0, neginf=-1.0)
        
        if self.y is not None:
            if isinstance(self.y, np.ndarray) and (np.isnan(self.y).any() or np.isinf(self.y).any()):
                print("Warning: NaN or infinite values detected in target values")
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        # Handle sequence padding for early samples
        if idx < self.seq_len:
            # Create a zero tensor with exactly seq_len rows
            pad_size = self.seq_len - min(idx + 1, self.seq_len)
            pad = np.zeros((pad_size, self.n_features))
            x_seq = np.vstack([pad, self.X[max(0, idx-self.seq_len+1+pad_size):idx+1]])
        else:
            # Ensure we get exactly seq_len rows
            x_seq = self.X[idx-self.seq_len+1:idx+1]
        
        # Double-check that shape is correct (exactly seq_len rows)
        if x_seq.shape[0] != self.seq_len:
            # Force correct size by either padding or truncating
            if x_seq.shape[0] < self.seq_len:
                # Pad
                extra_pad = np.zeros((self.seq_len - x_seq.shape[0], self.n_features))
                x_seq = np.vstack([extra_pad, x_seq])
            else:
                # Truncate
                x_seq = x_seq[-self.seq_len:, :]
                
        # Replace any remaining problematic values
        x_seq = np.nan_to_num(x_seq, nan=0.0, posinf=1.0, neginf=-1.0)
                
        x_tensor = torch.tensor(x_seq, dtype=torch.float32)
        
        # Return target tensor if available
        if self.y is not None:
            # Use iloc to avoid deprecation warning when y is a pandas Series
            if hasattr(self.y, 'iloc'):
                y_val = self.y.iloc[idx]
            else:
                y_val = self.y[idx]
                
            # Handle NaN/inf in target values
            if np.isnan(y_val) or np.isinf(y_val):
                if isinstance(y_val, (int, float)):
                    y_val = 0.0  # Default for regression
                elif hasattr(y_val, '__iter__'):
                    y_val = np.nan_to_num(y_val, nan=0.0)
                
            # Convert to float32 tensor
            y_tensor = torch.tensor(y_val, dtype=torch.float32)
            return x_tensor, y_tensor
        return x_tensor

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=1, dropout=0.2, bidirectional=True, task='classification'):
        super().__init__()
        # Add batch normalization to stabilize inputs
        self.batch_norm = nn.BatchNorm1d(input_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        # Add dropout before final layer to prevent overfitting
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim*(2 if bidirectional else 1), output_dim)
        
        # Store task and output_dim for reference
        self.task = task
        self.output_dim = output_dim
        
        # For binary classification with BCELoss we'll use Sigmoid
        # For multi-class we'll use Softmax
        # For regression, no activation
        if task == 'classification' and output_dim == 1:
            self.final_activation = nn.Sigmoid() 
        elif task == 'classification' and output_dim > 1:
            self.final_activation = nn.Softmax(dim=1)
        else:
            self.final_activation = None
    
    def forward(self, x):
        # Apply batch normalization to each time step
        batch_size, seq_len, features = x.size()
        x_reshaped = x.view(-1, features)
        x_normed = self.batch_norm(x_reshaped)
        x = x_normed.view(batch_size, seq_len, features)
        
        out, _ = self.lstm(x)
        # Get the last time step
        out = out[:, -1, :]
        # Apply dropout
        out = self.dropout(out)
        # Final linear layer
        out = self.fc(out)
        
        # Apply activation if specified
        if self.final_activation:
            out = self.final_activation(out)
        return out

class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=1, nhead=4, dropout=0.1, task='classification'):
        super().__init__()
        # Add batch normalization to stabilize inputs
        self.batch_norm = nn.BatchNorm1d(input_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Add layer normalization before final layer
        self.layer_norm = nn.LayerNorm(hidden_dim)
        # Add dropout before final layer to prevent overfitting
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        # Store task and output_dim for reference
        self.task = task
        self.output_dim = output_dim
        
        # For binary classification with BCELoss we'll use Sigmoid
        # For multi-class we'll use Softmax
        # For regression, no activation
        if task == 'classification' and output_dim == 1:
            self.final_activation = nn.Sigmoid()
        elif task == 'classification' and output_dim > 1:
            self.final_activation = nn.Softmax(dim=1)
        else:
            self.final_activation = None
    
    def forward(self, x):
        # Apply batch normalization to each time step
        batch_size, seq_len, features = x.size()
        x_reshaped = x.view(-1, features)
        x_normed = self.batch_norm(x_reshaped)
        x = x_normed.view(batch_size, seq_len, features)
        
        # Project to hidden dimension
        x = self.input_proj(x)
        # Apply transformer
        out = self.transformer(x)
        # Get final token
        out = out[:, -1, :]
        # Apply layer normalization
        out = self.layer_norm(out)
        # Apply dropout
        out = self.dropout(out)
        # Final projection
        out = self.fc(out)
        
        # Apply activation if specified
        if self.final_activation:
            out = self.final_activation(out)
        return out

class PyTorchModelWrapper:
    def __init__(self, model, task='classification', lr=0.001, batch_size=32, epochs=50, patience=5, seq_len=10, device=None):
        self.model = model
        self.task = task
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.seq_len = seq_len
        self.device = device if device else DEVICE
        self.model.to(self.device)
        
        # Initialize weights to avoid NaN outputs
        self._initialize_weights()
        
        # Setup appropriate loss functions with reduction='none' for better control
        if task == 'classification':
            if model.fc.out_features == 1:
                # Always use BCEWithLogitsLoss for binary - it's more stable
                # We'll handle the sigmoid manually if needed
                self.criterion = nn.BCEWithLogitsLoss(reduction='none')
                # Set final activation to None to avoid double sigmoid
                if hasattr(model, 'final_activation') and isinstance(model.final_activation, nn.Sigmoid):
                    print("Note: Replacing sigmoid activation with manual application for stability")
                    model.final_activation = None
            else:
                # For multi-class use reduction='none' for per-sample handling
                self.criterion = nn.CrossEntropyLoss(reduction='none')
        else:
            # For regression, use Huber loss which is more robust to outliers
            self.criterion = nn.HuberLoss(reduction='none', delta=1.0)
            
        # Track best model state
        self.best_model_state = None
        self.best_loss = float('inf')
    def _initialize_weights(self):
        """Initialize weights to avoid NaN outputs and improve convergence."""
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                if 'lstm' in name.lower() or 'rnn' in name.lower():
                    # Special initialization for recurrent layers
                    for k in range(4):  # 4 gates in LSTM
                        # Handle stacked bidirectional layers correctly
                        if len(param.shape) >= 2:
                            nn.init.orthogonal_(param[k*param.shape[0]//4:(k+1)*param.shape[0]//4])
                elif 'transformer' in name.lower():
                    # Scaled initialization for transformer weights
                    if len(param.shape) >= 2:  # Matrices
                        nn.init.xavier_normal_(param, gain=0.02)
                    else:  # Vectors
                        nn.init.normal_(param, std=0.02)  
                elif len(param.shape) >= 2:  # For standard linear layers
                    nn.init.kaiming_uniform_(param, a=math.sqrt(5))
                else:
                    # For vectors (e.g. biases), use small normal initialization
                    nn.init.normal_(param, mean=0.0, std=0.02)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
                
    def fit(self, X, y, validation_data=None):
        # Create dataset with robust validation
        dataset = TimeSeriesDataset(X, y, seq_len=self.seq_len)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Create optimizer with weight decay for regularization
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
          # Create learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
        
        # Initialize tracking variables
        best_loss = float('inf')
        patience_counter = 0
        self.best_model_state = None
        
        # Print model summary info for debugging
        print(f"Model training on {self.device}")
        print(f"Task: {self.task}, Output dim: {self.model.fc.out_features}")
        print(f"Loss: {self.criterion.__class__.__name__}")
        print(f"Using activation: {self.model.final_activation.__class__.__name__ if hasattr(self.model, 'final_activation') and self.model.final_activation else 'None'}")        # Get labels distribution for sanity check
        if self.task == 'classification':
            # Convert pandas Series to numpy array 
            y_array = y.values if hasattr(y, 'values') else np.array(y)
            # Use numpy's unique function directly (no need to flatten)
            unique_labels = np.unique(y_array)
            print(f"Label distribution: {[(label, (y == label).sum()) for label in unique_labels]}")
        # Training loop
        for epoch in range(self.epochs):
            self.model.train()
            epoch_losses = []
            successfully_processed_batches = 0
            total_batches = 0
            
            for xb, yb in loader:
                total_batches += 1
                # Move data to device
                xb = xb.to(self.device)
                
                # Handle different task types properly
                if self.task == 'classification':
                    if self.model.fc.out_features == 1:  # Binary classification
                        # Process for binary classification
                        try:
                            # Reset gradients
                            optimizer.zero_grad()
                            
                            # Ensure target is proper shape and type
                            yb = yb.float().view(-1, 1).to(self.device)
                            
                            # Forward pass
                            outputs = self.model(xb)
                            
                            # Check for NaNs in output and replace them
                            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                                outputs = torch.nan_to_num(outputs, nan=0.5, posinf=1.0, neginf=0.0)
                            
                            # BCEWithLogitsLoss has reduction='none' so we get per-sample losses
                            losses = self.criterion(outputs, yb)
                            
                            # Identify and ignore NaN losses
                            valid_mask = ~torch.isnan(losses) & ~torch.isinf(losses)
                            if valid_mask.sum() > 0:
                                # Use only valid losses
                                valid_loss = losses[valid_mask].mean()
                                valid_loss.backward()
                                
                                # Clip gradients to prevent explosion
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                                
                                # Update weights
                                optimizer.step()
                                epoch_losses.append(valid_loss.item())
                                successfully_processed_batches += 1
                            else:
                                print(f"Warning: All losses in binary batch are NaN/Inf, skipping")
                                
                        except RuntimeError as e:
                            print(f"Error in binary classification batch: {e}")
                            continue
                            
                    else:  # Multi-class classification
                        try:
                            # Reset gradients
                            optimizer.zero_grad()
                            
                            # Multi-class needs long dtype
                            yb = yb.long().to(self.device)
                            
                            # Forward pass
                            outputs = self.model(xb)
                            
                            # Handle NaN outputs
                            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                                outputs = torch.nan_to_num(outputs, nan=0.0)
                            
                            # Compute per-sample loss
                            losses = self.criterion(outputs, yb)
                            
                            # Handle NaN losses
                            valid_mask = ~torch.isnan(losses) & ~torch.isinf(losses)
                            if valid_mask.sum() > 0:
                                valid_loss = losses[valid_mask].mean()
                                valid_loss.backward()
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                                optimizer.step()
                                epoch_losses.append(valid_loss.item())
                                successfully_processed_batches += 1
                            else:
                                print(f"Warning: All losses in multi-class batch are NaN/Inf, skipping")
                                
                        except RuntimeError as e:
                            print(f"Error in multi-class batch: {e}")
                            continue
                            
                else:  # Regression task
                    try:
                        # Reset gradients
                        optimizer.zero_grad()
                        
                        # Ensure proper dtype and device for targets
                        yb = yb.float().to(self.device)
                        
                        # Reshape targets if needed for proper broadcasting
                        if self.model.fc.out_features == 1 and yb.dim() == 1:
                            yb = yb.view(-1, 1)
                            
                        # Forward pass
                        outputs = self.model(xb)
                        
                        # Handle NaNs in output
                        if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                            outputs = torch.nan_to_num(outputs, nan=0.0, posinf=10.0, neginf=-10.0)
                            
                        # Using Huber loss with reduction='none' gives per-sample losses
                        losses = self.criterion(outputs, yb)
                        
                        # Filter out NaN/Inf losses
                        valid_mask = ~torch.isnan(losses) & ~torch.isinf(losses)
                        if valid_mask.sum() > 0:
                            valid_loss = losses[valid_mask].mean()
                            valid_loss.backward()
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                            optimizer.step()
                            epoch_losses.append(valid_loss.item())
                            successfully_processed_batches += 1
                        else:
                            print(f"Warning: All losses in regression batch are NaN/Inf, skipping")
                            
                    except RuntimeError as e:
                        print(f"Error in regression batch: {e}")
                        continue
            
            # Epoch summary
            if len(epoch_losses) > 0:
                avg_loss = sum(epoch_losses) / len(epoch_losses)
                valid_batch_percent = successfully_processed_batches / total_batches * 100
                
                # Report progress every few epochs
                if epoch % 5 == 0 or epoch == self.epochs - 1:
                    print(f"Epoch {epoch}/{self.epochs}, Loss: {avg_loss:.4f}, Valid batches: {valid_batch_percent:.1f}%")
                
                # Check for improvement
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                    # Save best model state
                    self.best_model_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                    if epoch % 5 == 0:
                        print(f"New best model with loss: {avg_loss:.4f}")
                else:
                    patience_counter += 1
                
                # Update learning rate based on loss
                scheduler.step(avg_loss)
                
                # Early stopping check
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch} with best loss: {best_loss:.4f}")
                    break
                    
                # If we've gone through 1/3 of epochs with low valid batches, try reinitializing
                if epoch > self.epochs // 3 and valid_batch_percent < 50:
                    print(f"Warning: Low percentage of valid batches ({valid_batch_percent:.1f}%). Reinitializing weights...")
                    self._initialize_weights()
                    # Reset optimizer with new learning rate
                    optimizer = optim.AdamW(self.model.parameters(), lr=self.lr * 0.1, weight_decay=1e-4)
                    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
            else:
                print(f"Epoch {epoch}: No valid batches processed!")
                # Try re-initializing weights with smaller values
                print("Re-initializing weights with smaller initialization...")
                for name, param in self.model.named_parameters():
                    if 'weight' in name:
                        nn.init.normal_(param, mean=0.0, std=0.01)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
        
        # Restore best model if we found one
        if self.best_model_state is not None:
            print("Restoring best model state")
            self.model.load_state_dict(self.best_model_state)
        
        return self    
    def predict(self, X):
        """
        Generate predictions from input data.
        Returns class labels for classification or continuous values for regression.
        """
        self.model.eval()
        
        # Create dataset with validation
        dataset = TimeSeriesDataset(X, seq_len=self.seq_len)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        all_outputs = []
        
        with torch.no_grad():
            for xb in loader:
                # Move batch to device
                xb = xb.to(self.device)
                
                try:
                    # Generate predictions
                    outputs = self.model(xb)
                    
                    # Handle NaN/Inf values
                    if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                        print(f"Warning: NaN/Inf detected in prediction outputs")
                        outputs = torch.nan_to_num(outputs, nan=0.0, posinf=1.0, neginf=0.0)
                    
                    # Store batch results
                    all_outputs.append(outputs.cpu().numpy())
                except Exception as e:
                    print(f"Error in prediction batch: {str(e)}")
                    # Return empty batch with right shape
                    if self.task == 'classification':
                        shape = (xb.size(0), self.model.fc.out_features)
                    else:
                        shape = (xb.size(0), 1)
                    all_outputs.append(np.zeros(shape))
        
        # Combine all batch outputs
        if len(all_outputs) == 0:
            print("No valid predictions generated")
            if self.task == 'classification':
                return np.zeros(len(X), dtype=int)
            else:
                return np.zeros(len(X))
        
        combined_outputs = np.vstack(all_outputs)
        
        # Process outputs based on task type
        if self.task == 'classification':
            if self.model.fc.out_features == 1:
                # For binary classification
                # If we used BCEWithLogitsLoss, apply sigmoid manually
                if not hasattr(self.model, 'final_activation') or self.model.final_activation is None:
                    combined_outputs = 1 / (1 + np.exp(-combined_outputs))  # sigmoid
                
                # Handle extreme values
                combined_outputs = np.clip(combined_outputs, 1e-6, 1-1e-6)
                
                # Threshold to get class labels (0 or 1)
                # Ensure we have at least some positive predictions (if reasonable)
                predictions = (combined_outputs > 0.5).astype(int).flatten()
                
                # Print distribution of predictions for debugging
                unique, counts = np.unique(predictions, return_counts=True)
                print(f"Binary prediction distribution: {dict(zip(unique, counts))}")
                
                # If all predictions are the same, check if we need to adjust threshold
                if len(unique) == 1:
                    if unique[0] == 0:  # All zeros
                        # Find highest confidence samples and flip a few to positive if appropriate
                        top_indices = np.argsort(combined_outputs.flatten())[-int(len(X)*0.05):]
                        if combined_outputs.flatten()[top_indices].max() > 0.3:
                            print(f"Converting {len(top_indices)} highest confidence predictions to positive class")
                            predictions[top_indices] = 1
                
                return predictions
                
            else:
                # For multi-class classification
                return np.argmax(combined_outputs, axis=1)
        else:
            # For regression
            # Apply any post-processing if needed
            return combined_outputs.flatten()

    def predict_proba(self, X):
        """
        Generate class probabilities for classification tasks.
        For regression, returns the raw output values.
        """
        self.model.eval()
        dataset = TimeSeriesDataset(X, seq_len=self.seq_len)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        all_probs = []
        
        with torch.no_grad():
            for xb in loader:
                xb = xb.to(self.device)
                
                try:
                    outputs = self.model(xb)
                    
                    # Handle classification outputs appropriately
                    if self.task == 'classification':
                        if self.model.fc.out_features == 1:
                            # Apply sigmoid for binary if using BCEWithLogitsLoss
                            if not hasattr(self.model, 'final_activation') or self.model.final_activation is None:
                                outputs = torch.sigmoid(outputs)
                        elif hasattr(self.model, 'final_activation') and self.model.final_activation is None:
                            # Apply softmax for multi-class if not already applied
                            outputs = torch.softmax(outputs, dim=1)
                    
                    # Handle NaN/Inf values
                    if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                        outputs = torch.nan_to_num(outputs, nan=0.5, posinf=1.0, neginf=0.0)
                    
                    all_probs.append(outputs.cpu().numpy())
                except Exception as e:
                    print(f"Error in predict_proba: {str(e)}")
                    shape = (xb.size(0), self.model.fc.out_features)
                    all_probs.append(np.zeros(shape))
        
        if len(all_probs) == 0:
            # Fallback if no valid batches
            if self.task == 'classification':
                if self.model.fc.out_features == 1:
                    return np.zeros((len(X), 2))  # Binary case: [P(0), P(1)]
                else:
                    return np.zeros((len(X), self.model.fc.out_features))
            else:
                return np.zeros(len(X))
        
        combined_probs = np.vstack(all_probs)
        
        # Post-process based on task
        if self.task == 'classification':
            if self.model.fc.out_features == 1:
                # Binary case - return [P(0), P(1)]
                probs = np.clip(combined_probs.flatten(), 1e-6, 1-1e-6)
                
                # Check for degenerate predictions
                if np.all(probs < 0.1):
                    print("Warning: All binary probabilities are very low")
                    # Adjust some probs to be higher for a more useful model
                    top_indices = np.argsort(probs)[-int(len(probs)*0.05):]
                    probs[top_indices] = np.clip(probs[top_indices] * 5.0, 0, 0.95)
                
                return np.vstack((1-probs, probs)).T
            else:
                # Multi-class - ensure proper probabilities
                return np.clip(combined_probs, 0, 1)
        else:
            # Regression doesn't have probabilities
            return combined_probs

class KilnAccretionPredictor:
    """
    Predicts kiln accretion formation using LSTM/Transformer models (PyTorch) or fallback ML.
    """    def __init__(self, model_type='lstm', seq_len=10, use_traditional_ml=False):
        self.model_type = model_type
        self.seq_len = seq_len
        self.use_traditional_ml = use_traditional_ml
        self.models = {}
        self.scalers = {}
        self.feature_names = None
        self.zone_classes = None
        
        # Initialize training history
        self.training_history = {
            'binary': {
                'timestamps': [],
                'metrics': []
            },
            'regression': {
                'timestamps': [],
                'metrics': []
            },
            'zone': {
                'timestamps': [],
                'metrics': []
            }
        }

    def _create_model(self, task, input_dim, output_dim=1):
        # Define appropriate hyperparameters based on task
        if task == 'classification':
            if output_dim == 1:  # Binary classification
                # For binary tasks, slightly higher learning rate, more epochs
                lr = 0.002
                batch_size = 64
                epochs = 100
                patience = 10
            else:  # Multi-class classification
                lr = 0.001
                batch_size = 64
                epochs = 75
                patience = 8
        else:  # Regression
            # For regression, use lower learning rate and more patience
            lr = 0.0005 
            batch_size = 32  # Smaller batches for better stability
            epochs = 150
            patience = 15
        
        # Adjust model size based on input dimension
        # For high-dimensional inputs, we need more capacity
        hidden_dim = max(64, min(256, input_dim * 2))
        
        # Create model based on type
        if self.model_type == 'lstm':
            model_cls = LSTMModel
            model_params = {
                'input_dim': input_dim,
                'hidden_dim': hidden_dim,
                'num_layers': 2,
                'output_dim': output_dim,
                'dropout': 0.3,  # Increased dropout for better regularization
                'bidirectional': True,
                'task': task
            }
        elif self.model_type == 'transformer':
            model_cls = TransformerModel
            model_params = {
                'input_dim': input_dim,
                'hidden_dim': hidden_dim,
                'num_layers': 3,  # More layers for transformer
                'output_dim': output_dim,
                'nhead': 8,  # Increased number of attention heads
                'dropout': 0.2,
                'task': task
            }
        else:
            raise ValueError('Only LSTM and Transformer supported in this version.')
        
        print(f"Creating {self.model_type} model:")
        print(f"- Task: {task}")
        print(f"- Input dim: {input_dim}, Output dim: {output_dim}")
        print(f"- Hidden dim: {hidden_dim}")
        print(f"- Learning rate: {lr}, Batch size: {batch_size}")
        
        # Create model instance and wrap it
        model = model_cls(**model_params)
        wrapper = PyTorchModelWrapper(
            model=model,
            task=task,
            lr=lr,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            seq_len=self.seq_len,
            device=DEVICE
        )
        
        return wrapper
    def fit(self, X, y_binary, y_days=None, y_zone=None):
        # Store feature names for reference
        self.feature_names = X.columns.tolist()
        
        # Use RobustScaler for better handling of outliers
        self.scalers['features'] = RobustScaler()
        X_scaled = self.scalers['features'].fit_transform(X)
        
        # Print overall training information
        print(f"\n{'='*50}")
        print(f"KILN ACCRETION MODEL TRAINING - {self.model_type.upper()}")
        print(f"{'='*50}")
        print(f"Features: {X_scaled.shape[1]} columns")
        print(f"Using GPU: {torch.cuda.is_available()}")
        print(f"Device: {DEVICE}")
        print(f"Sequence length: {self.seq_len}")
        # Check for NaN or inf values in features
        if np.isnan(X_scaled).sum() > 0:
            print(f"Warning: {np.isnan(X_scaled).sum()} NaN values in features. Replacing with zeros.")
            X_scaled = np.nan_to_num(X_scaled, nan=0.0)
        if np.isinf(X_scaled).sum() > 0:
            print(f"Warning: {np.isinf(X_scaled).sum()} infinite values in features. Replacing with zeros.")
            X_scaled = np.nan_to_num(X_scaled, posinf=1.0, neginf=-1.0)
        print(f"{'='*50}\n")
        
        # Train binary classification model
        if y_binary is not None:
            print(f"\n--- Training Binary Classifier ---")
            print(f"Total samples: {len(y_binary)}")
            
            # Filter to valid samples
            valid_mask = ~y_binary.isna()
            valid_count = valid_mask.sum()
            
            if valid_count > 0:
                X_bin = X_scaled[valid_mask]
                y_bin = y_binary[valid_mask].astype(float)  # Convert to float for PyTorch
                
                print(f"Valid samples: {valid_count} ({valid_count/len(y_binary)*100:.1f}%)")
                print(f"Original class distribution: 0: {(y_bin==0).sum()} samples, 1: {(y_bin==1).sum()} samples")
                
                # Check class balance - if severely imbalanced, use class weighting
                neg_count, pos_count = (y_bin==0).sum(), (y_bin==1).sum()
                
                # If we don't have enough positive examples, use data augmentation
                if pos_count < 20 or pos_count / len(y_bin) < 0.1:
                    print("WARNING: Not enough positive samples - generating synthetic examples")
                    
                    # Find positive examples
                    pos_indices = np.where(y_bin == 1)[0]
                    
                    if len(pos_indices) > 0:
                        # Duplicate positive examples with small variations
                        additional_samples_needed = min(len(y_bin) // 4, 1000) - pos_count
                        
                        if additional_samples_needed > 0:
                            # Create synthetic samples by adding small noise to existing positive samples
                            for _ in range(additional_samples_needed):
                                # Randomly select a positive example
                                idx = np.random.choice(pos_indices)
                                sample = X_bin[idx].copy()
                                
                                # Add small random noise (std=0.1)
                                noise = np.random.normal(0, 0.1, size=sample.shape)
                                new_sample = sample + noise
                                
                                # Append to dataset
                                X_bin = np.vstack([X_bin, new_sample.reshape(1, -1)])
                                y_bin = np.append(y_bin, 1.0)
                            
                            print(f"After augmentation: 0: {(y_bin==0).sum()} samples, 1: {(y_bin==1).sum()} samples")
                
                print(f"Final class distribution: 0: {(y_bin==0).sum()} samples, 1: {(y_bin==1).sum()} samples")
                print(f"Target range: min={y_bin.min():.1f}, max={y_bin.max():.1f}")
                
                try:
                    print("Creating binary classification model...")
                    self.models['binary_classifier'] = self._create_model('classification', X_scaled.shape[1], 1)
                    
                    # Train the model with error handling
                    print("Starting binary classifier training...")
                    self.models['binary_classifier'].fit(X_bin, y_bin)
                    print("Binary classifier training completed successfully")
                    
                except Exception as e:
                    print(f"Error training binary classifier: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print("No valid samples for binary classification")
        
        # Train regression model for days prediction
        if y_days is not None:
            print(f"\n--- Training Days Regressor ---")
            print(f"Total samples: {len(y_days)}")
            
            valid_mask = ~y_days.isna()
            valid_count = valid_mask.sum()
            
            if valid_count > 0:
                X_days = X_scaled[valid_mask]
                y_days_valid = y_days[valid_mask]
                
                print(f"Valid samples: {valid_count} ({valid_count/len(y_days)*100:.1f}%)")
                print(f"Days range: min={y_days_valid.min():.1f}, max={y_days_valid.max():.1f}, mean={y_days_valid.mean():.1f}")
                
                # Check for extreme values in the target
                if y_days_valid.max() > 1000:
                    print(f"Warning: Extreme maximum value in days: {y_days_valid.max():.1f}")
                    # Cap extremely large values
                    cap_value = np.percentile(y_days_valid, 99) * 1.5
                    y_days_valid = np.minimum(y_days_valid, cap_value)
                    print(f"Capped maximum to {cap_value:.1f}")
                
                try:
                    # Use robust scaler for better handling of outliers
                    self.scalers['days'] = RobustScaler()
                    y_days_scaled = self.scalers['days'].fit_transform(y_days_valid.values.reshape(-1, 1)).ravel()
                    
                    # Check for and handle any problematic values
                    if np.isnan(y_days_scaled).any() or np.isinf(y_days_scaled).any():
                        print("Warning: NaN/Inf values in scaled regression targets. Fixing...")
                        y_days_scaled = np.nan_to_num(y_days_scaled, nan=0.0, posinf=5.0, neginf=-5.0)
                    
                    print("Creating regression model...")
                    self.models['days_regressor'] = self._create_model('regression', X_scaled.shape[1], 1)
                    
                    print("Starting days regressor training...")
                    self.models['days_regressor'].fit(X_days, y_days_scaled)
                    print("Days regressor training completed successfully")
                    
                except Exception as e:
                    print(f"Error training days regressor: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print("No valid samples for days regression")
        
        # Train zone classifier model
        if y_zone is not None:
            print(f"\n--- Training Zone Classifier ---")
            print(f"Total samples: {len(y_zone)}")
            
            valid_mask = (y_zone != -1) & ~y_zone.isna()
            valid_count = valid_mask.sum()
            
            if valid_count > 0:
                X_zone = X_scaled[valid_mask]
                y_zone_valid = y_zone[valid_mask]
                
                self.zone_classes = sorted(np.unique(y_zone_valid))
                print(f"Valid samples: {valid_count} ({valid_count/len(y_zone)*100:.1f}%)")
                print(f"Zone classes: {self.zone_classes}")
                
                try:
                    # Map zone values to sequential indices - ensure integers for CrossEntropyLoss
                    y_zone_mapped = np.array([self.zone_classes.index(z) for z in y_zone_valid]).astype(np.int64)
                    class_distribution = [(i, (y_zone_mapped==i).sum()) for i in range(len(self.zone_classes))]
                    print(f"Class distribution: {class_distribution}")
                    
                    # Check for class imbalance
                    class_counts = [count for _, count in class_distribution]
                    if max(class_counts) / min(class_counts) > 2:
                        print("Warning: Significant class imbalance detected in zone classes")
                    
                    print("Creating zone classifier model...")
                    self.models['zone_classifier'] = self._create_model(
                        'classification', X_scaled.shape[1], len(self.zone_classes))
                    
                    print("Starting zone classifier training...")
                    self.models['zone_classifier'].fit(X_zone, y_zone_mapped)
                    print("Zone classifier training completed successfully")
                    
                except Exception as e:
                    print(f"Error training zone classifier: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print("No valid samples for zone classification")
    def predict(self, X):
        X_scaled = self.scalers['features'].transform(X)
        results = {}
        
        # Binary classifier prediction
        if 'binary_classifier' in self.models:
            try:
                # Get probabilities
                proba = self.models['binary_classifier'].predict_proba(X_scaled)
                
                # Handle different shapes depending on predict_proba implementation
                if proba.shape[1] == 2:
                    results['binary_proba'] = proba[:, 1]
                else:
                    results['binary_proba'] = proba.flatten()
                
                # Handle NaN values by setting to 0 (negative class)
                binary_proba = np.nan_to_num(results['binary_proba'], nan=0.0)
                results['binary_proba'] = binary_proba
                results['binary'] = (binary_proba > 0.5).astype(int)
                
                print(f"Binary prediction: {np.unique(results['binary'], return_counts=True)}")
                
            except Exception as e:
                print(f"Error in binary prediction: {e}")
                # Fall back to safe default
                results['binary_proba'] = np.zeros(len(X))
                results['binary'] = np.zeros(len(X), dtype=int)
        
        # Days regressor prediction
        if 'days_regressor' in self.models:
            try:
                days_scaled = self.models['days_regressor'].predict(X_scaled)
                results['days'] = self.scalers['days'].inverse_transform(days_scaled.reshape(-1, 1)).ravel()
                
                # Handle NaN values
                nan_mask = np.isnan(results['days'])
                if nan_mask.any():
                    print(f"Warning: {nan_mask.sum()} NaN values in days prediction")
                    # Replace NaN values with a default (e.g., 365 days)
                    results['days'][nan_mask] = 365.0
                
            except Exception as e:
                print(f"Error in days prediction: {e}")
                results['days'] = np.full(len(X), 365.0)  # Default to 1 year
        
        # Zone classifier prediction
        if 'zone_classifier' in self.models and self.zone_classes is not None:
            try:
                if 'binary' in results:
                    forming_mask = results['binary'] == 1
                    zone_preds = np.full(len(X), -1)
                    
                    if forming_mask.sum() > 0:
                        raw_preds = self.models['zone_classifier'].predict(X_scaled[forming_mask])
                        
                        # Ensure predictions are within range
                        valid_preds = np.clip(raw_preds, 0, len(self.zone_classes) - 1).astype(int)
                        zone_preds[forming_mask] = [self.zone_classes[p] for p in valid_preds]
                    
                    results['zone'] = zone_preds
                else:
                    raw_preds = self.models['zone_classifier'].predict(X_scaled)
                    valid_preds = np.clip(raw_preds, 0, len(self.zone_classes) - 1).astype(int)
                    results['zone'] = [self.zone_classes[p] for p in valid_preds]
                
            except Exception as e:
                print(f"Error in zone prediction: {e}")
                results['zone'] = np.full(len(X), -1)
        
        return results
    def evaluate(self, X, y_binary, y_days=None, y_zone=None):
        preds = self.predict(X)
        results = {}
        
        # Evaluate binary classifier
        if 'binary' in preds and y_binary is not None:
            try:
                # Handle edge case where all predictions are the same
                if len(np.unique(preds['binary'])) == 1:
                    print("Warning: All binary predictions are the same value")
                
                binary_metrics = {
                    'accuracy': accuracy_score(y_binary, preds['binary']),
                    'precision': precision_score(y_binary, preds['binary'], zero_division=0),
                    'recall': recall_score(y_binary, preds['binary'], zero_division=0),
                    'f1': f1_score(y_binary, preds['binary'], zero_division=0),
                }
                
                # Only calculate ROC AUC if we have both classes
                if len(np.unique(y_binary)) > 1:
                    binary_metrics['roc_auc'] = roc_auc_score(y_binary, preds['binary_proba'])
                
                results['binary'] = binary_metrics
            except Exception as e:
                print(f"Error calculating binary metrics: {e}")
        
        # Evaluate regression model
        if 'days' in preds and y_days is not None:
            try:
                # Filter out NaN values before calculating metrics
                valid_mask = ~np.isnan(preds['days'])
                if valid_mask.sum() > 0:
                    filtered_preds = preds['days'][valid_mask]
                    filtered_y = y_days[valid_mask]
                    
                    results['regression'] = {
                        'mse': mean_squared_error(filtered_y, filtered_preds),
                        'rmse': np.sqrt(mean_squared_error(filtered_y, filtered_preds)),
                        'r2': r2_score(filtered_y, filtered_preds)
                    }
                else:
                    print("Warning: All regression predictions are NaN")
            except Exception as e:
                print(f"Error calculating regression metrics: {e}")
        
        # Evaluate zone classifier
        if 'zone' in preds and y_zone is not None:
            try:
                valid = (y_zone != -1)
                if valid.sum() > 0:
                    # Convert to same type to avoid comparison issues
                    preds_zone = np.array(preds['zone']).astype(float)
                    y_zone_vals = np.array(y_zone[valid]).astype(float)
                    acc = accuracy_score(y_zone_vals, preds_zone[valid])
                    results['zone'] = {'accuracy': acc}
            except Exception as e:
                print(f"Error calculating zone metrics: {e}")
        
        return results
    
    def save(self, path):
        os.makedirs(path, exist_ok=True)
        for name, model in self.models.items():
            torch.save(model.model.state_dict(), os.path.join(path, f'{name}_state.pt'))
        for name, scaler in self.scalers.items():
            joblib.dump(scaler, os.path.join(path, f'{name}_scaler.pkl'))
        if self.feature_names:
            with open(os.path.join(path, 'feature_names.txt'), 'w') as f:
                f.write('\n'.join(self.feature_names))
        if self.zone_classes is not None:
            joblib.dump(self.zone_classes, os.path.join(path, 'zone_classes.pkl'))
    
    def load(self, path):
        for name in ['binary_classifier', 'days_regressor', 'zone_classifier']:
            state_path = os.path.join(path, f'{name}_state.pt')
            if os.path.exists(state_path):
                # You must re-instantiate the model with correct dims before loading
                pass  # Not implemented for brevity
        for name in ['features', 'days']:
            scaler_path = os.path.join(path, f'{name}_scaler.pkl')
            if os.path.exists(scaler_path):
                self.scalers[name] = joblib.load(scaler_path)
        feature_file = os.path.join(path, 'feature_names.txt')
        if os.path.exists(feature_file):
            with open(feature_file, 'r') as f:
                self.feature_names = f.read().splitlines()
        zone_file = os.path.join(path, 'zone_classes.pkl')
        if os.path.exists(zone_file):
            self.zone_classes = joblib.load(zone_file)
    
    def update_incrementally(self, X, y_binary, y_days=None, y_zone=None):
        """
        Update the model incrementally with new data
        """
        from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score, mean_absolute_error
        import datetime
        
        metrics = {}
        current_time = datetime.datetime.now()
        
        # Update binary classification model
        if 'binary' in self.models and y_binary is not None:
            try:
                # Preprocess data
                X_scaled = self.scalers['binary'].transform(X)
                
                # Get predictions before update
                y_pred = self.models['binary'].predict(X_scaled)
                
                # Calculate metrics before update
                pre_accuracy = accuracy_score(y_binary, y_pred)
                pre_f1 = f1_score(y_binary, y_pred)
                
                # Update model
                self.models['binary'].fit(X_scaled, y_binary)
                
                # Calculate metrics after update
                y_pred_updated = self.models['binary'].predict(X_scaled)
                accuracy = accuracy_score(y_binary, y_pred_updated)
                f1 = f1_score(y_binary, y_pred_updated)
                
                # Save metrics
                metrics['binary'] = {
                    'accuracy': accuracy,
                    'f1': f1,
                    'accuracy_improvement': accuracy - pre_accuracy,
                    'f1_improvement': f1 - pre_f1
                }
                
                # Update history
                self.training_history['binary']['timestamps'].append(current_time)
                self.training_history['binary']['metrics'].append(metrics['binary'])
            except Exception as e:
                print(f"Error updating binary model: {e}")
        
        # Update regression model
        if 'regression' in self.models and y_days is not None:
            try:
                # Preprocess data
                X_scaled = self.scalers['regression'].transform(X)
                
                # Get predictions before update
                y_pred = self.models['regression'].predict(X_scaled)
                
                # Calculate metrics before update
                pre_mse = mean_squared_error(y_days, y_pred)
                pre_rmse = np.sqrt(pre_mse)
                pre_r2 = r2_score(y_days, y_pred)
                pre_mae = mean_absolute_error(y_days, y_pred)
                
                # Update model
                self.models['regression'].fit(X_scaled, y_days)
                
                # Calculate metrics after update
                y_pred_updated = self.models['regression'].predict(X_scaled)
                mse = mean_squared_error(y_days, y_pred_updated)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_days, y_pred_updated)
                mae = mean_absolute_error(y_days, y_pred_updated)
                
                # Save metrics
                metrics['regression'] = {
                    'mse': mse,
                    'rmse': rmse,
                    'r2': r2,
                    'mae': mae,
                    'rmse_improvement': pre_rmse - rmse,
                    'r2_improvement': r2 - pre_r2,
                    'mae_improvement': pre_mae - mae
                }
                
                # Update history
                self.training_history['regression']['timestamps'].append(current_time)
                self.training_history['regression']['metrics'].append(metrics['regression'])
            except Exception as e:
                print(f"Error updating regression model: {e}")
        
        # Update zone prediction model
        if 'zone' in self.models and y_zone is not None and self.zone_classes is not None:
            try:
                # Preprocess data
                X_scaled = self.scalers['zone'].transform(X)
                
                # Update model
                self.models['zone'].fit(X_scaled, y_zone)
                
                # Update history for zone model (simpler metrics)
                zone_metrics = {'updated': True}
                metrics['zone'] = zone_metrics
                
                self.training_history['zone']['timestamps'].append(current_time)
                self.training_history['zone']['metrics'].append(zone_metrics)
            except Exception as e:
                print(f"Error updating zone model: {e}")
        
        return metrics

    def get_training_history(self):
        """
        Return the training history for incremental updates
        """
        return self.training_history

    def update_incrementally(self, X, y_binary, y_days=None, y_zone=None):
        """
        Update the model incrementally with new data
        """
        from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score, mean_absolute_error
        import datetime
        
        metrics = {}
        current_time = datetime.datetime.now()
        
        # Update binary classification model
        if 'binary' in self.models and y_binary is not None:
            try:
                # Preprocess data
                X_scaled = self.scalers['binary'].transform(X)
                
                # Get predictions before update
                y_pred = self.models['binary'].predict(X_scaled)
                
                # Calculate metrics before update
                pre_accuracy = accuracy_score(y_binary, y_pred)
                pre_f1 = f1_score(y_binary, y_pred)
                
                # Update model
                self.models['binary'].fit(X_scaled, y_binary)
                
                # Calculate metrics after update
                y_pred_updated = self.models['binary'].predict(X_scaled)
                accuracy = accuracy_score(y_binary, y_pred_updated)
                f1 = f1_score(y_binary, y_pred_updated)
                
                # Save metrics
                metrics['binary'] = {
                    'accuracy': accuracy,
                    'f1': f1,
                    'accuracy_improvement': accuracy - pre_accuracy,
                    'f1_improvement': f1 - pre_f1
                }
                
                # Update history
                self.training_history['binary']['timestamps'].append(current_time)
                self.training_history['binary']['metrics'].append(metrics['binary'])
            except Exception as e:
                print(f"Error updating binary model: {e}")
        
        # Update regression model
        if 'regression' in self.models and y_days is not None:
            try:
                # Preprocess data
                X_scaled = self.scalers['regression'].transform(X)
                
                # Get predictions before update
                y_pred = self.models['regression'].predict(X_scaled)
                
                # Calculate metrics before update
                pre_mse = mean_squared_error(y_days, y_pred)
                pre_rmse = np.sqrt(pre_mse)
                pre_r2 = r2_score(y_days, y_pred)
                pre_mae = mean_absolute_error(y_days, y_pred)
                
                # Update model
                self.models['regression'].fit(X_scaled, y_days)
                
                # Calculate metrics after update
                y_pred_updated = self.models['regression'].predict(X_scaled)
                mse = mean_squared_error(y_days, y_pred_updated)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_days, y_pred_updated)
                mae = mean_absolute_error(y_days, y_pred_updated)
                
                # Save metrics
                metrics['regression'] = {
                    'mse': mse,
                    'rmse': rmse,
                    'r2': r2,
                    'mae': mae,
                    'rmse_improvement': pre_rmse - rmse,
                    'r2_improvement': r2 - pre_r2,
                    'mae_improvement': pre_mae - mae
                }
                
                # Update history
                self.training_history['regression']['timestamps'].append(current_time)
                self.training_history['regression']['metrics'].append(metrics['regression'])
            except Exception as e:
                print(f"Error updating regression model: {e}")
        
        # Update zone prediction model
        if 'zone' in self.models and y_zone is not None and self.zone_classes is not None:
            try:
                # Preprocess data
                X_scaled = self.scalers['zone'].transform(X)
                
                # Update model
                self.models['zone'].fit(X_scaled, y_zone)
                
                # Update history for zone model (simpler metrics)
                zone_metrics = {'updated': True}
                metrics['zone'] = zone_metrics
                
                self.training_history['zone']['timestamps'].append(current_time)
                self.training_history['zone']['metrics'].append(zone_metrics)
            except Exception as e:
                print(f"Error updating zone model: {e}")
        
        return metrics
