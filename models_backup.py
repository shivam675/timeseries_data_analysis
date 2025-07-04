import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import xgboost as xgb
import lightgbm as lgbm
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                            roc_auc_score, mean_squared_error, r2_score)

class KilnAccretionPredictor:
    """
    Predicts kiln accretion formation using multiple models:    1. Binary classification: Is accretion forming?
    2. Regression: Days until critical accretion    
    3. Zone prediction: Which zone will have accretion?
    """
    
    def __init__(self, model_type='xgb'):
        self.model_type = model_type
        self.models = {}
        self.scalers = {}
        self.feature_names = None
        self.zone_classes = None  # Will store the actual zone classes available in training data
        self.zone_class_mapper = None  # Maps original zone values to sequential indices
        self.zone_class_reverse_mapper = None  # Maps sequential indices back to original zone values
        self.training_history = {
            'binary': {'timestamps': [], 'samples': [], 'metrics': []},
            'regression': {'timestamps': [], 'samples': [], 'metrics': []},
            'zone': {'timestamps': [], 'samples': [], 'metrics': []}
        }  # Track incremental learning history
    
    def _create_model(self, task):
        """Create model based on selected algorithm"""
        if task == 'binary':
            if self.model_type == 'rf':
                return RandomForestClassifier(n_estimators=100, random_state=42)
            elif self.model_type == 'xgb':
                return xgb.XGBClassifier(n_estimators=100, random_state=42)
            elif self.model_type == 'lgbm':
                return lgbm.LGBMClassifier(n_estimators=100, random_state=42)
            elif self.model_type == 'lstm':
                # Placeholder for LSTM implementation
                return xgb.XGBClassifier(n_estimators=100, random_state=42)
        
        elif task == 'regression':
            if self.model_type == 'rf':
                return RandomForestRegressor(n_estimators=100, random_state=42)
            elif self.model_type == 'xgb':
                return xgb.XGBRegressor(n_estimators=100, random_state=42)
            elif self.model_type == 'lgbm':
                return lgbm.LGBMRegressor(n_estimators=100, random_state=42)
            elif self.model_type == 'lstm':
                # Placeholder for LSTM implementation
                return xgb.XGBRegressor(n_estimators=100, random_state=42)
        elif task == 'zone':
            if self.model_type == 'rf':
                return RandomForestClassifier(n_estimators=100, random_state=42)
            elif self.model_type == 'xgb':
                # Configure XGBoost to handle any class labels by enabling validation of parameters
                return xgb.XGBClassifier(n_estimators=100, random_state=42, validate_parameters=False)
            elif self.model_type == 'lgbm':
                return lgbm.LGBMClassifier(n_estimators=100, random_state=42)
            elif self.model_type == 'lstm':
                # Placeholder for LSTM implementation
                return xgb.XGBClassifier(n_estimators=100, random_state=42, validate_parameters=False)
    
    def fit(self, X, y_binary, y_days=None, y_zone=None):
        """Fit all applicable models"""
        self.feature_names = X.columns.tolist()
        
        # Scale features
        self.scalers['features'] = StandardScaler()
        X_scaled = self.scalers['features'].fit_transform(X)
        
        # Binary classification model
        if y_binary is not None:
            self.models['binary_classifier'] = self._create_model('binary')
            self.models['binary_classifier'].fit(X_scaled, y_binary)
          # Regression model for days prediction
        if y_days is not None:
            # Scale days to critical            
            self.scalers['days'] = StandardScaler()
            y_days_scaled = self.scalers['days'].fit_transform(y_days.values.reshape(-1, 1)).ravel()
            
            self.models['days_regressor'] = self._create_model('regression')
            self.models['days_regressor'].fit(X_scaled, y_days_scaled)
          # Zone prediction model
        if y_zone is not None:
            # Filter out rows where zone is -1 (no accretion zone)
            valid_zone_mask = (y_zone != -1)
            if valid_zone_mask.sum() > 0:
                X_zone = X_scaled[valid_zone_mask]
                y_zone_valid = y_zone[valid_zone_mask]
                
                # Get the unique zone classes and print them
                unique_zones = sorted(y_zone_valid.unique())
                print(f"Training zone classifier on {len(y_zone_valid)} samples with zones: {unique_zones}")
                
                # Create and fit the model with the actual classes                # Create a label mapper for zone classes to make them sequential
                print(f"Creating label mapping for zone classes: {unique_zones} -> [0, 1, 2...]")
                self.zone_class_mapper = {val: idx for idx, val in enumerate(unique_zones)}
                self.zone_class_reverse_mapper = {idx: val for idx, val in enumerate(unique_zones)}
                
                # Apply the mapping to make classes sequential
                y_zone_mapped = np.array([self.zone_class_mapper[val] for val in y_zone_valid])
                print(f"Mapped zone classes from {unique_zones} to {sorted(np.unique(y_zone_mapped))}")
                
                # Create the zone classifier
                self.models['zone_classifier'] = self._create_model('zone')
                
                # Fit the model with the mapped values
                self.models['zone_classifier'].fit(X_zone, y_zone_mapped)
                
                # Store the zone classes for later use in prediction
                self.zone_classes = unique_zones
            else:
                print("Warning: No valid zone data for training zone classifier")
                # Create a dummy classifier
                self.models['zone_classifier'] = self._create_model('zone')
                self.zone_classes = None
    
    def predict(self, X):
        """Make predictions with all models"""
        if not self.models:
            raise ValueError("Models not trained. Call fit() first.")
        
        X_scaled = self.scalers['features'].transform(X)
        results = {}
        
        # Binary classification prediction
        if 'binary_classifier' in self.models:
            results['binary_proba'] = self.models['binary_classifier'].predict_proba(X_scaled)[:, 1]
            # Note: With updated data generator, the model is trained to detect temperature drops
            # instead of increases as indicators of accretion
            results['binary'] = (results['binary_proba'] > 0.5).astype(int)
        
        # Days to critical prediction
        if 'days_regressor' in self.models:
            days_scaled = self.models['days_regressor'].predict(X_scaled)
            results['days'] = self.scalers['days'].inverse_transform(days_scaled.reshape(-1, 1)).ravel()        # Zone prediction
        if 'zone_classifier' in self.models and hasattr(self.models['zone_classifier'], 'classes_'):
            # First predict accretion forming/not forming
            if 'binary' in results:
                # Only predict zones where accretion is forming
                forming_mask = results['binary'] == 1
                
                # Initialize arrays with -1 (no accretion)
                zone_preds = np.full(len(X), -1)
                
                # Get number of zone classes from the classifier
                n_classes = len(self.models['zone_classifier'].classes_)
                zone_probs = np.zeros((len(X), n_classes))
                  # Make predictions only for rows where accretion is forming
                if forming_mask.sum() > 0:
                    X_forming = X_scaled[forming_mask]
                    raw_preds = self.models['zone_classifier'].predict(X_forming)
                    
                    # Map the predictions back to original zone numbers if we have a mapper
                    if hasattr(self, 'zone_class_mapper') and self.zone_class_reverse_mapper:
                        zone_preds[forming_mask] = np.array([self.zone_class_reverse_mapper[pred] for pred in raw_preds])
                        print(f"Mapped zone predictions from {sorted(np.unique(raw_preds))} back to {sorted(np.unique(zone_preds[forming_mask]))}")
                    else:
                        zone_preds[forming_mask] = raw_preds
                        
                    zone_probs[forming_mask] = self.models['zone_classifier'].predict_proba(X_forming)
                
                results['zone'] = zone_preds
                results['zone_proba'] = zone_probs
            else:
                # Fallback if binary predictions aren't available
                results['zone'] = self.models['zone_classifier'].predict(X_scaled)
                results['zone_proba'] = self.models['zone_classifier'].predict_proba(X_scaled)
        
        return results
    
    def evaluate(self, X, y_binary, y_days_test=None, y_zone_test=None):
        """Evaluate model performance"""
        predictions = self.predict(X)
        results = {}
        
        # Binary classification metrics
        if 'binary' in predictions and y_binary is not None:
            binary_preds = predictions['binary']
            binary_probs = predictions['binary_proba']
            
            results['binary'] = {
                'accuracy': accuracy_score(y_binary, binary_preds),
                'precision': precision_score(y_binary, binary_preds, zero_division=0),
                'recall': recall_score(y_binary, binary_preds, zero_division=0),
                'f1_score': f1_score(y_binary, binary_preds, zero_division=0),
                'roc_auc': roc_auc_score(y_binary, binary_probs)
            }
        
        # Regression metrics
        if 'days' in predictions and y_days_test is not None:
            days_preds = predictions['days']
            
            results['regression'] = {
                'mse': mean_squared_error(y_days_test, days_preds),
                'rmse': np.sqrt(mean_squared_error(y_days_test, days_preds)),
                'r2': r2_score(y_days_test, days_preds)
            }        # Zone classification metrics
        if 'zone' in predictions and y_zone_test is not None:
            zone_preds = predictions['zone']
            
            # Only evaluate on non-negative zone values (actual accretion zones)
            valid_mask = (y_zone_test >= 0)
            if valid_mask.sum() > 0:
                try:
                    # Get the actual values to compare
                    y_true = y_zone_test[valid_mask]
                    y_pred = zone_preds[valid_mask]
                    
                    print("Calculating zone prediction metrics...")
                    print(f"Truth values: {sorted(np.unique(y_true))}")
                    print(f"Prediction values: {sorted(np.unique(y_pred))}")
                    
                    # Calculate accuracy
                    acc = accuracy_score(y_true, y_pred)
                    
                    results['zone'] = {
                        'accuracy': acc,
                        'unique_classes_truth': sorted(np.unique(y_true)),
                        'unique_classes_pred': sorted(np.unique(y_pred))
                    }
                    print(f"Zone classification accuracy: {results['zone']['accuracy']:.4f}")
                    print(f"Zone classes in truth: {results['zone']['unique_classes_truth']}")
                    print(f"Zone classes in predictions: {results['zone']['unique_classes_pred']}")
                except Exception as e:
                    print(f"Error calculating zone classification metrics: {e}")
                    print(f"Error details: {str(e)}")
                    results['zone'] = {
                        'accuracy': float('nan'),
                        'error': str(e)
                    }
            else:
                results['zone'] = {
                    'accuracy': float('nan'),  # No valid zones to evaluate
                    'note': 'No valid accretion zones in test set'
                }
        
        return results
    
    def explain(self, X):
        """Get feature importance and explanations"""
        X_scaled = self.scalers['features'].transform(X)
        explanations = {}
        
        # Get feature importance for binary classifier
        if 'binary_classifier' in self.models:
            if hasattr(self.models['binary_classifier'], 'feature_importances_'):
                importances = self.models['binary_classifier'].feature_importances_
                feature_imp = list(zip(self.feature_names, importances))
                feature_imp.sort(key=lambda x: x[1], reverse=True)
                explanations['binary'] = {'top_features': feature_imp}
        
        # Get feature importance for regressor
        if 'days_regressor' in self.models:
            if hasattr(self.models['days_regressor'], 'feature_importances_'):
                importances = self.models['days_regressor'].feature_importances_
                feature_imp = list(zip(self.feature_names, importances))
                feature_imp.sort(key=lambda x: x[1], reverse=True)
                explanations['regression'] = {'top_features': feature_imp}
        
        # Get feature importance for zone classifier
        if 'zone_classifier' in self.models:
            if hasattr(self.models['zone_classifier'], 'feature_importances_'):
                importances = self.models['zone_classifier'].feature_importances_
                feature_imp = list(zip(self.feature_names, importances))
                feature_imp.sort(key=lambda x: x[1], reverse=True)
                explanations['zone'] = {'top_features': feature_imp}
        
        return explanations
    
    def save(self, path):
        """Save model and preprocessors"""
        os.makedirs(path, exist_ok=True)
        
        # Save models
        for name, model in self.models.items():
            joblib.dump(model, os.path.join(path, f'{name}.pkl'))
        
        # Save scalers
        for name, scaler in self.scalers.items():
            joblib.dump(scaler, os.path.join(path, f'{name}_scaler.pkl'))
        
        # Save feature names
        if self.feature_names:
            with open(os.path.join(path, 'feature_names.txt'), 'w') as f:
                f.write('\n'.join(self.feature_names))
    
    def load(self, path):
        """Load model and preprocessors"""
        # Load models
        for model_file in os.listdir(path):
            if model_file.endswith('.pkl'):
                name = model_file.split('.')[0]
                if name.endswith('_scaler'):
                    scaler_name = name.replace('_scaler', '')
                    self.scalers[scaler_name] = joblib.load(os.path.join(path, model_file))
                else:
                    self.models[name] = joblib.load(os.path.join(path, model_file))
        
        # Load feature names
        feature_file = os.path.join(path, 'feature_names.txt')
        if os.path.exists(feature_file):
            with open(feature_file, 'r') as f:
                self.feature_names = f.read().splitlines()
    
    def update_incrementally(self, X, y_binary, y_days=None, y_zone=None, sample_weight=None):
        """
        Update existing models with new data using incremental learning
        This allows the model to adapt to new patterns without complete retraining
        
        Args:
            X (pd.DataFrame): New feature data
            y_binary (pd.Series): New binary target data
            y_days (pd.Series): New days to critical target data
            y_zone (pd.Series): New zone target data
            sample_weight (array-like): Sample weights for training
            
        Returns:
            dict: Performance metrics from update
        """
        if not self.models:
            return self.fit(X, y_binary, y_days, y_zone)
            
        # Record timestamp for this update
        timestamp = pd.Timestamp.now()
        metrics = {}
        
        # Scale new features using existing scalers
        X_scaled = self.scalers['features'].transform(X)
        
        # Update binary classification model if applicable
        if 'binary_classifier' in self.models and y_binary is not None:
            if self.model_type in ['xgb', 'lgbm']:
                # Both XGBoost and LightGBM support incremental learning
                self.models['binary_classifier'].fit(
                    X_scaled, y_binary, 
                    xgb_model=self.models['binary_classifier'] if self.model_type == 'xgb' else None,
                    sample_weight=sample_weight
                )
                
                # Evaluate performance on new data
                binary_pred = (self.models['binary_classifier'].predict_proba(X_scaled)[:, 1] > 0.5).astype(int)
                metrics['binary'] = {
                    'accuracy': accuracy_score(y_binary, binary_pred),
                    'f1': f1_score(y_binary, binary_pred, zero_division=0)
                }
                
                # Update training history
                self.training_history['binary']['timestamps'].append(timestamp)
                self.training_history['binary']['samples'].append(len(X))
                self.training_history['binary']['metrics'].append(metrics['binary'])
        
        # Update days regressor if applicable
        if 'days_regressor' in self.models and y_days is not None:
            # Scale days to critical
            y_days_scaled = self.scalers['days'].transform(y_days.values.reshape(-1, 1)).ravel()
            
            if self.model_type in ['xgb', 'lgbm']:
                self.models['days_regressor'].fit(
                    X_scaled, y_days_scaled, 
                    xgb_model=self.models['days_regressor'] if self.model_type == 'xgb' else None,
                    sample_weight=sample_weight
                )
                
                # Evaluate performance on new data
                days_pred = self.models['days_regressor'].predict(X_scaled)
                days_pred_unscaled = self.scalers['days'].inverse_transform(days_pred.reshape(-1, 1)).ravel()
                metrics['regression'] = {
                    'rmse': np.sqrt(mean_squared_error(y_days, days_pred_unscaled)),
                    'r2': r2_score(y_days, days_pred_unscaled)
                }
                
                # Update training history
                self.training_history['regression']['timestamps'].append(timestamp)
                self.training_history['regression']['samples'].append(len(X))
                self.training_history['regression']['metrics'].append(metrics['regression'])
        
        # Update zone classifier if applicable
        if 'zone_classifier' in self.models and y_zone is not None:
            # Filter out rows where zone is -1 (no accretion zone)
            valid_zone_mask = (y_zone != -1)
            
            if valid_zone_mask.sum() > 0:
                X_zone = X_scaled[valid_zone_mask]
                y_zone_valid = y_zone[valid_zone_mask]
                
                # Map zone values to sequential indices 
                y_zone_mapped = np.array([self.zone_class_mapper.get(val, -1) for val in y_zone_valid])
                valid_mapping = (y_zone_mapped != -1)
                
                # Only proceed if we have valid mappings
                if valid_mapping.sum() > 0:
                    X_zone = X_zone[valid_mapping]
                    y_zone_mapped = y_zone_mapped[valid_mapping]
                    
                    # XGBoost and LightGBM support incremental learning
                    if self.model_type in ['xgb', 'lgbm']:
                        self.models['zone_classifier'].fit(
                            X_zone, y_zone_mapped, 
                            xgb_model=self.models['zone_classifier'] if self.model_type == 'xgb' else None,
                            sample_weight=None if sample_weight is None else sample_weight[valid_zone_mask][valid_mapping]
                        )
                        
                        # Evaluate performance
                        zone_preds = self.models['zone_classifier'].predict(X_zone)
                        metrics['zone'] = {
                            'accuracy': accuracy_score(y_zone_mapped, zone_preds)
                        }
                        
                        # Update training history
                        self.training_history['zone']['timestamps'].append(timestamp)
                        self.training_history['zone']['samples'].append(len(X_zone))
                        self.training_history['zone']['metrics'].append(metrics['zone'])
        
        return metrics
    
    def get_training_history(self):
        """Get the incremental learning history for visualization"""
        return self.training_history


class KilnAccretionPrescriptor:
    """
    Provides recommendations on how to adjust operational parameters
    to prevent or mitigate accretion formation.
    """
    def __init__(self):
        self.models = {}
        self.feature_names = None
        self.action_space = {}
        self.feature_importances = {}
    
    def define_action_space(self, action_space):
        """Define the allowed ranges for parameter adjustments"""
        self.action_space = action_space
    
    def fit(self, X, Y, sample_weight=None):
        """
        Fit prescription models for each adjustable parameter
        Y contains the target adjustments for each parameter
        
        Args:
            X (pd.DataFrame): Feature matrix
            Y (pd.DataFrame): Target adjustments for each parameter
            sample_weight (array-like, optional): Sample weights for training
            
        Returns:
            self: The fitted prescriptor
        """
        self.feature_names = X.columns.tolist()
        
        trained_params = []
        skipped_params = []
        
        for param in Y.columns:            # Check for parameter variability using standard deviation
            # More robust than just checking for sum == 0
            param_std = Y[param].std()
            
            # Skip parameters with very low or zero variation
            # (using a small epsilon to account for floating point precision)
            if param_std < 1e-9:  # Reduced threshold to include more parameters
                skipped_params.append(param)
                continue
                
            # Check for sufficient non-zero values
            non_zero = (Y[param].abs() > 1e-9).sum()
            if non_zero < len(Y) * 0.001:  # Reduced threshold from 1% to 0.1% to include more parameters
                skipped_params.append(param)
                continue
            
            # Create regressor for this parameter with better hyperparameters
            try:
                model = xgb.XGBRegressor(
                    n_estimators=100, 
                    learning_rate=0.05,
                    max_depth=5,
                    min_child_weight=2,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42
                )
                
                # Train the model to predict the adjustment
                model.fit(X, Y[param], sample_weight=sample_weight)
                
                # Store the model
                self.models[param] = model
                trained_params.append(param)
                
                # Store feature importances
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    sorted_idx = np.argsort(importances)[::-1]  # Sort in descending order
                    
                    # Store sorted feature importances for better interpretability
                    sorted_features = [self.feature_names[i] for i in sorted_idx]
                    sorted_importances = importances[sorted_idx]
                    self.feature_importances[param] = dict(zip(sorted_features, sorted_importances))
                    
            except Exception as e:
                print(f"Error training model for parameter {param}: {e}")
                skipped_params.append(param)
                
        print(f"Successfully trained models for {len(trained_params)} parameters")
        if trained_params:
            print(f"Trained parameters: {', '.join(trained_params)}")
        
        if skipped_params:
            print(f"Skipped {len(skipped_params)} parameters due to insufficient variation")
            print(f"Skipped parameters: {', '.join(skipped_params)}")
            
        return self
    
    def predict(self, X):
        """Predict recommended parameter adjustments"""
        if not self.models:
            return {}
        
        recommendations = {}
        
        for param, model in self.models.items():
            # Get raw prediction
            adjustment = model.predict(X)
            
            # Apply constraints from action space
            if param in self.action_space:
                constraints = self.action_space[param]
                min_val = constraints.get('min', -float('inf'))
                max_val = constraints.get('max', float('inf'))
                step = constraints.get('step', None)
                
                # Clip to min/max
                adjustment = np.clip(adjustment, min_val, max_val)
                
                # Round to nearest step if specified
                if step:
                    adjustment = np.round(adjustment / step) * step
            
            recommendations[param] = adjustment
        
        return recommendations
    
    def recommend_actions(self, X, current_params=None, top_n=3):
        """
        Generate actionable recommendations based on current parameters
        Returns top N most impactful recommendations
        """
        # Get raw adjustment predictions
        adjustments = self.predict(X)
        
        # If we have current parameters, we can show relative changes
        if current_params:
            recommendations = []
            
            for param, adj_values in adjustments.items():
                if param in current_params:
                    current = current_params[param]
                    
                    for i, adj in enumerate(adj_values):
                        # Skip very small adjustments
                        if abs(adj) < 0.01 * current:
                            continue
                        
                        new_value = current + adj
                        percent_change = (adj / current) * 100
                        
                        recommendations.append({
                            'parameter': param,
                            'current_value': current,
                            'recommended_value': new_value,
                            'adjustment': adj,
                            'percent_change': percent_change,
                            'impact_score': abs(percent_change)  # Use percent change as impact score
                        })
            
            # Sort by impact and return top N
            recommendations.sort(key=lambda x: x['impact_score'], reverse=True)
            return recommendations[:top_n]
        else:
            # Without current parameters, just return the raw adjustments
            return adjustments
    
    def save(self, path):
        """Save prescription models"""
        os.makedirs(path, exist_ok=True)
        
        # Save models
        for param, model in self.models.items():
            safe_param = param.replace(' ', '_').replace('/', '_')
            joblib.dump(model, os.path.join(path, f'{safe_param}.pkl'))
        
        # Save action space
        if self.action_space:
            joblib.dump(self.action_space, os.path.join(path, 'action_space.pkl'))
        
        # Save feature names
        if self.feature_names:
            with open(os.path.join(path, 'feature_names.txt'), 'w') as f:
                f.write('\n'.join(self.feature_names))
    
    def load(self, path):
        """Load prescription models"""
        # Load models
        for model_file in os.listdir(path):
            if model_file.endswith('.pkl'):
                name = model_file.split('.')[0]
                if name == 'action_space':
                    self.action_space = joblib.load(os.path.join(path, model_file))
                else:
                    # Convert underscores back to spaces and slashes
                    param = name.replace('_', ' ')
                    self.models[param] = joblib.load(os.path.join(path, model_file))
        
        # Load feature names
        feature_file = os.path.join(path, 'feature_names.txt')
        if os.path.exists(feature_file):
            with open(feature_file, 'r') as f:
                self.feature_names = f.read().splitlines()
