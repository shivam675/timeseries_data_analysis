"""
Synthetic Data Generator for Kiln Accretion Prediction

This script generates synthetic data for a rotary kiln in a sponge iron plant.
The data simulates various sensors, operational parameters, and accretion formation events.

Key features:
- 3+ years of data with different measurement frequencies
- Accretion events occurring every 60-90 days
- 48-hour lag between operational changes and effects
- 15-60 day lag between early symptoms and critical accretion
"""

import numpy as np
import pandas as pd
import datetime
import random
import os
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Define simulation parameters
START_DATE = datetime.datetime(2024, 6, 1)
END_DATE = datetime.datetime(2025, 6, 1)  # 3.5 years
ACCRETION_FREQUENCY_DAYS = random.randint(60, 90)  # Random between 60-90 days
OPERATIONAL_LAG_HOURS = 48  # Lag between operational changes and effects
SYMPTOM_TO_CRITICAL_DAYS = [15, 30, 45, 60]  # Different progression timelines

# Define zones and temperature ranges - updated to reflect normal operating temperatures
ZONES = list(range(11))  # 0 to 10
ZONE_TEMP_RANGES = {
    0: (750, 800),
    1: (775, 825),
    2: (800, 850),
    3: (825, 875),
    4: (850, 900),
    5: (875, 925),
    6: (875, 925),
    7: (875, 925),
    8: (850, 900),
    9: (825, 875),
    10: (800, 850),
}

QRT_TEMP_RANGES = {
    2: (900, 950),
    3: (925, 975),
    4: (950, 1000),
    5: (975, 1025),
    6: (1000, 1050),
    7: (1000, 1050),
    8: (975, 1025),
    9: (950, 1000),
    10: (925, 975),
}

# Define positions along the kiln (for shell temperature)
KILN_POSITIONS = [
    'O/L CONE', 'CONE-10th no.', '10th no.-tyre', 'tyre-9th no.', '9th no.', 
    '9th no.-8th no.', '8th no.', '8th no.-7th no.', '7th no.', '7th no.-6th no.', 
    '6th no.', '6th no.-5th no.', '5th no.', '5th no.-tyre', 'tyre-4th no.', 
    '4th no.', '4th no.-3rd no.', '3rd no.', '3rd no.-2nd no.', '2nd no.', 
    '2nd no.-1st no.', '1st no.'
]

# Define air fans
AIR_FANS = ['SAF02', 'SAF03', 'SAF04', 'SAF05', 'SAF06', 'SAF07', 'SAF08', 'SAF09', 'CB']

# Define normal operating ranges
NORMAL_AIRFLOW_BASE = 55000  # Base airflow
NORMAL_AIRFLOW_VARIANCE = 5000  # Airflow variance
ORE_TO_COAL_RATIO = 1 / 0.8  # Ore to coal ratio is 1:0.8

# Output directories
OUTPUT_DIR = 'synthetic_data'
os.makedirs(OUTPUT_DIR, exist_ok=True)


class AccretionModel:
    """
    Simulates the formation and effects of accretion in the kiln.
    Accretion formation is now influenced by material characteristics.
    """
    def __init__(self):
        self.accretion_events = []
        self.accretion_locations = []
        self.current_accretions = {}  # zone -> {severity, start_date, critical_date}
        # Add dynamic accretion chance based on material conditions
        self.dynamic_accretion_chance = 0.005  # Daily chance of accretion when conditions are poor
        self.generate_accretion_schedule()
    
    def generate_accretion_schedule(self):
        """Generate a schedule of accretion events throughout the simulation period"""
        current_date = START_DATE
        while current_date < END_DATE:
            # Decide random accretion start date
            days_to_next = random.randint(ACCRETION_FREQUENCY_DAYS - 10, ACCRETION_FREQUENCY_DAYS + 10)
            current_date += datetime.timedelta(days=days_to_next)
            
            if current_date >= END_DATE:
                break
                
            # Decide random zone for accretion - CONFINE TO ZONES 3-8 ONLY
            # Higher weights on zones 5-7 which are the hottest sections of the kiln
            limited_zones = [3, 4, 5, 6, 7, 8]
            zone_weights = [0.1, 0.15, 0.25, 0.25, 0.15, 0.1]  # weights for zones 3-8
            accretion_zone = random.choices(limited_zones, weights=zone_weights)[0]
            
            # Decide how long until critical
            days_to_critical = random.choice(SYMPTOM_TO_CRITICAL_DAYS)
            critical_date = current_date + datetime.timedelta(days=days_to_critical)
              # Record the event
            event = {
                'start_date': current_date,
                'critical_date': critical_date,
                'zone': accretion_zone,
                'cleared_date': None,  # Will be set if/when cleared
                'cause': 'Scheduled event',  # Default cause
                'material_factors': {
                    'fines_ratio': random.uniform(0.15, 0.3),  # Higher fines ratio increases accretion risk
                    'coal_quality': random.uniform(0.6, 0.8)   # Lower coal quality increases accretion risk
                }
            }
            self.accretion_events.append(event)
            self.accretion_locations.append(accretion_zone)
    
    def get_active_accretions(self, date):
        """Return information about any active accretions on a given date"""
        active = []
        for event in self.accretion_events:
            if event['start_date'] <= date and (event['cleared_date'] is None or date <= event['cleared_date']):
                # Calculate severity as a function of time progression
                if date >= event['critical_date']:
                    severity = 1.0  # Full accretion
                else:
                    total_days = (event['critical_date'] - event['start_date']).total_seconds() / (24 * 3600)
                    elapsed_days = (date - event['start_date']).total_seconds() / (24 * 3600)
                    # Non-linear growth: slow at first, then accelerating
                    severity = (elapsed_days / total_days) ** 1.5
                
                active.append({
                    'zone': event['zone'],
                    'severity': severity,
                    'start_date': event['start_date'],
                    'critical_date': event['critical_date'],
                    'days_active': (date - event['start_date']).days                })
        return active
    
    def clear_accretion(self, event_idx, clear_date):
        """Mark an accretion event as cleared"""
        if 0 <= event_idx < len(self.accretion_events):
            self.accretion_events[event_idx]['cleared_date'] = clear_date
            return True
        return False
    
    def get_temperature_impact(self, zone, date):
        """Calculate the temperature impact of accretions on a given zone"""
        impact = 0
        for accr in self.get_active_accretions(date):
            # Direct impact on the zone with accretion
            if accr['zone'] == zone:
                # Higher severity means more impact on temperature
                # Accretion causes temperature to drop (negative impact)
                impact -= 200 * accr['severity']  # up to 200°C decrease
            
            # Neighboring zones are also affected, but less so
            elif abs(accr['zone'] - zone) == 1:
                impact -= 100 * accr['severity']  # up to 100°C decrease
            elif abs(accr['zone'] - zone) == 2:
                impact -= 50 * accr['severity']  # up to 50°C decrease
        
        return impact
    
    def check_material_accretion_risk(self, inputs, date):
        """
        Calculate accretion risk based on current material inputs and conditions
        Returns a risk score 0-1 and the most likely zone for accretion
        """
        # High risk factors:
        # 1. High fines ratio (ESSAR_FINES + NCL_FINES)
        # 2. Poor coal quality (low HG to total coal ratio)
        # 3. Unbalanced iron to coal ratio (should be close to optimal)
        # 4. High percentage of fine particles
        
        risk_score = 0
        
        # Check fines ratio - higher is worse
        fines_ratio = inputs.get('FINES_RATIO', 0)
        if fines_ratio > 0.15:  # Above 15% is concerning
            risk_score += min(0.4, (fines_ratio - 0.15) * 2)  # Up to 0.4 points
        
        # Check coal quality - HG coal should be dominant
        hg_ratio = inputs.get('HG_TO_TOTAL_COAL', 0)
        if hg_ratio < 0.6:  # Below 60% HG coal is concerning
            risk_score += min(0.3, (0.6 - hg_ratio) * 1.5)  # Up to 0.3 points
            
        # Check iron to coal ratio - should be close to optimal (0.9-1.3)
        iron_coal = inputs.get('IRON_TO_COAL_RATIO', 1.0)
        if iron_coal < 0.9 or iron_coal > 1.3:
            deviation = min(abs(iron_coal - 0.9), abs(iron_coal - 1.3))
            risk_score += min(0.3, deviation * 0.6)  # Up to 0.3 points
        
        # Check fine particle percentage - above 30% is concerning
        fine_particles = inputs.get('FINE_PARTICLES', 25)
        if fine_particles > 30:
            risk_score += min(0.2, (fine_particles - 30) * 0.02)  # Up to 0.2 points
            
        # Select the most likely zone for accretion
        # Zones 5-6 are most susceptible with poor material conditions
        limited_zones = [3, 4, 5, 6, 7, 8]
        
        if risk_score > 0.7:  # Very high risk - most likely in hottest zones
            zone_weights = [0.05, 0.15, 0.35, 0.35, 0.05, 0.05]
        elif risk_score > 0.4:  # Medium risk
            zone_weights = [0.1, 0.15, 0.25, 0.25, 0.15, 0.1]
        else:  # Lower risk - more evenly distributed
            zone_weights = [0.15, 0.17, 0.18, 0.18, 0.17, 0.15]
            
        likely_zone = random.choices(limited_zones, weights=zone_weights)[0]
        
        return risk_score, likely_zone
    
    def check_for_dynamic_accretion(self, date, inputs):
        """
        Check if a new accretion should form based on material conditions
        This runs daily to simulate accretion formation due to poor material conditions
        """
        # Skip check if already have active accretions
        if self.get_active_accretions(date):
            return False
            
        # Calculate material-based risk
        risk_score, likely_zone = self.check_material_accretion_risk(inputs, date)
        
        # Roll for accretion chance based on risk score
        # Higher risk means higher chance of accretion
        accretion_chance = self.dynamic_accretion_chance * (1 + risk_score * 10)  # Up to 11x higher with max risk
        
        if random.random() < accretion_chance:
            # Accretion will form due to poor material conditions
            days_to_critical = random.choice(SYMPTOM_TO_CRITICAL_DAYS)
            critical_date = date + datetime.timedelta(days=days_to_critical)
            
            # Material conditions determine severity progression rate
            # Higher risk means faster progression (shorter time to critical)
            critical_date_adjusted = date + datetime.timedelta(days=int(days_to_critical * (1 - risk_score * 0.5)))
            
            event = {
                'start_date': date,
                'critical_date': critical_date_adjusted,
                'zone': likely_zone,
                'cleared_date': None,
                'cause': 'Material conditions',
                'material_factors': {
                    'fines_ratio': inputs.get('FINES_RATIO', 0),
                    'coal_quality': inputs.get('HG_TO_TOTAL_COAL', 0),
                    'fine_particles': inputs.get('FINE_PARTICLES', 0),
                    'risk_score': risk_score
                }
            }
            
            self.accretion_events.append(event)
            self.accretion_locations.append(likely_zone)
            return True
            
        return False


class InputModel:
    """
    Simulates the input materials and their changes over time.
    Tracks material quality, ratios, and particle size distributions.
    """
    def __init__(self):
        self.current_inputs = {
            # Base consumption values
            'PELLETS_CONSUMPTION': 500,  # tons/day
            'IRON_ORE_CONSUMPTION': 1000,  # tons/day
            'HG_COAL_CONSUMPTION': 800,  # tons/day
            'SA_COAL_CONSUMPTION': 400,  # tons/day
            'ESSAR_FINES': 50,  # tons/day
            'NCL_FINES': 70,  # tons/day
            'WASH_COAL': 250,  # tons/day
            'DOLO_CONSUMPTION': 150,  # tons/day
            
            # Material quality factors (0-1 scale)
            'IRON_QUALITY': 0.85,  # Quality of iron ore
            'COAL_QUALITY': 0.82,  # Quality of coal
            
            # Particle size distribution percentages
            'FINE_PARTICLES': 25,  # % of fine particles in input feed
            'MEDIUM_PARTICLES': 50,  # % of medium particles in input feed
            'COARSE_PARTICLES': 25,  # % of coarse particles in input feed
        }
        self.input_changes = []  # Track when inputs are changed
        
    def adjust_inputs(self, date, changes, reason="Regular adjustment"):
        """Record an adjustment to input parameters"""
        for param, value in changes.items():
            if param in self.current_inputs:
                self.current_inputs[param] = value
                
        self.input_changes.append({
            'date': date,
            'changes': changes.copy(),
            'reason': reason
        })
    
    def get_current_inputs(self, date):
        """Get the input values for a specific date"""
        # Find the most recent changes before this date
        relevant_inputs = self.current_inputs.copy()
        for change in self.input_changes:
            if change['date'] <= date:
                for param, value in change['changes'].items():
                    relevant_inputs[param] = value
        
        # Calculate derived metrics based on inputs
        inputs = relevant_inputs.copy()
        
        # Calculate key ratios that affect accretion and product grade
        inputs['IRON_TO_COAL_RATIO'] = (inputs['PELLETS_CONSUMPTION'] + inputs['IRON_ORE_CONSUMPTION']) / \
                                      (inputs['HG_COAL_CONSUMPTION'] + inputs['SA_COAL_CONSUMPTION'] + 
                                       inputs['ESSAR_FINES'] + inputs['NCL_FINES'] + inputs['WASH_COAL'])
        
        # Quality ratio of high-grade to low-grade coal
        inputs['HG_TO_TOTAL_COAL'] = inputs['HG_COAL_CONSUMPTION'] / \
                                    (inputs['HG_COAL_CONSUMPTION'] + inputs['SA_COAL_CONSUMPTION'] + 
                                     inputs['ESSAR_FINES'] + inputs['NCL_FINES'] + inputs['WASH_COAL'])
                                     
        # Fines ratio - higher fines tend to cause accretion issues
        inputs['FINES_RATIO'] = (inputs['ESSAR_FINES'] + inputs['NCL_FINES']) / \
                               (inputs['HG_COAL_CONSUMPTION'] + inputs['SA_COAL_CONSUMPTION'] + 
                                inputs['ESSAR_FINES'] + inputs['NCL_FINES'] + inputs['WASH_COAL'])
        
        return inputs


class KilnSimulator:
    """
    Main simulator class that generates all data for the kiln.
    """
    def __init__(self):
        self.input_model = InputModel()
        self.accretion_model = AccretionModel()
        self.maintenance_events = []
        
        # Link input model to accretion model for risk assessment
        self.accretion_model.input_model = self.input_model
        
        # Regular maintenance schedule (every ~120 days)
        current_date = START_DATE + datetime.timedelta(days=120)
        while current_date < END_DATE:
            duration_days = random.randint(3, 7)
            self.maintenance_events.append({
                'start_date': current_date,
                'end_date': current_date + datetime.timedelta(days=duration_days),
                'type': 'Regular maintenance'
            })
            current_date += datetime.timedelta(days=random.randint(110, 130))
          # Generate random operational changes to test lag effects
        current_date = START_DATE + datetime.timedelta(days=30)
        while current_date < END_DATE:
            # Make random adjustments to inputs periodically
            if random.random() < 0.7:  # 70% chance of making changes
                changes = {}
                # First, select normal input consumption parameters to change
                consumption_params = ['PELLETS_CONSUMPTION', 'IRON_ORE_CONSUMPTION', 'HG_COAL_CONSUMPTION', 
                                      'SA_COAL_CONSUMPTION', 'ESSAR_FINES', 'NCL_FINES', 'WASH_COAL', 
                                      'DOLO_CONSUMPTION']
                
                for param in random.sample(consumption_params, random.randint(1, 3)):
                    current_val = self.input_model.current_inputs[param]
                    changes[param] = current_val * random.uniform(0.9, 1.1)  # ±10%
                
                # Occasionally change material quality parameters
                if random.random() < 0.3:  # 30% chance to change quality parameters
                    quality_params = ['IRON_QUALITY', 'COAL_QUALITY', 'FINE_PARTICLES', 
                                      'MEDIUM_PARTICLES', 'COARSE_PARTICLES']
                    
                    # Select 1-2 quality parameters to change
                    for param in random.sample(quality_params, random.randint(1, 2)):
                        current_val = self.input_model.current_inputs[param]
                        
                        # For particle distribution, keep the sum at 100%
                        if 'PARTICLES' in param:
                            # Random variation of ±5 percentage points
                            delta = random.uniform(-5, 5)
                            changes[param] = current_val + delta
                            
                            # If we modify one particle size, adjust another to maintain total of 100%
                            other_particle_params = [p for p in ['FINE_PARTICLES', 'MEDIUM_PARTICLES', 'COARSE_PARTICLES'] if p != param]
                            compensate_param = random.choice(other_particle_params)
                            changes[compensate_param] = self.input_model.current_inputs[compensate_param] - delta
                        else:
                            # For quality parameters, random variation of ±10%
                            changes[param] = min(1.0, max(0.5, current_val * random.uniform(0.9, 1.1)))
                
                self.input_model.adjust_inputs(
                    current_date, 
                    changes,
                    reason="Optimization attempt"
                )
            
            # Next change after 5-15 days
            current_date += datetime.timedelta(days=random.randint(5, 15))
    
    def handle_maintenance(self, date):
        """Check if maintenance is happening and handle accretion clearing"""
        for maint in self.maintenance_events:
            if maint['start_date'] <= date <= maint['end_date']:
                # During maintenance, clear any accretions that are present
                for i, event in enumerate(self.accretion_model.accretion_events):
                    if (event['cleared_date'] is None and 
                        event['start_date'] <= maint['start_date']):
                        self.accretion_model.clear_accretion(i, maint['end_date'])
                return True
        return False
    
    def is_running(self, date):
        """Check if the kiln is running or down for maintenance"""
        for maint in self.maintenance_events:
            if maint['start_date'] <= date <= maint['end_date']:
                return False
        return True
    
    def generate_zone_temperatures(self, date, base_noise=0.02):
        """Generate temperature data for all zones for a specific date"""
        is_maintenance = not self.is_running(date)
        if is_maintenance:
            # During maintenance, temperatures are much lower
            return {zone: random.uniform(100, 200) for zone in ZONES}
        
        active_accretions = self.accretion_model.get_active_accretions(date)
        
        temps = {}
        for zone in ZONES:
            min_temp, max_temp = ZONE_TEMP_RANGES[zone]
            base_temp = random.uniform(min_temp, max_temp)
            
            # Apply noise
            noise_factor = random.uniform(1 - base_noise, 1 + base_noise)
            
            # Apply accretion effects
            accretion_impact = self.accretion_model.get_temperature_impact(zone, date)
            
            final_temp = base_temp * noise_factor + accretion_impact
            temps[zone] = max(50, min(1200, final_temp))  # Keep in reasonable range
            
        return temps
    
    def generate_qrt_temperatures(self, date, zone_temps, base_noise=0.03):
        """Generate QRT temperature data based on zone temperatures"""
        is_maintenance = not self.is_running(date)
        if is_maintenance:
            # During maintenance, no QRT measurements
            return {}
        
        active_accretions = self.accretion_model.get_active_accretions(date)
        
        qrt_temps = {}
        for zone in range(2, 11):  # QRT zones 2-10 only
            # QRT values are generally higher than zone temps
            zone_temp = zone_temps[zone]
            min_qrt, max_qrt = QRT_TEMP_RANGES[zone]
            
            # Check for accretion in this zone or adjacent zones
            accretion_impact = 0
            for accr in active_accretions:
                if accr['zone'] == zone:
                    # Significant drop in QRT temperature when accretion forms
                    accretion_impact -= 250 * accr['severity']  # Up to 250°C drop
                elif abs(accr['zone'] - zone) == 1:
                    accretion_impact -= 150 * accr['severity']  # Up to 150°C drop in adjacent zones
                elif abs(accr['zone'] - zone) == 2:
                    accretion_impact -= 75 * accr['severity']  # Up to 75°C drop in nearby zones
            
            # Base QRT is higher than zone temp
            base_qrt = max(min_qrt, zone_temp + random.uniform(50, 150))
            
            # Apply noise and accretion impact
            noise_factor = random.uniform(1 - base_noise, 1 + base_noise)
            final_temp = base_qrt * noise_factor + accretion_impact
            
            # Ensure temperature stays in reasonable range
            qrt_temps[zone] = max(650, min(1200, final_temp))
            
        return qrt_temps
    
    def generate_shell_temperatures(self, date, zone_temps):
        """Generate shell temperature data for the entire kiln"""
        is_maintenance = not self.is_running(date)
        if is_maintenance:
            # During maintenance, shell is cooling down
            return {pos: {'0': 50, '90': 50, '180': 50, '270': 50, 'AVG': 50} 
                   for pos in KILN_POSITIONS}
        
        shell_temps = {}
        
        # Map kiln positions roughly to zones (for correlation)
        pos_to_zone = {
            'O/L CONE': 10,
            'CONE-10th no.': 10,
            '10th no.-tyre': 9,
            'tyre-9th no.': 9,
            '9th no.': 9,
            '9th no.-8th no.': 8,
            '8th no.': 8,
            '8th no.-7th no.': 7,
            '7th no.': 7,
            '7th no.-6th no.': 6,
            '6th no.': 6,
            '6th no.-5th no.': 5,
            '5th no.': 5,
            '5th no.-tyre': 4,
            'tyre-4th no.': 4,
            '4th no.': 4,
            '4th no.-3rd no.': 3,
            '3rd no.': 3,
            '3rd no.-2nd no.': 2,
            '2nd no.': 2,
            '2nd no.-1st no.': 1,
            '1st no.': 0
        }
        
        active_accretions = self.accretion_model.get_active_accretions(date)
        
        for pos in KILN_POSITIONS:
            zone = pos_to_zone.get(pos, 5)  # Default to zone 5 if mapping is missing
            zone_temp = zone_temps[zone]
            
            # Shell temperature is generally lower than zone temperature
            base_shell_temp = zone_temp * 0.3  # ~30% of internal temperature
              # Check for accretion at this position
            accretion_factor = 1.0
            for accr in active_accretions:
                if accr['zone'] == zone:
                    # Accretion decreases the shell temperature at that location
                    # because it insulates the shell from the internal heat
                    accretion_factor = 1.0 - (accr['severity'] * 0.6)  # Up to 60% lower
                
            # Generate temperatures at different angles with some variation
            angles = {}
            avg_temp = 0
            for angle in ['0', '90', '180', '270']:
                # Add some random variation by angle
                angle_variation = random.uniform(0.9, 1.1)
                temp = base_shell_temp * accretion_factor * angle_variation
                angles[angle] = max(40, min(500, temp))
                avg_temp += angles[angle]
            
            angles['AVG'] = avg_temp / 4
            shell_temps[pos] = angles
            
        return shell_temps
    
    def generate_air_calibration(self, date):
        """Generate air calibration data for fans"""
        is_maintenance = not self.is_running(date)
        if is_maintenance:
            return {fan: {'DAMPER': 0, 'VELOCITY': 0, 'AIR_FLOW': 0} for fan in AIR_FANS}
        
        active_accretions = self.accretion_model.get_active_accretions(date)
        
        air_data = {}
        for fan in AIR_FANS:
            # Map fans to zones they primarily affect
            if fan == 'SAF02':
                zone = 1
            elif fan == 'SAF03':
                zone = 2
            elif fan == 'SAF04':
                zone = 3
            elif fan == 'SAF05':
                zone = 4
            elif fan == 'SAF06':
                zone = 5
            elif fan == 'SAF07':
                zone = 6
            elif fan == 'SAF08':
                zone = 7
            elif fan == 'SAF09':
                zone = 8
            else:  # CB
                zone = 9
            
            # Check for accretion in this zone
            accretion_impact = 0
            for accr in active_accretions:
                if accr['zone'] == zone:
                    # Accretion reduces effective airflow
                    accretion_impact = accr['severity'] * 0.3  # Up to 30% reduction
            
            damper = random.uniform(70, 90)  # Damper setting (%)
            velocity = random.uniform(18, 25)  # Velocity in m/s
            
            # Calculate airflow
            base_airflow = NORMAL_AIRFLOW_BASE + random.uniform(
                -NORMAL_AIRFLOW_VARIANCE, 
                NORMAL_AIRFLOW_VARIANCE
            )
            airflow = base_airflow * (1 - accretion_impact)
            air_data[fan] = {
                'DAMPER': damper,
                'VELOCITY': velocity,
                'AIR_FLOW': airflow
            }
            
        return air_data
        
    def generate_mis_report(self, date):
        """Generate daily MIS report data"""
        is_maintenance = not self.is_running(date)
        
        # Get input values and check for new accretion formation due to material conditions
        inputs = self.input_model.get_current_inputs(date)
        
        # Check if new accretion forms based on material conditions
        # This simulates accretion formation based on material inputs rather than just random events
        if not is_maintenance and random.random() < 0.1:  # 10% chance to check daily
            self.accretion_model.check_for_dynamic_accretion(date, inputs)
        
        # Check for accretion impacts
        active_accretions = self.accretion_model.get_active_accretions(date)
        accretion_severity = sum(accr['severity'] for accr in active_accretions)
        
        # Calculate material quality factors that influence production grades
        # These are derived from the input material characteristics
        iron_quality = inputs.get('IRON_QUALITY', 0.85)
        coal_quality = inputs.get('COAL_QUALITY', 0.82)
        hg_coal_ratio = inputs.get('HG_TO_TOTAL_COAL', 0.7)
        fines_ratio = inputs.get('FINES_RATIO', 0.1)
        fine_particles = inputs.get('FINE_PARTICLES', 25)
        
        # Higher values mean better quality and better Grade A production
        material_quality_factor = (iron_quality * 0.4 + 
                                 coal_quality * 0.3 + 
                                 hg_coal_ratio * 0.2 + 
                                 (1 - fines_ratio) * 0.1)
          # Calculate production impact from accretions
        # More severe accretion means more impact on total production
        production_impact = max(0, min(0.6, accretion_severity * 0.2))  # Up to 60% reduction
        
        # Define base_production here so it's available in all branches
        base_production = 2000  # Base tons per day
        
        if is_maintenance:
            # No production during maintenance
            production_actual = 0
            grade_a = 0
            grade_b = 0
            dri_fines = 0
            dri_dust = 0
            production_plan = 0
            prod_loss = 100  # 100% loss
            kiln_availability = 0
        else:
            # Normal production with possible accretion impact
            production_plan = base_production
            
            # Calculate efficiency considering both accretion impact and material quality
            base_efficiency = random.uniform(0.9, 0.98)
            material_efficiency = base_efficiency * material_quality_factor  # Better materials = better efficiency
            final_efficiency = material_efficiency * (1 - production_impact)  # Accretion reduces efficiency
            
            # Calculate actual production
            production_actual = production_plan * final_efficiency
            
            # Calculate prod loss and availability
            prod_loss = 100 * (1 - final_efficiency)
            kiln_availability = random.uniform(95, 100) if not active_accretions else random.uniform(80, 95)
            
            # Calculate Grade A and Grade B distribution based on accretion and material quality
            # Grade A is premium quality, Grade B is standard quality
            
            # Start with material quality factor (higher is better for Grade A)
            grade_a_fraction = material_quality_factor
            
            # Accretion greatly reduces Grade A and increases Grade B
            # No accretion or early stages: more Grade A
            # Severe accretion: more Grade B
            accretion_quality_impact = sum(accr['severity'] for accr in active_accretions) * 0.5  # Up to 50% shift
            
            # Adjust grade distribution based on accretion severity
            grade_a_fraction = max(0.2, min(0.8, grade_a_fraction - accretion_quality_impact))
            grade_b_fraction = 0.9 - grade_a_fraction  # Grade A + Grade B should be ~90% of production
            
            # Account for particle size - more fine particles means lower Grade A ratio
            fine_particle_factor = min(0.2, (fine_particles - 20) / 100)  # Penalty for excessive fines
            grade_a_fraction -= fine_particle_factor
            grade_b_fraction += fine_particle_factor * 0.5  # Only half goes to Grade B, rest to fines/dust
            
            # Apply some random variation
            grade_a_fraction *= random.uniform(0.9, 1.1)
            grade_b_fraction *= random.uniform(0.9, 1.1)
            
            # Calculate actual tonnage
            grade_a = production_actual * min(0.8, max(0.2, grade_a_fraction))
            grade_b = production_actual * min(0.7, max(0.1, grade_b_fraction))
            
            # Calculate fines and dust (remaining production)
            dri_fines = production_actual * min(0.2, max(0.05, (1 - grade_a_fraction - grade_b_fraction) * 0.7))
            dri_dust = production_actual - (grade_a + grade_b + dri_fines)
          # Adjust input consumption based on production and accretion status
        if is_maintenance:
            for key in inputs:
                inputs[key] = 0
        else:
            # Base efficiency with day-to-day variation
            efficiency_factor = random.uniform(0.95, 1.05)
            
            # Calculate coal consumption increase due to accretion
            coal_increase = 1.0
            if active_accretions:
                # Accretion makes the process less efficient, requiring more coal
                coal_increase = 1.0 + sum(accr['severity'] for accr in active_accretions) * 0.4  # Up to 40% more coal
            
            for key in inputs:
                # Base adjustment for all inputs
                inputs[key] = inputs[key] * efficiency_factor * (production_actual / base_production)
                
                # Additional adjustment for coal inputs when accretion is forming
                if 'COAL' in key and active_accretions:
                    inputs[key] *= coal_increase
        
        # Calculate total values
        total_iron = inputs['PELLETS_CONSUMPTION'] + inputs['IRON_ORE_CONSUMPTION']
        gross_coal = (inputs['HG_COAL_CONSUMPTION'] + inputs['SA_COAL_CONSUMPTION'] + 
                     inputs['ESSAR_FINES'] + inputs['NCL_FINES'] + inputs['WASH_COAL'])
        
        coal_byproducts = gross_coal * random.uniform(0.05, 0.1)  # 5-10% of gross coal
        coal_per_dri = 0 if production_actual == 0 else gross_coal / production_actual
        
        # Generate char data
        char_generation = gross_coal * random.uniform(0.4, 0.5)  # 40-50% of coal becomes char
        plus_6_char = char_generation * random.uniform(0.6, 0.7)
        minus_6_char = char_generation * random.uniform(0.2, 0.25)
        mag_char = char_generation * random.uniform(0.05, 0.1)
        mix_char = char_generation - (plus_6_char + minus_6_char + mag_char)
        
        # Power and steam
        power = random.uniform(800, 950) if not is_maintenance else random.uniform(100, 200)
        total_steam = random.uniform(25, 30) * production_actual / base_production
        avg_steam = total_steam / 24  # Average per hour
          # Loss data
        if is_maintenance:
            feed_loss = 1440  # Total minutes in a day
            feed_loss_reason = "Scheduled maintenance"
            slinger_loss = 1440
            slinger_loss_reason = "Scheduled maintenance"
        else:
            feed_loss = random.randint(0, 180)  # Up to 3 hours of feed loss
            feed_loss_reason = "Normal operation" if feed_loss < 60 else random.choice([
                "Material bridging", "Feeder malfunction", "Raw material shortage", "Power interruption"
            ])
            slinger_loss = random.randint(0, 120)  # Up to 2 hours of slinger loss
            slinger_loss_reason = "Normal operation" if slinger_loss < 30 else random.choice([
                "Mechanical failure", "Alignment issues", "Material overload", "Power interruption"
            ])
        return {
            'DATE': date.strftime('%Y-%m-%d'),
            'CAMP_DAY': (date - START_DATE).days + 1,
            'PRODUCTION ACTUAL': production_actual,  # Total production
            'GRADE_A': grade_a,  # Premium quality product
            'GRADE_B': grade_b,  # Standard quality product
            'DRI_FINES': dri_fines,
            'DRI_DUST': dri_dust,
            'PRODUCTION PLAN': production_plan,  # Changed to match visualization code
            'PROD_LOSS': prod_loss,
            'PELLETS_CONSUMPTION': inputs['PELLETS_CONSUMPTION'],
            'IRON ORE CONSUMPTION': inputs['IRON_ORE_CONSUMPTION'],  # Changed to match visualization code
            'TOTAL_IRON_ORE_PELLETS': total_iron,
            'HG_COAL_CONSUMPTION': inputs['HG_COAL_CONSUMPTION'],
            'SA_COAL_CONSUMPTION': inputs['SA_COAL_CONSUMPTION'],
            'ESSAR_FINES': inputs['ESSAR_FINES'],
            'NCL_FINES': inputs['NCL_FINES'],
            'WASH_COAL': inputs['WASH_COAL'],
            'COAL_LOSSES_BYPRODUCTS': coal_byproducts,
            'GROSS COAL CONSUMPTION': gross_coal,  # Changed to match visualization code
            'COAL_PER_TDRI': coal_per_dri,
            'DOLO_CONSUMPTION': inputs['DOLO_CONSUMPTION'],
            'CHAR_GENERATION': char_generation,
            'PLUS_6_CHAR': plus_6_char,
            'MINUS_6_CHAR': minus_6_char,
            'MAG_CHAR': mag_char,
            'MIX_CHAR': mix_char,
            'POWER': power,
            'KILN_AVAILABILITY': kiln_availability,
            'TOTAL_STEAM_FLOW': total_steam,
            'AVERAGE_STEAM': avg_steam,
            'FEED_LOSS_TOTAL': feed_loss,            'FEED_LOSS_REASON': feed_loss_reason,
            'SLINGER_LOSS': slinger_loss,
            'SLINGER_LOSS_REASON': slinger_loss_reason,
            'REMARKS': "Normal operation" if not active_accretions else f"Potential accretion in zone(s) {[a['zone'] for a in active_accretions]}"
        }
        
    def generate_datasets(self):
        """Generate all datasets for the entire time period"""
        print("=============================================")
        print("Starting entire dataset generation process...")
        print("=============================================")
        
        # Ensure output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        total_days = (END_DATE - START_DATE).days
        print(f"Generating MIS Report data for {total_days} days ({START_DATE} to {END_DATE})...")
        
        mis_data = []
        current_date = START_DATE
        day_counter = 0
        
        # Use tqdm if available, otherwise use a simple progress counter
        try:
            from tqdm import tqdm
            date_range = tqdm(range(total_days), desc="Generating MIS Report")
            for _ in date_range:
                mis_record = self.generate_mis_report(current_date)
                mis_data.append(mis_record)
                current_date += datetime.timedelta(days=1)
                if day_counter % 50 == 0:  # Every 50 days
                    date_range.set_description(f"MIS Report: {current_date.strftime('%Y-%m-%d')}")
                day_counter += 1
        except ImportError:
            while current_date < END_DATE:
                if day_counter % 50 == 0:  # Print progress every 50 days
                    print(f"Generating MIS data for day {day_counter}/{total_days}: {current_date.strftime('%Y-%m-%d')}")
                mis_record = self.generate_mis_report(current_date)
                mis_data.append(mis_record)
                current_date += datetime.timedelta(days=1)
                day_counter += 1
        
        df_mis = pd.DataFrame(mis_data)
        df_mis.to_csv(f"{OUTPUT_DIR}/mis_report.csv", index=False)
        print(f"MIS Report data saved: {len(mis_data)} records, {len(df_mis.columns)} columns")
        
        print("Generating Shell Temperature data...")
        shell_temp_data = []
        current_date = START_DATE
        while current_date < END_DATE:
            # Generate zone temps first (needed for shell temps)
            zone_temps = self.generate_zone_temperatures(current_date)
            shell_temps = self.generate_shell_temperatures(current_date, zone_temps)
            
            for pos, angles in shell_temps.items():
                record = {
                    'DATE': current_date.strftime('%Y-%m-%d'),
                    'POSITION': pos
                }
                record.update({f"SHELL_TEMP_{angle}": temp for angle, temp in angles.items()})
                shell_temp_data.append(record)
            
            current_date += datetime.timedelta(days=1)
        
        df_shell = pd.DataFrame(shell_temp_data)
        df_shell.to_csv(f"{OUTPUT_DIR}/shell_temperature.csv", index=False)
        print(f"Shell Temperature data saved: {len(shell_temp_data)} records")
        
        print("Generating Air Calibration data...")
        air_data = []
        current_date = START_DATE
        while current_date < END_DATE:
            air_calibration = self.generate_air_calibration(current_date)
            
            for fan, metrics in air_calibration.items():
                record = {
                    'DATE': current_date.strftime('%Y-%m-%d'),
                    'FAN': fan
                }
                record.update(metrics)
                air_data.append(record)
            
            current_date += datetime.timedelta(days=1)
        
        df_air = pd.DataFrame(air_data)
        df_air.to_csv(f"{OUTPUT_DIR}/air_calibration.csv", index=False)
        print(f"Air Calibration data saved: {len(air_data)} records")
        
        print("Generating QRT Temperature data...")
        qrt_data = []
        current_date = START_DATE
        current_datetime = datetime.datetime.combine(current_date, datetime.time(0, 0))
        end_datetime = datetime.datetime.combine(END_DATE, datetime.time(23, 59))
        
        # QRT readings every 2 hours
        two_hours = datetime.timedelta(hours=2)
        while current_datetime < end_datetime:
            # Skip if during maintenance
            if self.is_running(current_datetime):
                zone_temps = self.generate_zone_temperatures(current_datetime)
                qrt_temps = self.generate_qrt_temperatures(current_datetime, zone_temps)
                
                for zone, temp in qrt_temps.items():
                    record = {
                        'DATETIME': current_datetime.strftime('%Y-%m-%d %H:%M:%S'),
                        'ZONE': zone,
                        'TEMPERATURE': temp
                    }
                    qrt_data.append(record)
            
            current_datetime += two_hours
        
        df_qrt = pd.DataFrame(qrt_data)
        df_qrt.to_csv(f"{OUTPUT_DIR}/qrt_temperature.csv", index=False)
        print(f"QRT Temperature data saved: {len(qrt_data)} records")
        
        print("Generating Zone Temperature data...")
        zone_data = []
        current_date = START_DATE
        current_datetime = datetime.datetime.combine(current_date, datetime.time(0, 0))
        end_datetime = datetime.datetime.combine(END_DATE, datetime.time(23, 59))
        
        # Zone temperature readings every 2 minutes
        two_minutes = datetime.timedelta(minutes=2)
        while current_datetime < end_datetime:
            # Generate at a lower frequency to make the dataset manageable
            # We'll use spline interpolation later to get 2-minute intervals
            if current_datetime.minute % 30 == 0:  # Every 30 minutes
                zone_temps = self.generate_zone_temperatures(current_datetime)
                
                for zone, temp in zone_temps.items():
                    record = {
                        'DATETIME': current_datetime.strftime('%Y-%m-%d %H:%M:%S'),
                        'ZONE': zone,
                        'TEMPERATURE': temp
                    }
                    zone_data.append(record)
            
            current_datetime += datetime.timedelta(minutes=30)
        
        # Convert to DataFrame for easier manipulation
        df_zone_sparse = pd.DataFrame(zone_data)
        
        # Now expand to 2-minute intervals using interpolation
        print("Interpolating Zone Temperature data to 2-minute intervals...")
        
        # Create complete index with 2-minute intervals
        all_times = pd.date_range(start=START_DATE, end=END_DATE, freq='2Min')
        
        # Create a dataframe for each zone, interpolate, then combine
        zone_dfs = []
        for zone in ZONES:
            # Filter data for this zone
            zone_df = df_zone_sparse[df_zone_sparse['ZONE'] == zone].copy()
            zone_df['DATETIME'] = pd.to_datetime(zone_df['DATETIME'])
            zone_df = zone_df.set_index('DATETIME')
            
            # Reindex to all times and interpolate
            zone_df = zone_df.reindex(all_times, method='nearest')
            
            # Add time index as column
            zone_df = zone_df.reset_index()
            zone_df = zone_df.rename(columns={'index': 'DATETIME'})
            
            # Only keep rows where minutes are even (0, 2, 4, etc.)
            zone_df = zone_df[zone_df['DATETIME'].dt.minute % 2 == 0]
            
            # Smoothen the data
            zone_df['TEMPERATURE'] = savgol_filter(zone_df['TEMPERATURE'].values, 11, 3)
            
            zone_dfs.append(zone_df)
          # Combine all zone data
        df_zone_full = pd.concat(zone_dfs)
        df_zone_full['DATETIME'] = df_zone_full['DATETIME'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Reshape the dataframe to have columns 'ZONE_0', 'ZONE_1', etc. for the visualization code
        # First convert datetime back to datetime type for the pivot
        df_zone_full['DATETIME'] = pd.to_datetime(df_zone_full['DATETIME'])
        
        # Pivot the dataframe to have zones as columns
        df_zone_pivoted = df_zone_full.pivot(index='DATETIME', columns='ZONE', values='TEMPERATURE')
        
        # Rename columns to 'ZONE_0', 'ZONE_1', etc.
        df_zone_pivoted.columns = [f'ZONE_{zone}' for zone in df_zone_pivoted.columns]
        
        # Reset index to make DATETIME a column again
        df_zone_pivoted.reset_index(inplace=True)
        
        # Convert datetime back to string
        df_zone_pivoted['DATETIME'] = df_zone_pivoted['DATETIME'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        df_zone_pivoted.to_csv(f"{OUTPUT_DIR}/zone_temperature.csv", index=False)
        print(f"Zone Temperature data saved: {len(df_zone_pivoted)} records")
        
        # Use the pivoted dataframe for the return value
        df_zone_full = df_zone_pivoted
        
        # Generate accretion event log
        print("Generating accretion event log...")
        accretion_events = []
        for i, event in enumerate(self.accretion_model.accretion_events):
            record = {
                'EVENT_ID': i + 1,
                'START_DATE': event['start_date'].strftime('%Y-%m-%d %H:%M:%S'),
                'CRITICAL_DATE': event['critical_date'].strftime('%Y-%m-%d %H:%M:%S'),
                'ZONE': event['zone'],
                'CLEARED_DATE': event['cleared_date'].strftime('%Y-%m-%d %H:%M:%S') if event['cleared_date'] else None,
                'DURATION_DAYS': (event['cleared_date'] - event['start_date']).days if event['cleared_date'] else (END_DATE - event['start_date']).days
            }
            accretion_events.append(record)
        
        df_events = pd.DataFrame(accretion_events)
        df_events.to_csv(f"{OUTPUT_DIR}/accretion_events.csv", index=False)
        print(f"Accretion event log saved: {len(accretion_events)} events")
        
        # Generate a truth table for model evaluation
        print("Generating accretion truth data...")
        truth_data = []
        current_date = START_DATE
        while current_date < END_DATE:
            # Check daily for accretions
            active = self.accretion_model.get_active_accretions(current_date)
            
            record = {
                'DATE': current_date.strftime('%Y-%m-%d'),
                'HAS_ACCRETION': len(active) > 0,
                'ACTIVE_ACCRETION_COUNT': len(active),
                'ZONES_AFFECTED': ','.join(str(a['zone']) for a in active) if active else None,
                'MAX_SEVERITY': max([a['severity'] for a in active], default=0)
            }
            truth_data.append(record)
            current_date += datetime.timedelta(days=1)
        
        df_truth = pd.DataFrame(truth_data)
        df_truth.to_csv(f"{OUTPUT_DIR}/accretion_truth.csv", index=False)
        print(f"Accretion truth data saved: {len(truth_data)} records")
        
        print("Data generation complete!")
        self.generate_summary_plots()
        
        return {
            'mis_report': df_mis,
            'shell_temperature': df_shell,
            'air_calibration': df_air,
            'qrt_temperature': df_qrt,
            'zone_temperature': df_zone_full,
            'accretion_events': df_events,
            'accretion_truth': df_truth
        }
        
        
    def generate_summary_plots(self):
        """Generate summary plots of the synthetic data"""
        os.makedirs(f"{OUTPUT_DIR}/plots", exist_ok=True)
        
        # Plot accretion events timeline
        df_events = pd.read_csv(f"{OUTPUT_DIR}/accretion_events.csv")
        df_events['START_DATE'] = pd.to_datetime(df_events['START_DATE'])
        df_events['CRITICAL_DATE'] = pd.to_datetime(df_events['CRITICAL_DATE'])
        df_events['CLEARED_DATE'] = pd.to_datetime(df_events['CLEARED_DATE'])
        
        plt.figure(figsize=(15, 8))
        for _, event in df_events.iterrows():
            start = event['START_DATE']
            critical = event['CRITICAL_DATE']
            cleared = event['CLEARED_DATE'] if not pd.isna(event['CLEARED_DATE']) else END_DATE
            zone = event['ZONE']
            
            # Plot progression: onset to critical as yellow-orange gradient
            plt.plot([start, critical], [zone, zone], 'y-', linewidth=6, alpha=0.7)
            # Plot critical to cleared as red
            plt.plot([critical, cleared], [zone, zone], 'r-', linewidth=6, alpha=0.7)
            
            # Add dots at key points
            plt.plot(start, zone, 'go', markersize=8)  # Start (green)
            plt.plot(critical, zone, 'yo', markersize=8)  # Critical (yellow)
            if not pd.isna(event['CLEARED_DATE']):
                plt.plot(cleared, zone, 'bo', markersize=8)  # Cleared (blue)
        
        plt.yticks(ZONES)
        plt.grid(True)
        plt.title('Accretion Events Timeline by Zone')
        plt.xlabel('Date')
        plt.ylabel('Zone')
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/plots/accretion_timeline.png")
        plt.close()
        
        # Load zone temperature data to see impact
        df_zone = pd.read_csv(f"{OUTPUT_DIR}/zone_temperature.csv")
        df_zone['DATETIME'] = pd.to_datetime(df_zone['DATETIME'])
        
        # Plot temperature for a few specific events to show impact
        if len(df_events) > 0:
            # Take first event as example
            event = df_events.iloc[0]
            zone = event['ZONE']
            start_date = event['START_DATE']
            critical_date = event['CRITICAL_DATE']
            cleared_date = event['CLEARED_DATE'] if not pd.isna(event['CLEARED_DATE']) else None
            
            # With the new format, we have columns ZONE_0, ZONE_1, etc.
            # Get data from 10 days before to 10 days after the event
            start_window = start_date - pd.Timedelta(days=10)
            end_window = critical_date + pd.Timedelta(days=10)
            
            zone_column = f'ZONE_{zone}'
            zone_window = df_zone[(df_zone['DATETIME'] >= start_window) & 
                                 (df_zone['DATETIME'] <= end_window)].copy()
            
            plt.figure(figsize=(15, 6))
            plt.plot(zone_window['DATETIME'], zone_window[zone_column], 'b-')
            plt.axvline(start_date, color='g', linestyle='--', label='Accretion Starts')
            plt.axvline(critical_date, color='r', linestyle='--', label='Critical Accretion')
            if cleared_date is not None:
                plt.axvline(cleared_date, color='b', linestyle='--', label='Accretion Cleared')
            
            plt.grid(True)
            plt.legend()
            plt.title(f'Temperature in Zone {zone} During Accretion Event')
            plt.xlabel('Date')
            plt.ylabel('Temperature (°C)')
            plt.tight_layout()
            plt.savefig(f"{OUTPUT_DIR}/plots/temperature_during_accretion.png")
            plt.close()
        
        # Plot MIS data to show production impact
        df_mis = pd.read_csv(f"{OUTPUT_DIR}/mis_report.csv")
        df_mis['DATE'] = pd.to_datetime(df_mis['DATE'])
        
        plt.figure(figsize=(15, 6))
        plt.plot(df_mis['DATE'], df_mis['PRODUCTION ACTUAL'], 'b-', label='Actual Production')
        plt.plot(df_mis['DATE'], df_mis['PRODUCTION PLAN'], 'g--', label='Planned Production')
        
        # Add vertical lines for accretion critical points
        for _, event in df_events.iterrows():
            plt.axvline(event['CRITICAL_DATE'], color='r', linestyle=':', alpha=0.5)
        
        plt.grid(True)
        plt.legend()
        plt.title('Production Actual vs. Planned with Accretion Events')
        plt.xlabel('Date')
        plt.ylabel('Production (tons)')
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/plots/production_impact.png")
        plt.close()


if __name__ == "__main__":
    # Create simulator and generate data
    simulator = KilnSimulator()
    simulator.generate_datasets()
    
    print("Synthetic data generation completed. Data saved to the 'synthetic_data' directory.")