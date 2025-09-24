"""
NOMA System Configuration File
==============================
This file contains all the input parameters for the NOMA simulation.
Modify these values to test different scenarios.
"""

import numpy as np

class NOMAConfig:
    """Configuration class containing all simulation parameters"""
    
    def __init__(self):
        # ===========================================
        # BASIC SYSTEM PARAMETERS (INPUT)
        # ===========================================
        
        # Simulation Control
        self.num_time_samples = 1000      # Number of channel realizations to generate
        self.monte_carlo_runs = 100       # Number of independent simulation runs
        
        # Physical Layer Parameters
        self.carrier_frequency = 3.5e9    # 5G carrier frequency (Hz) - 3.5 GHz
        self.bandwidth = 20e6             # System bandwidth (Hz) - 20 MHz
        self.total_power = 1.0            # Total transmit power (Watts) - normalized
        self.noise_density = -174         # Noise power density (dBm/Hz)
        
        # User Setup (2-User NOMA)
        self.num_users = 2
        self.user_distances = [100, 500]  # Distance from base station (meters)
                                         # User 1: 100m (near user)
                                         # User 2: 500m (far user)
        
        # Channel Model Parameters
        self.path_loss_exponent = 4.0     # Free space + shadowing
        self.speed_of_light = 3e8         # m/s
        
        # Power Allocation Schemes to Compare
        self.power_schemes = {
            'fixed': [0.4, 0.6],          # [User1_power, User2_power] - NOMA principle
            'equal': [0.5, 0.5],          # Equal power allocation
            'adaptive': 'optimize'         # Will be calculated based on channel
        }
        
        # Calculate derived parameters
        self._calculate_derived_params()
    
    def _calculate_derived_params(self):
        """Calculate parameters derived from basic inputs"""
        
        # Noise power calculation
        self.noise_power_watts = 10**((self.noise_density - 30)/10) * self.bandwidth
        
        # Path loss calculation for each user
        self.path_loss = []
        for distance in self.user_distances:
            # Free space path loss: (4πdf/c)²
            path_loss_linear = (4 * np.pi * distance * self.carrier_frequency / self.speed_of_light)**2
            self.path_loss.append(path_loss_linear)
    
    def get_summary(self):
        """Return a summary of all configuration parameters"""
        summary = f"""
        NOMA System Configuration Summary
        ================================
        
        INPUTS:
        -------
        • System Bandwidth: {self.bandwidth/1e6:.1f} MHz
        • Carrier Frequency: {self.carrier_frequency/1e9:.1f} GHz  
        • Total Power: {self.total_power:.1f} W
        • Number of Users: {self.num_users}
        • User Distances: {self.user_distances} meters
        • Simulation Samples: {self.num_time_samples}
        
        DERIVED PARAMETERS:
        ------------------
        • Noise Power: {10*np.log10(self.noise_power_watts*1000):.1f} dBm
        • User 1 Path Loss: {10*np.log10(self.path_loss[0]):.1f} dB
        • User 2 Path Loss: {10*np.log10(self.path_loss[1]):.1f} dB
        
        POWER ALLOCATION SCHEMES:
        ------------------------
        • Fixed NOMA: User1={self.power_schemes['fixed'][0]*100}%, User2={self.power_schemes['fixed'][1]*100}%
        • Equal Power: User1={self.power_schemes['equal'][0]*100}%, User2={self.power_schemes['equal'][1]*100}%
        • Adaptive: Optimized based on channel conditions
        """
        return summary