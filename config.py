"""
NOMA System Configuration File
==============================
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
        self.path_loss_exponent = 3.0     # Urban environment (reduced from 4.0 for realistic SNR)
        self.speed_of_light = 3e8         # m/s
        
        # CSI Channel Model Parameters
        self.channel_type = 'rayleigh'    # 'rayleigh', 'rician', 'frequency_selective'
        self.rician_k_factor = 3.0        # K-factor for Rician fading (dB) - LOS component strength
        self.user_velocities = [5, 10]    # User velocities (m/s) for time-varying channels
        self.sampling_time = 1e-3         # Sampling time (seconds) for time-varying channels
        
        # OFDM Parameters (for frequency-selective channels)
        self.num_subcarriers = 64         # Number of OFDM subcarriers
        self.cp_length = 16               # Cyclic prefix length
        
        # Channel Estimation Parameters
        self.pilot_ratio = 0.1            # Ratio of pilot symbols to data symbols
        self.estimation_method = 'LS'     # 'LS', 'MMSE', 'DFT'
        self.pilot_pattern = 'uniform'   # 'uniform' or 'block'
        
        # CSI Imperfection Parameters
        self.estimation_error_variance = 0.0  # FIX: Set to 0.0. Error is already simulated.
        self.quantization_bits = 8        # Number of feedback bits (4, 6, 8)
        self.feedback_delay = 0           # Feedback delay in samples
        self.temporal_correlation = True  # Enable temporal correlation (Jakes model)
        self.max_doppler = None           # Maximum Doppler frequency (will be calculated)
        
        # SNR for Channel Estimation
        self.pilot_snr_dB = 20            # SNR for pilot symbols (dB)
        
        # Power Allocation Schemes to Compare
        self.power_schemes = {
            'fixed': [0.7, 0.3],          # [User1_power, User2_power] - NOMA principle
            'equal': [0.5, 0.5],          # Equal power allocation
            'adaptive': 'optimize'         # Will be calculated based on channel
        }
        
        # Calculate derived parameters
        self._calculate_derived_params()
    
    def _calculate_derived_params(self):
        """Calculate parameters derived from basic inputs"""
        
        # Noise power calculation
        # noise_density is in dBm/Hz, convert to W/Hz: 10^((dBm - 30)/10)
        # Then multiply by bandwidth to get total noise power in Watts
        noise_power_density_watts_per_hz = 10**((self.noise_density - 30)/10)  # Convert dBm/Hz to W/Hz
        self.noise_power_watts = noise_power_density_watts_per_hz * self.bandwidth  # Total noise power in W
        
        # Verification: noise power should be very small (around 10^-13 to 10^-14 W for typical values)
        if self.noise_power_watts > 1e-10:
            print(f"⚠️  WARNING: Noise power seems too large: {self.noise_power_watts:.2e} W")
        
        # Path loss calculation for each user
        # ===
        # BUG FIX: The original code used the Friis formula (exponent=2)
        # This fix uses the log-distance model with the configured exponent
        # ===
        self.path_loss = []
        
        # Calculate free space path loss at a reference distance (d0) of 1 meter
        # PL(d_0) = (4 * pi * d_0 * f / c)^2
        pl_0_linear = (4 * np.pi * 1.0 * self.carrier_frequency / self.speed_of_light)**2
        
        for distance in self.user_distances:
            if distance <= 1.0:
                # Use reference path loss
                path_loss_linear = pl_0_linear
            else:
                # Use log-distance path loss model: PL(d) = PL(d_0) * (d/d_0)^n
                path_loss_linear = pl_0_linear * (distance / 1.0)**self.path_loss_exponent
            
            self.path_loss.append(path_loss_linear)
        # === END OF FIX ===
        
        # Calculate maximum Doppler frequency for each user
        if self.temporal_correlation:
            self.max_doppler = []
            for velocity in self.user_velocities:
                f_d = (velocity * self.carrier_frequency) / self.speed_of_light
                self.max_doppler.append(f_d)
        
        # Calculate pilot SNR in linear scale
        self.pilot_snr_linear = 10**(self.pilot_snr_dB / 10)
    
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