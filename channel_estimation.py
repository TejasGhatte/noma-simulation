"""
Channel Estimation Algorithms for NOMA System
=============================================
Implements various channel estimation methods:
- LS (Least Squares) - Simple, fast, noise-sensitive
- MMSE (Minimum Mean Square Error) - Better in low SNR
- DFT-based - Exploits channel sparsity
- Pilot design (uniform spacing vs block)
"""

import numpy as np

class ChannelEstimator:
    """Channel estimation algorithms for CSI acquisition"""
    
    def __init__(self, config):
        self.config = config
        self.pilot_positions = None
        self.pilot_symbols = None
        
    def design_pilots(self, num_symbols):
        """
        Design pilot symbols and positions
        
        Args:
            num_symbols: Total number of symbols (pilot + data)
            
        Returns:
            pilot_positions: Indices where pilots are placed
            pilot_symbols: Pilot symbol values
        """
        num_pilots = int(num_symbols * self.config.pilot_ratio)
        
        if self.config.pilot_pattern == 'uniform':
            # Uniform spacing: pilots evenly distributed
            pilot_spacing = num_symbols // num_pilots
            self.pilot_positions = np.arange(0, num_symbols, pilot_spacing)[:num_pilots]
        else:
            # Block pattern: pilots grouped together at the beginning
            self.pilot_positions = np.arange(num_pilots)
        
        # Generate pilot symbols (known at receiver)
        # Use QPSK-like pilots: +1, -1, +j, -j
        pilot_values = np.array([1, -1, 1j, -1j])
        self.pilot_symbols = np.tile(pilot_values, (num_pilots // 4 + 1))[:num_pilots]
        
        return self.pilot_positions, self.pilot_symbols
    
    def ls_estimation(self, received_pilots, pilot_symbols):
        """
        Least Squares (LS) channel estimation
        
        Formula: h_hat = (X^H * X)^(-1) * X^H * y = X^(-1) * y
        where X is diagonal matrix of pilot symbols, y is received signal
        
        Args:
            received_pilots: Received signal at pilot positions
            pilot_symbols: Transmitted pilot symbols
            
        Returns:
            h_est: Estimated channel coefficients
        """
        # For diagonal pilot matrix: h_est = y / x (element-wise division)
        h_est = received_pilots / (pilot_symbols + 1e-10)  # Avoid division by zero
        return h_est
    
    def mmse_estimation(self, received_pilots, pilot_symbols, true_channel=None):
        """
        Minimum Mean Square Error (MMSE) channel estimation
        
        Formula: h_hat = R_hh * (R_hh + sigma_n^2 * I)^(-1) * h_LS
        where R_hh is channel covariance matrix
        
        Args:
            received_pilots: Received signal at pilot positions
            pilot_symbols: Transmitted pilot symbols
            true_channel: True channel (for computing covariance, optional)
            
        Returns:
            h_est: Estimated channel coefficients
        """
        # First get LS estimate
        h_ls = self.ls_estimation(received_pilots, pilot_symbols)
        
        # Noise variance (from SNR)
        sigma_n_sq = 1.0 / self.config.pilot_snr_linear
        
        # Channel covariance (assume unit variance for simplicity)
        # In practice, this would be estimated from channel statistics
        R_hh = 1.0  # Channel variance
        
        # MMSE filter coefficient
        mmse_gain = R_hh / (R_hh + sigma_n_sq)
        
        # Apply MMSE filter
        h_est = mmse_gain * h_ls
        
        return h_est
    
    def dft_based_estimation(self, received_pilots, pilot_symbols, num_subcarriers=None):
        """
        DFT-based channel estimation (exploits channel sparsity in delay domain)
        
        Steps:
        1. Get LS estimate at pilot positions
        2. Interpolate to all positions (zero-padding in frequency)
        3. Transform to time domain (IDFT)
        4. Keep only significant taps (sparse)
        5. Transform back to frequency domain (DFT)
        
        Args:
            received_pilots: Received signal at pilot positions
            pilot_symbols: Transmitted pilot symbols
            num_subcarriers: Number of subcarriers (for OFDM), None for flat fading
            
        Returns:
            h_est: Estimated channel coefficients
        """
        # Step 1: LS estimation at pilot positions
        h_ls_pilots = self.ls_estimation(received_pilots, pilot_symbols)
        
        if num_subcarriers is None:
            # Flat fading: return LS estimate directly
            return h_ls_pilots
        
        # Step 2: Interpolate to all subcarriers (zero-padding)
        h_freq = np.zeros(num_subcarriers, dtype=complex)
        h_freq[self.pilot_positions] = h_ls_pilots
        
        # Step 3: Transform to time domain
        h_time = np.fft.ifft(h_freq)
        
        # Step 4: Keep only significant taps (assume 6 taps for multipath)
        num_taps = 6
        h_time[num_taps:] = 0  # Zero out insignificant taps
        
        # Step 5: Transform back to frequency domain
        h_est = np.fft.fft(h_time)
        
        return h_est
    
    def estimate_channel(self, true_channel, noise_power=None):
        """
        Perform channel estimation using configured method
        
        Args:
            true_channel: True channel coefficients (complex array)
            noise_power: Noise power (if None, calculated from SNR)
            
        Returns:
            h_estimated: Estimated channel coefficients
            received_pilots: Received signal at pilot positions
        """
        num_samples = len(true_channel)
        
        # Design pilots
        pilot_pos, pilot_syms = self.design_pilots(num_samples)
        num_pilots = len(pilot_pos)
        
        # Extract channel at pilot positions
        h_true_pilots = true_channel[pilot_pos]
        
        # Generate received signal: y = h * x + n
        if noise_power is None:
            noise_power = 1.0 / self.config.pilot_snr_linear
        
        noise = np.random.normal(0, np.sqrt(noise_power/2), num_pilots) + \
                1j * np.random.normal(0, np.sqrt(noise_power/2), num_pilots)
        
        received_pilots = h_true_pilots * pilot_syms + noise
        
        # Perform estimation based on method
        if self.config.estimation_method == 'LS':
            h_est_pilots = self.ls_estimation(received_pilots, pilot_syms)
        elif self.config.estimation_method == 'MMSE':
            h_est_pilots = self.mmse_estimation(received_pilots, pilot_syms, h_true_pilots)
        elif self.config.estimation_method == 'DFT':
            h_est_pilots = self.dft_based_estimation(received_pilots, pilot_syms, 
                                                      self.config.num_subcarriers if 
                                                      self.config.channel_type == 'frequency_selective' else None)
        else:
            raise ValueError(f"Unknown estimation method: {self.config.estimation_method}")
        
        # Interpolate to all positions (simple linear interpolation for non-DFT methods)
        if self.config.estimation_method != 'DFT' or self.config.channel_type != 'frequency_selective':
            h_estimated = np.zeros(num_samples, dtype=complex)
            h_estimated[pilot_pos] = h_est_pilots
            
            # Linear interpolation for non-pilot positions
            for i in range(num_samples):
                if i not in pilot_pos:
                    # Find nearest pilots
                    left_pilot = pilot_pos[pilot_pos <= i]
                    right_pilot = pilot_pos[pilot_pos > i]
                    
                    if len(left_pilot) > 0 and len(right_pilot) > 0:
                        left_idx = left_pilot[-1]
                        right_idx = right_pilot[0]
                        left_val = h_est_pilots[np.where(pilot_pos == left_idx)[0][0]]
                        right_val = h_est_pilots[np.where(pilot_pos == right_idx)[0][0]]
                        
                        # Linear interpolation
                        alpha = (i - left_idx) / (right_idx - left_idx)
                        h_estimated[i] = (1 - alpha) * left_val + alpha * right_val
                    elif len(left_pilot) > 0:
                        # Use last pilot
                        left_idx = left_pilot[-1]
                        h_estimated[i] = h_est_pilots[np.where(pilot_pos == left_idx)[0][0]]
                    else:
                        # Use first pilot
                        right_idx = right_pilot[0]
                        h_estimated[i] = h_est_pilots[np.where(pilot_pos == right_idx)[0][0]]
        else:
            # DFT method already returns full estimate
            h_estimated = h_est_pilots
        
        return h_estimated, received_pilots

