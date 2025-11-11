"""
CSI Imperfections Module for NOMA System
========================================
Implements various CSI imperfections:
- Estimation error (Gaussian noise)
- Quantization (limited feedback bits: 4, 6, 8 bits)
- Feedback delay (outdated CSI due to user movement)
- Temporal correlation (Jake's model)
"""

import numpy as np

class CSIImperfections:
    """Handle various CSI imperfections in the system"""
    
    def __init__(self, config):
        self.config = config
    
    def add_estimation_error(self, h_estimated):
        """
        Add estimation error (Gaussian noise) to channel estimate
        
        Args:
            h_estimated: Estimated channel coefficients
            
        Returns:
            h_imperfect: Channel with estimation error
        """
        error_variance = self.config.estimation_error_variance
        
        # Generate complex Gaussian error
        error_real = np.random.normal(0, np.sqrt(error_variance/2), h_estimated.shape)
        error_imag = np.random.normal(0, np.sqrt(error_variance/2), h_estimated.shape)
        error = error_real + 1j * error_imag
        
        h_imperfect = h_estimated + error
        
        return h_imperfect
    
    def quantize_channel(self, h_estimated):
        """
        Quantize channel estimate for limited feedback
        
        Args:
            h_estimated: Estimated channel coefficients (complex)
            
        Returns:
            h_quantized: Quantized channel coefficients
        """
        num_bits = self.config.quantization_bits
        num_levels = 2**num_bits
        
        # Separate magnitude and phase
        magnitude = np.abs(h_estimated)
        phase = np.angle(h_estimated)
        
        # Normalize magnitude to [0, 1] range
        max_mag = np.max(magnitude) if np.max(magnitude) > 0 else 1.0
        magnitude_norm = magnitude / max_mag
        
        # Quantize magnitude
        magnitude_quantized = np.round(magnitude_norm * (num_levels - 1)) / (num_levels - 1)
        magnitude_quantized = magnitude_quantized * max_mag
        
        # Quantize phase to [0, 2π]
        phase_quantized = np.round(phase / (2 * np.pi) * num_levels) * (2 * np.pi / num_levels)
        
        # Reconstruct complex channel
        h_quantized = magnitude_quantized * np.exp(1j * phase_quantized)
        
        return h_quantized
    
    def apply_feedback_delay(self, h_estimated):
        """
        Apply feedback delay (use outdated CSI)
        
        Args:
            h_estimated: Current channel estimate
            
        Returns:
            h_delayed: Outdated channel estimate
        """
        delay = self.config.feedback_delay
        
        if delay == 0:
            return h_estimated
        
        # Shift channel by delay samples
        h_delayed = np.zeros_like(h_estimated)
        
        if delay < len(h_estimated):
            # Use delayed version
            h_delayed[delay:] = h_estimated[:-delay]
            # Pad beginning with first value
            h_delayed[:delay] = h_estimated[0]
        else:
            # Delay too large, use first value
            h_delayed[:] = h_estimated[0]
        
        return h_delayed
    
    def apply_temporal_correlation_error(self, h_estimated, true_channel_prev):
        """
        Model temporal correlation error (channel changes between estimation and use)
        
        This models the fact that even with perfect estimation, the channel
        may have changed by the time it's used due to user mobility.
        
        Args:
            h_estimated: Estimated channel
            true_channel_prev: Previous true channel (for correlation)
            
        Returns:
            h_correlated: Channel with temporal correlation error
        """
        if not self.config.temporal_correlation or self.config.max_doppler is None:
            return h_estimated
        
        # Use average Doppler (or could be per-user)
        avg_doppler = np.mean(self.config.max_doppler)
        T_s = self.config.sampling_time
        
        # Temporal correlation coefficient (Jakes model)
        from scipy.special import jv
        rho = jv(0, 2 * np.pi * avg_doppler * T_s)
        
        # Model: current channel = rho * previous + innovation
        # Estimated channel has correlation error
        if true_channel_prev is not None:
            # Add correlation-based error
            correlation_error = (1 - rho) * (h_estimated - true_channel_prev)
            h_correlated = h_estimated - correlation_error
        else:
            h_correlated = h_estimated
        
        return h_correlated
    
    def apply_all_imperfections(self, h_estimated, true_channel_prev=None):
        """
        Apply all CSI imperfections in sequence
        
        Order:
        1. Estimation error
        2. Quantization
        3. Feedback delay
        4. Temporal correlation (if applicable)
        
        Args:
            h_estimated: Estimated channel coefficients
            true_channel_prev: Previous true channel (for temporal correlation)
            
        Returns:
            h_imperfect: Channel with all imperfections applied
        """
        h_imperfect = h_estimated.copy()
        
        # 1. Estimation error
        if self.config.estimation_error_variance > 0:
            h_imperfect = self.add_estimation_error(h_imperfect)
        
        # 2. Quantization
        if self.config.quantization_bits < 16:  # Assume 16+ bits is effectively unquantized
            h_imperfect = self.quantize_channel(h_imperfect)
        
        # 3. Feedback delay
        if self.config.feedback_delay > 0:
            h_imperfect = self.apply_feedback_delay(h_imperfect)
        
        # 4. Temporal correlation error
        if self.config.temporal_correlation and true_channel_prev is not None:
            h_imperfect = self.apply_temporal_correlation_error(h_imperfect, true_channel_prev)
        
        return h_imperfect

