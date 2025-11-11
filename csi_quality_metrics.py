"""
CSI Quality Metrics for NOMA System
===================================
Implements metrics to quantify CSI quality:
- MSE/NMSE (Mean Squared Error / Normalized MSE)
- EVM (Error Vector Magnitude)
- Correlation coefficient
- Capacity loss analysis
"""

import numpy as np

class CSIQualityMetrics:
    """Calculate various metrics to quantify CSI quality"""
    
    def __init__(self, config):
        self.config = config
    
    def calculate_mse(self, h_true, h_estimated):
        """
        Calculate Mean Squared Error (MSE)
        
        MSE = E[|h_true - h_estimated|²]
        
        Args:
            h_true: True channel coefficients
            h_estimated: Estimated channel coefficients
            
        Returns:
            mse: Mean squared error
        """
        error = h_true - h_estimated
        mse = np.mean(np.abs(error)**2)
        return mse
    
    def calculate_nmse(self, h_true, h_estimated):
        """
        Calculate Normalized Mean Squared Error (NMSE)
        
        NMSE = E[|h_true - h_estimated|²] / E[|h_true|²]
        
        Args:
            h_true: True channel coefficients
            h_estimated: Estimated channel coefficients
            
        Returns:
            nmse: Normalized mean squared error
        """
        mse = self.calculate_mse(h_true, h_estimated)
        power_true = np.mean(np.abs(h_true)**2)
        
        if power_true == 0:
            return np.inf
        
        nmse = mse / power_true
        return nmse
    
    def calculate_evm(self, h_true, h_estimated):
        """
        Calculate Error Vector Magnitude (EVM)
        
        EVM = sqrt(E[|h_true - h_estimated|²] / E[|h_true|²]) * 100%
        
        Args:
            h_true: True channel coefficients
            h_estimated: Estimated channel coefficients
            
        Returns:
            evm: Error vector magnitude (percentage)
        """
        nmse = self.calculate_nmse(h_true, h_estimated)
        evm = np.sqrt(nmse) * 100
        return evm
    
    def calculate_correlation(self, h_true, h_estimated):
        """
        Calculate correlation coefficient between true and estimated channel
        
        ρ = E[h_true * h_estimated*] / (sqrt(E[|h_true|²]) * sqrt(E[|h_estimated|²]))
        
        Args:
            h_true: True channel coefficients
            h_estimated: Estimated channel coefficients
            
        Returns:
            correlation: Correlation coefficient (complex, magnitude is typically used)
        """
        # Flatten if multi-dimensional
        h_true_flat = h_true.flatten()
        h_estimated_flat = h_estimated.flatten()
        
        # Calculate correlation
        numerator = np.mean(h_true_flat * np.conj(h_estimated_flat))
        denominator = np.sqrt(np.mean(np.abs(h_true_flat)**2) * np.mean(np.abs(h_estimated_flat)**2))
        
        if denominator == 0:
            return 0.0
        
        correlation = numerator / denominator
        return correlation
    
    def calculate_capacity_loss(self, h_true, h_estimated, snr_dB):
        """
        Calculate capacity loss due to imperfect CSI using effective SNR formula
        
        Correct formula:
        - NMSE (linear) = E[|h_true - h_estimated|²] / E[|h_true|²]
        - Effective SNR = (1 - NMSE) * SNR / (NMSE * SNR + 1)
        - Capacity with perfect CSI: C_perfect = log2(1 + SNR)
        - Capacity with imperfect CSI: C_imperfect = log2(1 + SNR_eff)
        - Capacity loss = C_perfect - C_imperfect
        
        Args:
            h_true: True channel coefficients (already includes path loss)
            h_estimated: Estimated channel coefficients
            snr_dB: Per-user received SNR in dB (after path loss)
            
        Returns:
            capacity_loss: Average capacity loss in bits/s/Hz
            capacity_perfect: Average capacity with perfect CSI
            capacity_imperfect: Average capacity with imperfect CSI
        """
        # Convert SNR from dB to linear
        snr_linear = 10**(snr_dB / 10)
        
        # Calculate NMSE in linear scale (not dB)
        nmse_linear = self.calculate_nmse(h_true, h_estimated)
        
        # Clamp NMSE to valid range [0, 1) to avoid numerical issues
        nmse_linear = max(1e-12, min(0.999999, nmse_linear))
        
        # Calculate effective SNR using the correct formula
        # SNR_eff = (1 - NMSE) * SNR / (NMSE * SNR + 1)
        snr_eff = ((1.0 - nmse_linear) * snr_linear) / (nmse_linear * snr_linear + 1.0)
        
        # For perfect CSI, use the received SNR directly
        # Capacity with perfect CSI: C = log2(1 + SNR_received)
        capacity_perfect = np.log2(1.0 + snr_linear)
        
        # Capacity with imperfect CSI: C = log2(1 + SNR_eff)
        capacity_imperfect = np.log2(1.0 + snr_eff)
        
        # Capacity loss (imperfect cannot exceed perfect)
        capacity_loss = max(0, capacity_perfect - capacity_imperfect)
        avg_capacity_imperfect = min(capacity_perfect, capacity_imperfect)
        
        return capacity_loss, capacity_perfect, avg_capacity_imperfect
    
    def calculate_all_metrics(self, h_true, h_estimated, snr_dB=20):
        """
        Calculate all CSI quality metrics
        
        Args:
            h_true: True channel coefficients
            h_estimated: Estimated channel coefficients
            snr_dB: SNR for capacity calculation
            
        Returns:
            metrics: Dictionary containing all metrics
        """
        metrics = {}
        
        metrics['mse'] = self.calculate_mse(h_true, h_estimated)
        metrics['nmse'] = self.calculate_nmse(h_true, h_estimated)
        metrics['nmse_dB'] = 10 * np.log10(metrics['nmse'] + 1e-10)
        evm_raw = self.calculate_evm(h_true, h_estimated)
        # Clamp EVM to reasonable range (0-1000% for display, though >100% indicates very poor estimation)
        metrics['evm'] = min(1000.0, max(0.0, evm_raw))
        metrics['correlation'] = self.calculate_correlation(h_true, h_estimated)
        metrics['correlation_magnitude'] = np.abs(metrics['correlation'])
        
        capacity_loss, cap_perfect, cap_imperfect = self.calculate_capacity_loss(
            h_true, h_estimated, snr_dB
        )
        metrics['capacity_loss'] = max(0, capacity_loss)  # Capacity loss cannot be negative
        metrics['capacity_perfect'] = cap_perfect
        metrics['capacity_imperfect'] = min(cap_perfect, cap_imperfect)  # Imperfect cannot exceed perfect
        # Clamp capacity loss percentage to reasonable range
        if cap_perfect > 0:
            loss_pct = (metrics['capacity_loss'] / cap_perfect * 100)
            metrics['capacity_loss_percent'] = min(100, max(0, loss_pct))  # Clamp to 0-100%
        else:
            metrics['capacity_loss_percent'] = 0
        
        return metrics
    
    def compare_estimation_methods(self, h_true, estimates_dict, snr_dB=20):
        """
        Compare different estimation methods
        
        Args:
            h_true: True channel coefficients
            estimates_dict: Dictionary of {method_name: h_estimated}
            snr_dB: SNR for capacity calculation
            
        Returns:
            comparison: Dictionary of metrics for each method
        """
        comparison = {}
        
        for method_name, h_est in estimates_dict.items():
            comparison[method_name] = self.calculate_all_metrics(h_true, h_est, snr_dB)
        
        return comparison

