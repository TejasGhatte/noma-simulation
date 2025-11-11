"""
Channel Model for NOMA System
=============================
Implements multiple channel models:
- Rayleigh fading (no line-of-sight)
- Rician fading (with line-of-sight)
- Frequency-selective channels (for OFDM)
- Time-varying channels (user mobility with Jakes model)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from scipy.special import jv  # Bessel function for Jakes model

class NOMAChannelModel:
    """Advanced channel model for NOMA system with multiple fading types"""
    
    def __init__(self, config):
        self.config = config
        self.channel_gains = None
        self.channel_gains_dB = None
        self.channel_coefficients = None  # Complex channel coefficients h
        self.frequency_response = None     # For frequency-selective channels
        
    def generate_rayleigh_fading(self, num_samples):
        """Generate Rayleigh fading channel (no LOS)"""
        # Rayleigh: h = h_real + j*h_imag (both Gaussian)
        h_real = np.random.normal(0, 1/np.sqrt(2), num_samples)
        h_imag = np.random.normal(0, 1/np.sqrt(2), num_samples)
        return h_real + 1j * h_imag
    
    def generate_rician_fading(self, num_samples, k_factor_dB):
        """
        Generate Rician fading channel (with LOS)
        
        Args:
            num_samples: Number of samples
            k_factor_dB: K-factor in dB (ratio of LOS to NLOS power)
        """
        # Convert K-factor from dB to linear
        k_linear = 10**(k_factor_dB / 10)
        
        # LOS component (deterministic)
        los_power = k_linear / (1 + k_linear)
        los_component = np.sqrt(los_power) * np.ones(num_samples, dtype=complex)
        
        # NLOS component (Rayleigh)
        nlos_power = 1 / (1 + k_linear)
        nlos_real = np.random.normal(0, np.sqrt(nlos_power/2), num_samples)
        nlos_imag = np.random.normal(0, np.sqrt(nlos_power/2), num_samples)
        nlos_component = nlos_real + 1j * nlos_imag
        
        return los_component + nlos_component
    
    def generate_time_varying_channel(self, num_samples, user_idx):
        """
        Generate time-varying channel using Jakes model (for user mobility)
        
        Args:
            num_samples: Number of time samples
            user_idx: User index (0 or 1)
        """
        if not self.config.temporal_correlation or self.config.max_doppler is None:
            # Fall back to independent fading
            if self.config.channel_type == 'rician':
                return self.generate_rician_fading(num_samples, self.config.rician_k_factor)
            else:
                return self.generate_rayleigh_fading(num_samples)
        
        f_d = self.config.max_doppler[user_idx]  # Maximum Doppler frequency
        T_s = self.config.sampling_time  # Sampling time
        
        # Jakes model: correlated fading using sum of sinusoids
        # Simplified version using AR(1) model with correlation coefficient
        rho = jv(0, 2 * np.pi * f_d * T_s)  # Temporal correlation coefficient
        
        # Initialize
        h = np.zeros(num_samples, dtype=complex)
        
        if self.config.channel_type == 'rician':
            # Rician with temporal correlation
            k_linear = 10**(self.config.rician_k_factor / 10)
            los_power = k_linear / (1 + k_linear)
            nlos_power = 1 / (1 + k_linear)
            
            # LOS component (constant)
            los_component = np.sqrt(los_power)
            
            # NLOS component (correlated)
            nlos_real = np.random.normal(0, np.sqrt(nlos_power/2))
            nlos_imag = np.random.normal(0, np.sqrt(nlos_power/2))
            h[0] = los_component + (nlos_real + 1j * nlos_imag)
            
            for t in range(1, num_samples):
                # AR(1) model for temporal correlation
                innovation_real = np.random.normal(0, np.sqrt(nlos_power/2 * (1 - rho**2)))
                innovation_imag = np.random.normal(0, np.sqrt(nlos_power/2 * (1 - rho**2)))
                innovation = innovation_real + 1j * innovation_imag
                
                h[t] = los_component + rho * (h[t-1] - los_component) + innovation
        else:
            # Rayleigh with temporal correlation
            h[0] = self.generate_rayleigh_fading(1)[0]
            
            for t in range(1, num_samples):
                innovation = self.generate_rayleigh_fading(1)[0] * np.sqrt(1 - rho**2)
                h[t] = rho * h[t-1] + innovation
        
        return h
    
    def generate_frequency_selective_channel(self, num_samples, user_idx):
        """
        Generate frequency-selective channel for OFDM (multipath)
        
        Args:
            num_samples: Number of OFDM symbols
            user_idx: User index
        """
        num_subcarriers = self.config.num_subcarriers
        num_taps = 6  # Number of multipath taps (typical for urban environment)
        
        # Initialize frequency response
        h_freq = np.zeros((num_samples, num_subcarriers), dtype=complex)
        
        for t in range(num_samples):
            # Generate time-domain channel impulse response (multipath)
            h_time = np.zeros(num_taps, dtype=complex)
            
            # Generate taps with exponential power delay profile
            # For frequency-selective, we can use Rician for first tap if configured
            use_rician_first_tap = (self.config.rician_k_factor > 0)
            
            for tap in range(num_taps):
                if use_rician_first_tap and tap == 0:
                    # First tap has LOS (Rician)
                    h_tap = self.generate_rician_fading(1, self.config.rician_k_factor)[0]
                else:
                    # Other taps are Rayleigh
                    h_tap = self.generate_rayleigh_fading(1)[0]
                
                # Exponential power delay profile
                power_delay = np.exp(-tap / 2.0)
                h_time[tap] = h_tap * np.sqrt(power_delay)
            
            # Convert to frequency domain (FFT)
            h_freq[t, :] = np.fft.fft(h_time, num_subcarriers)
        
        return h_freq
    
    def generate_channels(self):
        """
        Generate channel realizations based on configured channel type
        
        Returns:
            channel_gains: Linear channel gains |h|² for both users
            channel_gains_dB: Channel gains in dB
        """
        num_users = self.config.num_users
        num_samples = self.config.num_time_samples
        
        # Initialize arrays
        self.channel_gains = np.zeros((num_users, num_samples))
        self.channel_coefficients = np.zeros((num_users, num_samples), dtype=complex)
        
        # This will hold the complex channel coefficients *before* path loss
        # We need to apply path loss *after* all channel types are generated
        h_fading = np.zeros((num_users, num_samples), dtype=complex)
        
        # Generate channels based on type
        if self.config.channel_type == 'frequency_selective':
            # OFDM frequency-selective channel
            self.frequency_response = []
            for user in range(num_users):
                h_freq = self.generate_frequency_selective_channel(num_samples, user)
                self.frequency_response.append(h_freq)
                # Store average complex gain
                h_fading[user, :] = np.mean(h_freq, axis=1)
        else:
            # Flat fading channels (Rayleigh or Rician)
            for user in range(num_users):
                if self.config.temporal_correlation:
                    # Time-varying channel with mobility
                    h = self.generate_time_varying_channel(num_samples, user)
                else:
                    # Static or independent fading
                    if self.config.channel_type == 'rician':
                        h = self.generate_rician_fading(num_samples, self.config.rician_k_factor)
                    else:
                        h = self.generate_rayleigh_fading(num_samples)
                
                # Store complex channel fading
                h_fading[user, :] = h
        
        # ===
        # BUG FIX: Apply path loss attenuation directly to channel coefficients
        # Path loss scale factor: sqrt(1/path_loss) for amplitude attenuation
        # This ensures different users have different average channel gains based on distance
        # ===
        # Pre-calculate path losses in dB for verification
        path_losses_dB = [10 * np.log10(pl) for pl in self.config.path_loss]
        
        for user in range(num_users):
            # Get path loss for this user (should be different for each user based on distance)
            path_loss_linear = self.config.path_loss[user]
            
            # Calculate path loss scale factor for amplitude attenuation
            # path_loss_scale = sqrt(1/path_loss) ensures average power is reduced by path_loss
            path_loss_scale = np.sqrt(1.0 / path_loss_linear)
            
            # Apply path loss attenuation to complex channel coefficients
            # Multiply fading by path_loss_scale to attenuate the signal
            self.channel_coefficients[user, :] = h_fading[user, :] * path_loss_scale
            
            # Calculate channel power gains (already includes path loss attenuation)
            self.channel_gains[user, :] = np.abs(self.channel_coefficients[user, :])**2
        
        # Convert to dB for visualization — handle zeros without masking tiny but different values
        eps = np.finfo(float).tiny  # machine tiny (≈1e-308), much smaller than 1e-10
        safe_gains = np.maximum(self.channel_gains, eps)
        self.channel_gains_dB = 10.0 * np.log10(safe_gains)
        
        return self.channel_gains, self.channel_gains_dB
    
    def plot_channels(self, save_plots=True):
        """
        Visualize channel characteristics
        
        Creates 4 plots:
        1. Time series of channel gains
        2. Channel gain distributions
        3. CDF comparison
        4. Channel correlation
        """
        if self.channel_gains is None:
            raise ValueError("Generate channels first using generate_channels()")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('NOMA Channel Model Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Time series
        time_axis = np.arange(self.config.num_time_samples)
        axes[0, 0].plot(time_axis, self.channel_gains_dB[0, :], 'b-', alpha=0.7, label='User 1 (Near)')
        axes[0, 0].plot(time_axis, self.channel_gains_dB[1, :], 'r-', alpha=0.7, label='User 2 (Far)')
        axes[0, 0].set_xlabel('Time Sample')
        axes[0, 0].set_ylabel('Channel Gain (dB)')
        axes[0, 0].set_title('Channel Gain Time Series')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Histograms
        axes[0, 1].hist(self.channel_gains_dB[0, :], bins=50, alpha=0.6, label='User 1', color='blue')
        axes[0, 1].hist(self.channel_gains_dB[1, :], bins=50, alpha=0.6, label='User 2', color='red')
        axes[0, 1].set_xlabel('Channel Gain (dB)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Channel Gain Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: CDF
        for user in range(2):
            sorted_gains = np.sort(self.channel_gains_dB[user, :])
            cdf = np.arange(1, len(sorted_gains) + 1) / len(sorted_gains)
            label = f'User {user+1}'
            axes[1, 0].plot(sorted_gains, cdf, linewidth=2, label=label)
        
        axes[1, 0].set_xlabel('Channel Gain (dB)')
        axes[1, 0].set_ylabel('CDF')
        axes[1, 0].set_title('Cumulative Distribution Function')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Channel quality comparison
        # === RECTIFIED CODE ===
        # Average in linear space first, then convert to dB (mathematically correct)
        avg_gain_user1_linear = np.mean(self.channel_gains[0, :])
        avg_gain_user2_linear = np.mean(self.channel_gains[1, :])

        avg_gain_user1 = 10 * np.log10(avg_gain_user1_linear)
        avg_gain_user2 = 10 * np.log10(avg_gain_user2_linear)
        # === END OF RECTIFICATION ===
        
        users = ['User 1\n(Near)', 'User 2\n(Far)']
        avg_gains = [avg_gain_user1, avg_gain_user2]
        colors = ['blue', 'red']
        
        bars = axes[1, 1].bar(users, avg_gains, color=colors, alpha=0.7)
        axes[1, 1].set_ylabel('Average Channel Gain (dB)')
        axes[1, 1].set_title('Average Channel Quality Comparison')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, avg_gains):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                           f'{value:.1f} dB', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('channel_analysis.png', dpi=300, bbox_inches='tight')
            print("📊 Channel analysis saved as 'channel_analysis.png'")
        
        plt.close(fig)  # Close figure to free memory
        
        return fig
    
    def get_channel_stats(self):
        """
        Calculate and return channel statistics
        
        Returns:
            dict: Channel statistics for analysis
        """
        if self.channel_gains is None:
            raise ValueError("Generate channels first!")
        
        stats = {}
        for user in range(self.config.num_users):
            user_key = f'user_{user+1}'
            
            # === RECTIFIED LOGIC ===
            # Calculate dB of the mean linear gain, not mean of the dB gains
            mean_linear_gain = np.mean(self.channel_gains[user, :])
            mean_dB_gain = 10 * np.log10(mean_linear_gain + 1e-20)
            
            stats[user_key] = {
                'mean_dB': mean_dB_gain, # Use the corrected value
                'std_dB': np.std(self.channel_gains_dB[user, :]),
                'median_dB': np.median(self.channel_gains_dB[user, :]),
                'min_dB': np.min(self.channel_gains_dB[user, :]),
                'max_dB': np.max(self.channel_gains_dB[user, :])
            }
            # === END RECTIFICATION ===
        
        return stats