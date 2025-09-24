"""
Channel Model for NOMA System
=============================
Implements Rayleigh fading channel with path loss for 2-user NOMA system.

Key Concepts:
- Rayleigh fading: Models multipath propagation in urban environments
- Path loss: Signal attenuation due to distance
- Channel State Information (CSI): |h|² values used for power allocation
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

class NOMAChannelModel:
    """Rayleigh fading channel model for NOMA system"""
    
    def __init__(self, config):
        self.config = config
        self.channel_gains = None
        self.channel_gains_dB = None
        
    def generate_channels(self):
        """
        Generate Rayleigh fading channel realizations
        
        Returns:
            channel_gains: Linear channel gains |h|² for both users
            channel_gains_dB: Channel gains in dB
        """
        num_users = self.config.num_users
        num_samples = self.config.num_time_samples
        
        # Initialize arrays
        self.channel_gains = np.zeros((num_users, num_samples))
        
        # Generate Rayleigh fading for each user
        for user in range(num_users):
            # Rayleigh fading: h = h_real + j*h_imag (both Gaussian)
            h_real = np.random.normal(0, 1/np.sqrt(2), num_samples)
            h_imag = np.random.normal(0, 1/np.sqrt(2), num_samples)
            
            # Channel power gain |h|²
            channel_power = h_real**2 + h_imag**2
            
            # Apply path loss
            self.channel_gains[user, :] = channel_power / self.config.path_loss[user]
        
        # Convert to dB for visualization
        self.channel_gains_dB = 10 * np.log10(self.channel_gains)
        
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
        avg_gain_user1 = np.mean(self.channel_gains_dB[0, :])
        avg_gain_user2 = np.mean(self.channel_gains_dB[1, :])
        
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
            stats[user_key] = {
                'mean_dB': np.mean(self.channel_gains_dB[user, :]),
                'std_dB': np.std(self.channel_gains_dB[user, :]),
                'median_dB': np.median(self.channel_gains_dB[user, :]),
                'min_dB': np.min(self.channel_gains_dB[user, :]),
                'max_dB': np.max(self.channel_gains_dB[user, :])
            }
        
        return stats