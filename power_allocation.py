"""
NOMA Power Allocation and SINR Calculator
=========================================
Implements different power allocation schemes and calculates SINR for NOMA users.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

class NOMASystem:
    """NOMA system with power allocation and SINR calculation"""
    
    def __init__(self, config, channel_gains):
        self.config = config
        self.channel_gains = channel_gains  # Linear channel gains |h|²
        self.sinr_results = {}
        
    def calculate_sinr_for_scheme(self, power_allocation, scheme_name):
        """
        Calculate SINR for both users given power allocation
        
        Args:
            power_allocation: [alpha_1, alpha_2] where alpha_1 + alpha_2 = 1
            scheme_name: Name of the allocation scheme
            
        Returns:
            dict: SINR values for both users
        """
        alpha_1, alpha_2 = power_allocation
        num_samples = self.config.num_time_samples
        
        # Allocated powers
        P1 = alpha_1 * self.config.total_power  # Power for User 1
        P2 = alpha_2 * self.config.total_power  # Power for User 2
        
        # Initialize SINR arrays
        sinr_user1_linear = np.zeros(num_samples)
        sinr_user2_linear = np.zeros(num_samples)
        
        for t in range(num_samples):
            h1_sq = self.channel_gains[0, t]  # |h1|²
            h2_sq = self.channel_gains[1, t]  # |h2|²
            
            # Determine who is stronger user (better channel)
            if h1_sq >= h2_sq:
                # User 1 is stronger
                strong_user, weak_user = 1, 2
                h_strong, h_weak = h1_sq, h2_sq
                P_strong, P_weak = P1, P2
            else:
                # User 2 is stronger  
                strong_user, weak_user = 2, 1
                h_strong, h_weak = h2_sq, h1_sq
                P_strong, P_weak = P2, P1
            
            # NOMA SINR Calculations:
            
            # Strong user decodes weak user's message first
            # Then decodes its own message (SIC - Successive Interference Cancellation)
            sinr_strong_user = (P_strong * h_strong) / (P_weak * h_strong + self.config.noise_power_watts)
            
            # Weak user decodes its message directly (assumes perfect SIC by strong user)
            sinr_weak_user = (P_weak * h_weak) / self.config.noise_power_watts
            
            # Assign back to correct users
            if strong_user == 1:
                sinr_user1_linear[t] = sinr_strong_user
                sinr_user2_linear[t] = sinr_weak_user
            else:
                sinr_user1_linear[t] = sinr_weak_user
                sinr_user2_linear[t] = sinr_strong_user
        
        # Convert to dB
        sinr_user1_dB = 10 * np.log10(sinr_user1_linear + 1e-10)  # Avoid log(0)
        sinr_user2_dB = 10 * np.log10(sinr_user2_linear + 1e-10)
        
        # Calculate throughput using Shannon formula (simplified)
        throughput_user1 = self.config.bandwidth * np.log2(1 + sinr_user1_linear) / 1e6  # Mbps
        throughput_user2 = self.config.bandwidth * np.log2(1 + sinr_user2_linear) / 1e6  # Mbps
        
        results = {
            'power_allocation': power_allocation,
            'sinr_user1_dB': sinr_user1_dB,
            'sinr_user2_dB': sinr_user2_dB,
            'throughput_user1_Mbps': throughput_user1,
            'throughput_user2_Mbps': throughput_user2,
            'avg_sinr_user1': np.mean(sinr_user1_dB),
            'avg_sinr_user2': np.mean(sinr_user2_dB),
            'avg_throughput_user1': np.mean(throughput_user1),
            'avg_throughput_user2': np.mean(throughput_user2),
            'total_throughput': np.mean(throughput_user1) + np.mean(throughput_user2)
        }
        
        return results
    
    def adaptive_power_allocation(self):
        """
        Simple adaptive power allocation based on average channel conditions
        
        Strategy: Give more power to the user with worse average channel
        """
        avg_h1 = np.mean(self.channel_gains[0, :])
        avg_h2 = np.mean(self.channel_gains[1, :])
        
        if avg_h1 > avg_h2:
            # User 1 has better channel, gets less power
            alpha_1, alpha_2 = 0.3, 0.7
        else:
            # User 2 has better channel, gets less power
            alpha_1, alpha_2 = 0.7, 0.3
        
        return [alpha_1, alpha_2]
    
    def evaluate_all_schemes(self):
        """
        Evaluate all power allocation schemes
        
        Returns:
            dict: Results for all schemes
        """
        results = {}
        
        # Fixed scheme
        results['Fixed'] = self.calculate_sinr_for_scheme(
            self.config.power_schemes['fixed'], 'Fixed'
        )
        
        # Equal scheme
        results['Equal'] = self.calculate_sinr_for_scheme(
            self.config.power_schemes['equal'], 'Equal'
        )
        
        # Adaptive scheme
        adaptive_allocation = self.adaptive_power_allocation()
        results['Adaptive'] = self.calculate_sinr_for_scheme(
            adaptive_allocation, 'Adaptive'
        )
        
        self.sinr_results = results
        return results
    
    def plot_sinr_comparison(self, save_plots=True):
        """Plot SINR comparison across different schemes"""
        if not self.sinr_results:
            raise ValueError("Run evaluate_all_schemes() first!")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('NOMA Power Allocation Schemes Comparison', fontsize=16, fontweight='bold')
        
        schemes = list(self.sinr_results.keys())
        colors = ['blue', 'red', 'green']
        
        # Plot 1: Average SINR comparison
        user1_sinr = [self.sinr_results[scheme]['avg_sinr_user1'] for scheme in schemes]
        user2_sinr = [self.sinr_results[scheme]['avg_sinr_user2'] for scheme in schemes]
        
        x = np.arange(len(schemes))
        width = 0.35
        
        bars1 = axes[0, 0].bar(x - width/2, user1_sinr, width, label='User 1', alpha=0.7, color='blue')
        bars2 = axes[0, 0].bar(x + width/2, user2_sinr, width, label='User 2', alpha=0.7, color='red')
        
        axes[0, 0].set_xlabel('Power Allocation Scheme')
        axes[0, 0].set_ylabel('Average SINR (dB)')
        axes[0, 0].set_title('Average SINR Comparison')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(schemes)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                               f'{height:.1f}', ha='center', va='bottom', fontsize=9)
        
        # Plot 2: Throughput comparison
        user1_tput = [self.sinr_results[scheme]['avg_throughput_user1'] for scheme in schemes]
        user2_tput = [self.sinr_results[scheme]['avg_throughput_user2'] for scheme in schemes]
        
        bars1 = axes[0, 1].bar(x - width/2, user1_tput, width, label='User 1', alpha=0.7, color='blue')
        bars2 = axes[0, 1].bar(x + width/2, user2_tput, width, label='User 2', alpha=0.7, color='red')
        
        axes[0, 1].set_xlabel('Power Allocation Scheme')
        axes[0, 1].set_ylabel('Average Throughput (Mbps)')
        axes[0, 1].set_title('Average Throughput Comparison')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(schemes)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                               f'{height:.1f}', ha='center', va='bottom', fontsize=9)
        
        # Plot 3: SINR CDF for Fixed scheme
        scheme_data = self.sinr_results['Fixed']
        sinr1_sorted = np.sort(scheme_data['sinr_user1_dB'])
        sinr2_sorted = np.sort(scheme_data['sinr_user2_dB'])
        cdf = np.arange(1, len(sinr1_sorted) + 1) / len(sinr1_sorted)
        
        axes[1, 0].plot(sinr1_sorted, cdf, 'b-', linewidth=2, label='User 1')
        axes[1, 0].plot(sinr2_sorted, cdf, 'r-', linewidth=2, label='User 2')
        axes[1, 0].set_xlabel('SINR (dB)')
        axes[1, 0].set_ylabel('CDF')
        axes[1, 0].set_title('SINR CDF - Fixed Scheme')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Total system throughput
        total_tput = [self.sinr_results[scheme]['total_throughput'] for scheme in schemes]
        bars = axes[1, 1].bar(schemes, total_tput, color=colors, alpha=0.7)
        axes[1, 1].set_xlabel('Power Allocation Scheme')
        axes[1, 1].set_ylabel('Total Throughput (Mbps)')
        axes[1, 1].set_title('Total System Throughput')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.2,
                           f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('power_allocation_comparison.png', dpi=300, bbox_inches='tight')
            print("📊 Power allocation analysis saved as 'power_allocation_comparison.png'")
        
        plt.close(fig)  # Close figure to free memory
        
        return fig