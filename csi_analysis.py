"""
Comprehensive CSI Analysis Module for NOMA System
=================================================
Integrates channel estimation, imperfections, and quality metrics
to provide complete CSI analysis for NOMA systems.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from channel_estimation import ChannelEstimator
from csi_imperfections import CSIImperfections
from csi_quality_metrics import CSIQualityMetrics

class CSIAnalyzer:
    """Comprehensive CSI analysis for NOMA system"""
    
    def __init__(self, config, channel_model):
        self.config = config
        self.channel_model = channel_model
        self.estimator = ChannelEstimator(config)
        self.imperfections = CSIImperfections(config)
        self.metrics = CSIQualityMetrics(config)
        
        self.true_channels = None
        self.estimated_channels = {}
        self.imperfect_channels = {}
        self.quality_metrics = {}
    
    def analyze_csi(self, snr_dB_system):
        """
        Perform complete CSI analysis:
        1. Get true channels
        2. Estimate channels using different methods
        3. Apply imperfections
        4. Calculate quality metrics
        
        Args:
            snr_dB_system: System transmit SNR in dB (P_tx / N_0)
        """
        if self.channel_model.channel_coefficients is None:
            raise ValueError("Generate channels first using channel_model.generate_channels()")
        
        self.true_channels = self.channel_model.channel_coefficients
        
        num_users = self.config.num_users
        num_samples = self.config.num_time_samples
        
        # Convert system SNR to linear
        snr_system_linear = 10**(snr_dB_system / 10)
        
        # Analyze for each user
        for user in range(num_users):
            h_true = self.true_channels[user, :]
            
            # Calculate per-user received SNR (after path loss)
            # Received SNR = (P_tx * E[|h|^2]) / N_0 = snr_system * E[|h|^2]
            avg_channel_power = np.mean(np.abs(h_true)**2)
            snr_received_linear = snr_system_linear * avg_channel_power
            snr_received_dB = 10 * np.log10(snr_received_linear + 1e-12)
            
            # Test different estimation methods
            estimation_methods = ['LS', 'MMSE', 'DFT'] if self.config.channel_type == 'frequency_selective' else ['LS', 'MMSE']
            
            user_estimates = {}
            user_imperfect = {}
            user_metrics = {}
            
            for method in estimation_methods:
                # Temporarily change estimation method
                original_method = self.config.estimation_method
                self.config.estimation_method = method
                self.estimator.config.estimation_method = method
                
                # Estimate channel
                h_estimated, _ = self.estimator.estimate_channel(h_true)
                user_estimates[method] = h_estimated
                
                # Apply imperfections
                h_prev = h_true[0] if len(h_true) > 1 else h_true
                h_imperfect = self.imperfections.apply_all_imperfections(h_estimated, h_prev)
                user_imperfect[method] = h_imperfect
                
                # Calculate quality metrics using per-user received SNR
                user_metrics[method] = self.metrics.calculate_all_metrics(
                    h_true, h_imperfect, snr_received_dB
                )
                
                # Restore original method
                self.config.estimation_method = original_method
                self.estimator.config.estimation_method = original_method
            
            self.estimated_channels[user] = user_estimates
            self.imperfect_channels[user] = user_imperfect
            self.quality_metrics[user] = user_metrics
    
    def plot_csi_analysis(self, save_plots=True):
        """Create comprehensive CSI analysis plots"""
        if not self.quality_metrics:
            raise ValueError("Run analyze_csi() first!")
        
        num_users = self.config.num_users
        methods = list(self.quality_metrics[0].keys())
        
        # Create figure with subplots
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        fig.suptitle('Comprehensive CSI Analysis for NOMA System', fontsize=16, fontweight='bold')
        
        # Plot 1: True vs Estimated Channels (Time Series)
        ax1 = fig.add_subplot(gs[0, 0])
        user_idx = 0
        method = methods[0]
        time_axis = np.arange(len(self.true_channels[user_idx, :]))
        
        ax1.plot(time_axis, np.abs(self.true_channels[user_idx, :]), 'b-', 
                linewidth=2, label='True Channel', alpha=0.7)
        ax1.plot(time_axis, np.abs(self.estimated_channels[user_idx][method]), 'r--', 
                linewidth=1.5, label=f'Estimated ({method})', alpha=0.7)
        ax1.plot(time_axis, np.abs(self.imperfect_channels[user_idx][method]), 'g:', 
                linewidth=1.5, label='With Imperfections', alpha=0.7)
        ax1.set_xlabel('Time Sample')
        ax1.set_ylabel('Channel Magnitude')
        ax1.set_title('True vs Estimated Channel (User 1)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: NMSE Comparison
        ax2 = fig.add_subplot(gs[0, 1])
        nmse_data = []
        for method in methods:
            nmse_vals = [self.quality_metrics[u][method]['nmse_dB'] for u in range(num_users)]
            nmse_data.append(nmse_vals)
        
        x = np.arange(len(methods))
        width = 0.35
        for u in range(num_users):
            offset = (u - num_users/2 + 0.5) * width / num_users
            nmse_vals = [self.quality_metrics[u][method]['nmse_dB'] for method in methods]
            ax2.bar(x + offset, nmse_vals, width/num_users, 
                   label=f'User {u+1}', alpha=0.7)
        
        ax2.set_xlabel('Estimation Method')
        ax2.set_ylabel('NMSE (dB)')
        ax2.set_title('Normalized MSE Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(methods)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Plot 5: Capacity Loss
        ax5 = fig.add_subplot(gs[0, 2])
        for u in range(num_users):
            cap_loss = [self.quality_metrics[u][method]['capacity_loss'] for method in methods]
            ax5.bar(x + (u - 0.5) * width/num_users, cap_loss, width/num_users, 
                   label=f'User {u+1}', alpha=0.7)
        
        ax5.set_xlabel('Estimation Method')
        ax5.set_ylabel('Capacity Loss (bits/s/Hz)')
        ax5.set_title('Capacity Loss Due to Imperfect CSI')
        ax5.set_xticks(x)
        ax5.set_xticklabels(methods)
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')
        
        # Plot 6: Capacity Loss Percentage
        ax6 = fig.add_subplot(gs[1, 0])
        for u in range(num_users):
            cap_loss_pct = [self.quality_metrics[u][method]['capacity_loss_percent'] for method in methods]
            ax6.bar(x + (u - 0.5) * width/num_users, cap_loss_pct, width/num_users, 
                   label=f'User {u+1}', alpha=0.7)
        
        ax6.set_xlabel('Estimation Method')
        ax6.set_ylabel('Capacity Loss (%)')
        ax6.set_title('Relative Capacity Loss')
        ax6.set_xticks(x)
        ax6.set_xticklabels(methods)
        ax6.legend()
        ax6.grid(True, alpha=0.3, axis='y')
        
        # Plot 9: Summary Metrics Table
        ax9 = fig.add_subplot(gs[1, 1])
        ax9.axis('off')
        
        # Create summary table
        summary_text = "CSI Quality Summary\n" + "="*30 + "\n\n"
        method = methods[0]  # Use first method for summary
        for u in range(num_users):
            summary_text += f"User {u+1} ({method}):\n"
            metrics = self.quality_metrics[u][method]
            summary_text += f"  NMSE: {metrics['nmse_dB']:.2f} dB\n"
            summary_text += f"  EVM: {metrics['evm']:.2f}%\n"
            summary_text += f"  Correlation: {metrics['correlation_magnitude']:.3f}\n"
            summary_text += f"  Cap. Loss: {metrics['capacity_loss']:.3f} b/s/Hz\n"
            summary_text += f"  Cap. Loss %: {metrics['capacity_loss_percent']:.2f}%\n\n"
        
        ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes,
                fontsize=9, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        if save_plots:
            plt.savefig('csi_analysis.png', dpi=300, bbox_inches='tight')
            print("📊 CSI analysis saved as 'csi_analysis.png'")
        
        plt.close(fig)
        return fig
    
    def print_csi_report(self):
        """Print comprehensive CSI analysis report"""
        if not self.quality_metrics:
            raise ValueError("Run analyze_csi() first!")
        
        print("\n" + "="*70)
        print("  COMPREHENSIVE CSI ANALYSIS REPORT")
        print("="*70)
        
        methods = list(self.quality_metrics[0].keys())
        num_users = self.config.num_users
        
        for user in range(num_users):
            print(f"\n📡 USER {user+1} CSI ANALYSIS:")
            print("-" * 70)
            
            for method in methods:
                metrics = self.quality_metrics[user][method]
                print(f"\n  Estimation Method: {method}")
                print(f"    • NMSE: {metrics['nmse_dB']:.2f} dB")
                print(f"    • EVM: {metrics['evm']:.2f}%")
                print(f"    • Correlation: {metrics['correlation_magnitude']:.3f}")
                print(f"    • Capacity Loss: {metrics['capacity_loss']:.4f} bits/s/Hz ({metrics['capacity_loss_percent']:.2f}%)")
                print(f"    • Perfect CSI Capacity: {metrics['capacity_perfect']:.4f} bits/s/Hz")
                print(f"    • Imperfect CSI Capacity: {metrics['capacity_imperfect']:.4f} bits/s/Hz")
        
        # Find best method
        print("\n" + "="*70)
        print("  BEST ESTIMATION METHOD COMPARISON")
        print("="*70)
        
        for user in range(num_users):
            best_nmse = min(methods, key=lambda m: self.quality_metrics[user][m]['nmse'])
            best_corr = max(methods, key=lambda m: self.quality_metrics[user][m]['correlation_magnitude'])
            best_cap = min(methods, key=lambda m: self.quality_metrics[user][m]['capacity_loss'])
            
            print(f"\n  User {user+1}:")
            print(f"    • Best NMSE: {best_nmse}")
            print(f"    • Best Correlation: {best_corr}")
            print(f"    • Best Capacity (Lowest Loss): {best_cap}")

