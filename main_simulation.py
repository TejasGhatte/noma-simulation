"""
Main NOMA System Simulation with Complete CSI Implementation
============================================================

SIMULATION FLOW:
1. Load configuration parameters
2. Generate channels (Rayleigh/Rician/Frequency-selective/Time-varying)
3. Channel Estimation (LS/MMSE/DFT)
4. Apply CSI Imperfections (estimation error, quantization, delay, correlation)
5. Calculate CSI Quality Metrics (MSE/NMSE, EVM, correlation, capacity loss)
6. Calculate SINR for different power allocation schemes
7. Analyze and visualize results
8. Generate comprehensive performance report
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from config import NOMAConfig
from channel_model import NOMAChannelModel
from power_allocation import NOMASystem
from csi_analysis import CSIAnalyzer

def print_separator(title):
    """Print a nice separator with title"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def calculate_fairness_index(throughput_user1, throughput_user2):
    """Calculate Jain's fairness index"""
    total = throughput_user1 + throughput_user2
    sum_squares = throughput_user1**2 + throughput_user2**2
    
    if sum_squares == 0:
        return 0
    
    fairness = (total**2) / (2 * sum_squares)
    return fairness

def generate_performance_report(config, channel_stats, noma_results):
    """Generate a comprehensive performance report"""
    
    print_separator("NOMA SYSTEM PERFORMANCE REPORT")
    
    # System Configuration Summary
    print("\n🔧 SYSTEM CONFIGURATION:")
    print(f"   • Bandwidth: {config.bandwidth/1e6:.0f} MHz")
    print(f"   • Total Power: {config.total_power:.1f} W") 
    print(f"   • User Distances: {config.user_distances} m")
    print(f"   • Simulation Samples: {config.num_time_samples}")
    
    # Channel Characteristics
    print("\n📡 CHANNEL CHARACTERISTICS:")
    print(f"   • User 1 Avg Channel Gain: {channel_stats['user_1']['mean_dB']:.1f} dB")
    print(f"   • User 2 Avg Channel Gain: {channel_stats['user_2']['mean_dB']:.1f} dB")
    print(f"   • Channel Gain Difference: {channel_stats['user_1']['mean_dB'] - channel_stats['user_2']['mean_dB']:.1f} dB")
    
    # Performance Analysis
    print("\n📊 PERFORMANCE ANALYSIS:")
    print(f"{'Scheme':<12} {'User1 SINR':<12} {'User2 SINR':<12} {'User1 Tput':<12} {'User2 Tput':<12} {'Total Tput':<12} {'Fairness':<10}")
    print(f"{'='*12} {'='*12} {'='*12} {'='*12} {'='*12} {'='*12} {'='*10}")
    
    for scheme_name, results in noma_results.items():
        fairness = calculate_fairness_index(
            results['avg_throughput_user1'], 
            results['avg_throughput_user2']
        )
        
        print(f"{scheme_name:<12} "
              f"{results['avg_sinr_user1']:<12.1f} "
              f"{results['avg_sinr_user2']:<12.1f} "
              f"{results['avg_throughput_user1']:<12.1f} "
              f"{results['avg_throughput_user2']:<12.1f} "
              f"{results['total_throughput']:<12.1f} "
              f"{fairness:<10.3f}")
    
    # Key Insights
    print("\n💡 KEY INSIGHTS:")
    
    # Find best scheme for different metrics
    best_total_tput = max(noma_results.keys(), 
                         key=lambda x: noma_results[x]['total_throughput'])
    
    fairness_scores = {scheme: calculate_fairness_index(
        results['avg_throughput_user1'], results['avg_throughput_user2']
    ) for scheme, results in noma_results.items()}
    
    best_fairness = max(fairness_scores.keys(), key=lambda x: fairness_scores[x])
    
    print(f"   ✅ Best Total Throughput: {best_total_tput} ({noma_results[best_total_tput]['total_throughput']:.1f} Mbps)")
    print(f"   ✅ Best Fairness: {best_fairness} (Index: {fairness_scores[best_fairness]:.3f})")
    
    # NOMA vs Equal Power Analysis
    if 'Fixed' in noma_results and 'Equal' in noma_results:
        noma_tput = noma_results['Fixed']['total_throughput']
        equal_tput = noma_results['Equal']['total_throughput']
        improvement = ((noma_tput - equal_tput) / equal_tput) * 100
        
        print(f"   📈 NOMA vs Equal Power: {improvement:+.1f}% throughput difference")
    
    # Power Allocation Summary
    print("\n⚡ POWER ALLOCATION SUMMARY:")
    for scheme_name, results in noma_results.items():
        alpha1, alpha2 = results['power_allocation']
        print(f"   • {scheme_name}: User1={alpha1*100:.0f}%, User2={alpha2*100:.0f}%")
def main():
    """Main simulation function"""
    
    print_separator("NOMA SYSTEM SIMULATION STARTING")
    
    # Step 1: Initialize Configuration
    print("\n🔧 Step 1: Loading Configuration...")
    config = NOMAConfig()
    print(config.get_summary())
    
    # === Calculate the true system SNR (P_tx / N_0) in dB ===
    # This is the transmit SNR before path loss
    # Note: Actual received SNR at users will be much lower due to path loss
    # Verify both values are in linear scale (Watts) before division
    if config.total_power <= 0 or config.noise_power_watts <= 0:
        raise ValueError(f"Invalid power values: P_tx={config.total_power} W, N_0={config.noise_power_watts:.2e} W")
    
    snr_linear_system = config.total_power / config.noise_power_watts
    snr_dB_system = 10 * np.log10(snr_linear_system)
    
    # Calculate expected received SNR for each user (accounting for path loss)
    received_snr_user1_dB = snr_dB_system - 10*np.log10(config.path_loss[0])
    received_snr_user2_dB = snr_dB_system - 10*np.log10(config.path_loss[1])
    
    print(f"   • System Transmit SNR (P_tx/N_0): {snr_dB_system:.1f} dB")
    print(f"   • Expected Received SNR - User 1: {received_snr_user1_dB:.1f} dB")
    print(f"   • Expected Received SNR - User 2: {received_snr_user2_dB:.1f} dB")
    print(f"     (Note: These are approximate, actual values depend on fading)")
    
    # Step 2: Generate Channel Model
    print_separator("STEP 2: GENERATING CHANNEL MODEL")
    channel_type_str = config.channel_type.upper()
    if config.temporal_correlation:
        channel_type_str += " (Time-Varying)"
    print(f"📡 Generating {channel_type_str} fading channels...")
    
    channel_model = NOMAChannelModel(config)
    channel_gains, channel_gains_dB = channel_model.generate_channels()
    
    print(f"✅ Generated {config.num_time_samples} channel realizations for {config.num_users} users")
    
    # Get channel statistics
    channel_stats = channel_model.get_channel_stats()
    print(f"📊 User 1 average channel gain: {channel_stats['user_1']['mean_dB']:.1f} dB")
    print(f"📊 User 2 average channel gain: {channel_stats['user_2']['mean_dB']:.1f} dB")
    
    # Visualize channels
    print("📈 Creating channel analysis plots...")
    channel_model.plot_channels(save_plots=True)
    
    # Step 3: CSI Analysis (Channel Estimation, Imperfections, Quality Metrics)
    print_separator("STEP 3: COMPREHENSIVE CSI ANALYSIS")
    print("🔍 Performing channel estimation and quality analysis...")
    print(f"   • Estimation Method: {config.estimation_method}")
    print(f"   • Pilot Ratio: {config.pilot_ratio*100:.1f}%")
    print(f"   • Quantization: {config.quantization_bits} bits")
    print(f"   • Feedback Delay: {config.feedback_delay} samples")
    print(f"   • Estimation Error Variance: {config.estimation_error_variance}")
    
    csi_analyzer = CSIAnalyzer(config, channel_model)
    
    # FIX: Pass the correct system SNR to the analyzer
    csi_analyzer.analyze_csi(snr_dB_system)
    
    print("✅ CSI analysis completed")
    csi_analyzer.print_csi_report()
    
    print("📈 Creating CSI analysis plots...")
    csi_analyzer.plot_csi_analysis(save_plots=True)
    
    # Step 4: NOMA System Analysis
    print_separator("STEP 4: NOMA POWER ALLOCATION ANALYSIS")
    print("⚡ Analyzing power allocation schemes...")
    
    noma_system = NOMASystem(config, channel_gains)
    noma_results = noma_system.evaluate_all_schemes()
    
    print("✅ Evaluated all power allocation schemes:")
    for scheme_name in noma_results.keys():
        total_tput = noma_results[scheme_name]['total_throughput']
        print(f"   • {scheme_name}: {total_tput:.1f} Mbps total throughput")
    
    # Visualize NOMA results
    print("📈 Creating power allocation comparison plots...")
    noma_system.plot_sinr_comparison(save_plots=True)
    
    # Step 5: Generate Performance Report
    generate_performance_report(config, channel_stats, noma_results)
    
    # Step 6: Summary and Conclusions
    print_separator("SIMULATION COMPLETED SUCCESSFULLY")
    print("\n🎉 COMPREHENSIVE SIMULATION SUMMARY:")
    print("   ✅ Channel model generated (Rayleigh/Rician/Frequency-selective/Time-varying)")
    print("   ✅ Channel estimation performed (LS/MMSE/DFT)")
    print("   ✅ CSI imperfections applied (estimation error, quantization, delay, correlation)")
    print("   ✅ CSI quality metrics calculated (MSE/NMSE, EVM, correlation, capacity loss)")
    print("   ✅ NOMA power allocation schemes compared")
    print("   ✅ Performance metrics calculated")
    print("   ✅ Comprehensive visualization plots created")
    print("   ✅ Performance reports generated")
    
    print("\n📊 GENERATED OUTPUT FILES:")
    print("   • channel_analysis.png - Channel model characteristics")
    print("   • csi_analysis.png - Comprehensive CSI analysis")
    print("   • power_allocation_comparison.png - NOMA power allocation comparison")
    
    print("\n💡 KEY CONCLUSIONS:")
    # Get best estimation method from CSI analysis
    if csi_analyzer.quality_metrics:
        methods = list(csi_analyzer.quality_metrics[0].keys())
        for user in range(config.num_users):
            best_method = min(methods, 
                            key=lambda m: csi_analyzer.quality_metrics[user][m]['nmse'])
            best_metrics = csi_analyzer.quality_metrics[user][best_method]
            print(f"   • User {user+1}: Best estimation = {best_method} "
                  f"(NMSE: {best_metrics['nmse_dB']:.2f} dB, "
                  f"Capacity Loss: {best_metrics['capacity_loss_percent']:.2f}%)")
    
    return {
        'config': config,
        'channel_stats': channel_stats,
        'noma_results': noma_results,
        'channel_model': channel_model,
        'noma_system': noma_system,
        'csi_analyzer': csi_analyzer
    }

if __name__ == "__main__":
    # Run the simulation
    results = main()
    
    # Optional: Save results for further analysis
    print("\n💾 Simulation data saved in 'results' dictionary")
    print("   Access results using: results['noma_results']['Fixed']['avg_throughput_user1']")
    print("   Example: Print Fixed scheme User 1 throughput")
    print(f"   → {results['noma_results']['Fixed']['avg_throughput_user1']:.2f} Mbps")