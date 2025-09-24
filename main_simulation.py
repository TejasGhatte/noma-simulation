"""
Main NOMA System Simulation
===========================
This is the main script that runs the complete NOMA simulation.

SIMULATION FLOW:
1. Load configuration parameters
2. Generate Rayleigh fading channels  
3. Calculate SINR for different power allocation schemes
4. Analyze and visualize results
5. Generate performance report

RUN THIS SCRIPT TO START THE SIMULATION!
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from config import NOMAConfig
from channel_model import NOMAChannelModel
from power_allocation import NOMASystem

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
    
    # Step 2: Generate Channel Model
    print_separator("STEP 2: GENERATING CHANNEL MODEL")
    print("📡 Generating Rayleigh fading channels...")
    
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
    
    # Step 3: NOMA System Analysis
    print_separator("STEP 3: NOMA POWER ALLOCATION ANALYSIS")
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
    
    # Step 4: Generate Performance Report
    generate_performance_report(config, channel_stats, noma_results)
    
    # Step 5: Summary and Next Steps
    print_separator("SIMULATION COMPLETED SUCCESSFULLY")
    print("\n🎉 SIMULATION SUMMARY:")
    print("   ✅ Channel model generated and analyzed")
    print("   ✅ NOMA power allocation schemes compared")
    print("   ✅ Performance metrics calculated")
    print("   ✅ Visualization plots created")
    print("   ✅ Performance report generated")
    
    return {
        'config': config,
        'channel_stats': channel_stats,
        'noma_results': noma_results,
        'channel_model': channel_model,
        'noma_system': noma_system
    }

if __name__ == "__main__":
    # Run the simulation
    results = main()
    
    # Optional: Save results for further analysis
    print("\n💾 Simulation data saved in 'results' dictionary")
    print("   Access results using: results['noma_results']['Fixed']['avg_throughput_user1']")
    print("   Example: Print Fixed scheme User 1 throughput")
    print(f"   → {results['noma_results']['Fixed']['avg_throughput_user1']:.2f} Mbps")