"""
RIS Comparison Demo
===================
This script demonstrates and compares different RIS phase-shift strategies:
1. Constant Phase-Shift RIS (frequency-independent)
2. True-Time Delay (TTD) RIS (frequency-dependent)
3. Phase-Aligned RIS (optimized for channel conditions)
4. Random Phase RIS (baseline)
"""

import numpy as np
import matplotlib.pyplot as plt
from MISO_DEMO1 import simulate, params as default_params
import copy


def compare_ris_modes():
    """Compare different RIS phase modes."""
    
    modes = [
        ("identity", "Identity (no phase shift)"),
        ("constant_phase", "Constant Phase-Shift"),
        ("align", "Phase-Aligned (Optimized)"),
        ("ttd", "True-Time Delay (TTD)"),
        ("random", "Random Phase"),
    ]
    
    results = {}
    
    print("=" * 80)
    print("RIS Phase Mode Comparison")
    print("=" * 80)
    print(f"System Configuration:")
    print(f"  BS antennas: {default_params['Nt']}")
    print(f"  RIS elements: {default_params['Nr']}")
    print(f"  OFDM subcarriers: {default_params['ofdm']['K']}")
    print(f"  Carrier frequency: {default_params['fc']/1e9:.2f} GHz")
    print(f"  Subcarrier spacing: {default_params['ofdm']['subcarrier_spacing']/1e3:.1f} kHz")
    print(f"  Direct link: {'Blocked' if default_params['links']['direct']['blocked'] else 'Active'}")
    print("=" * 80)
    
    for mode, description in modes:
        print(f"\nSimulating: {description}")
        sim_params = copy.deepcopy(default_params)
        sim_params["ris_phase_mode"] = mode
        
        # For constant phase mode, set a specific phase
        if mode == "constant_phase":
            sim_params["ris_constant_phase"] = 0.0  # 0 radians
        
        result = simulate(sim_params)
        results[mode] = {
            "result": result,
            "description": description,
        }
        
        # Compute performance metrics
        H_total = result["H_total"]
        power_total = np.mean(np.abs(H_total) ** 2)
        power_direct = np.mean(np.abs(result["H_direct"]) ** 2)
        power_ris = np.mean(np.abs(result["H_ris"]) ** 2)
        
        print(f"  Average channel power: {10*np.log10(power_total):.2f} dB")
        print(f"  Direct contribution: {10*np.log10(power_direct):.2f} dB")
        print(f"  RIS contribution: {10*np.log10(power_ris):.2f} dB")
        print(f"  RIS gain over direct: {10*np.log10(power_total/power_direct):.2f} dB")
    
    # Plot comparison
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    colors = {
        "identity": "gray",
        "constant_phase": "blue",
        "align": "green",
        "ttd": "red",
        "random": "orange",
    }
    
    linestyles = {
        "identity": ":",
        "constant_phase": "-.",
        "align": "-",
        "ttd": "--",
        "random": ":",
    }
    
    for mode, description in modes:
        result = results[mode]["result"]
        freqs_khz = result["freqs"] / 1e3
        H_total = result["H_total"]
        
        # Magnitude plot
        mag_db = 20 * np.log10(np.abs(H_total) + 1e-12)
        axes[0].plot(freqs_khz, mag_db, 
                    label=description, 
                    color=colors[mode], 
                    linestyle=linestyles[mode],
                    linewidth=2 if mode in ["align", "ttd"] else 1.5)
        
        # Phase plot
        phase_deg = np.angle(H_total, deg=True)
        axes[1].plot(freqs_khz, phase_deg,
                    label=description,
                    color=colors[mode],
                    linestyle=linestyles[mode],
                    linewidth=2 if mode in ["align", "ttd"] else 1.5)
    
    axes[0].set_ylabel("Channel Magnitude [dB]")
    axes[0].set_title("Frequency Response Comparison: Different RIS Phase Modes")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='best', fontsize=9)
    
    axes[1].set_xlabel("Frequency Offset [kHz]")
    axes[1].set_ylabel("Channel Phase [deg]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='best', fontsize=9)
    
    plt.tight_layout()
    
    # Plot detailed comparison for TTD vs Constant Phase
    fig2, axes2 = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    for mode in ["constant_phase", "align", "ttd"]:
        if mode in results:
            result = results[mode]["result"]
            freqs_khz = result["freqs"] / 1e3
            
            # Direct channel
            H_direct = result["H_direct"]
            mag_direct = 20 * np.log10(np.abs(H_direct) + 1e-12)
            
            # RIS channel
            H_ris = result["H_ris"]
            mag_ris = 20 * np.log10(np.abs(H_ris) + 1e-12)
            
            # Total channel
            H_total = result["H_total"]
            mag_total = 20 * np.log10(np.abs(H_total) + 1e-12)
            
            axes2[0].plot(freqs_khz, mag_direct, ':', alpha=0.5, color=colors[mode])
            axes2[0].plot(freqs_khz, mag_ris, '--', alpha=0.7, color=colors[mode])
            axes2[0].plot(freqs_khz, mag_total, '-', linewidth=2, 
                         color=colors[mode], label=results[mode]["description"])
            
            # Phase flatness (variance across frequency)
            phase_variance = np.var(np.angle(H_total))
            axes2[1].plot(freqs_khz, np.angle(H_total, deg=True), 
                         linewidth=2, color=colors[mode],
                         label=f"{results[mode]['description']} (var={phase_variance:.2f})")
    
    axes2[0].set_ylabel("Channel Magnitude [dB]")
    axes2[0].set_title("Detailed Comparison: Constant Phase vs Phase-Aligned vs TTD RIS")
    axes2[0].grid(True, alpha=0.3)
    axes2[0].legend(loc='best')
    
    axes2[1].set_xlabel("Frequency Offset [kHz]")
    axes2[1].set_ylabel("Channel Phase [deg]")
    axes2[1].grid(True, alpha=0.3)
    axes2[1].legend(loc='best', fontsize=9)
    
    plt.tight_layout()
    
    return results


def demonstrate_ris_concepts():
    """Demonstrate key RIS concepts with explanations."""
    
    print("\n" + "=" * 80)
    print("RIS Technical Concepts Demonstration")
    print("=" * 80)
    
    print("\n1. CONSTANT PHASE-SHIFT RIS")
    print("-" * 80)
    print("All RIS elements apply the same phase shift, independent of frequency.")
    print("• Frequency-flat: Same phase across all OFDM subcarriers")
    print("• Simple hardware implementation")
    print("• May cause frequency-selective fading in wideband systems")
    print("• Phase shift: φ = exp(jθ) where θ is constant")
    
    print("\n2. TRUE-TIME DELAY (TTD) RIS")
    print("-" * 80)
    print("Each RIS element introduces a true time delay τ.")
    print("• Frequency-dependent phase: φ(f) = exp(j2πfτ)")
    print("• Maintains beam coherence across wideband signals")
    print("• Better wideband performance than constant phase-shift")
    print("• At carrier fc: φ = exp(j2πfcτ)")
    print("• At subcarrier fc+Δf: φ = exp(j2π(fc+Δf)τ)")
    
    print("\n3. PHASE-ALIGNED RIS")
    print("-" * 80)
    print("Optimizes phases to maximize received signal strength.")
    print("• Uses Channel State Information (CSI)")
    print("• Phase alignment: φ = exp(-j∠(h_ru ⊙ H_br·f_tx))")
    print("• Maximizes coherent combining at receiver")
    print("• Can align with direct path for constructive interference")
    
    print("\n4. RICE FADING CHANNEL MODEL")
    print("-" * 80)
    print("Combines Line-of-Sight (LOS) and scattered (Rayleigh) components.")
    print("• Channel: h = h_LOS + h_scattered")
    print("• K-factor (dB): Ratio of LOS power to scattered power")
    print("• High K → Strong LOS, low fading")
    print("• Low K → Weak LOS, more fading")
    print("• h_LOS = √(K/(K+1)) · e^(jθ)")
    print("• h_scattered = √(1/(K+1)) · (complex Gaussian)")
    
    print("\n5. WIDEBAND OFDM TRANSMISSION")
    print("-" * 80)
    print("Orthogonal Frequency Division Multiplexing for wideband signals.")
    print("• Multiple parallel narrowband subcarriers")
    print(f"• Subcarriers: K = {default_params['ofdm']['K']}")
    print(f"• Spacing: Δf = {default_params['ofdm']['subcarrier_spacing']/1e3} kHz")
    print(f"• Total bandwidth: {default_params['ofdm']['K'] * default_params['ofdm']['subcarrier_spacing']/1e6:.2f} MHz")
    print("• Each subcarrier experiences frequency-selective channel")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Print technical concepts
    demonstrate_ris_concepts()
    
    # Run comparison
    print("\n\nRunning RIS mode comparison...")
    results = compare_ris_modes()
    
    print("\n" + "=" * 80)
    print("Comparison complete! Displaying plots...")
    print("=" * 80)
    plt.show()
