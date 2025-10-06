"""
Complete Example: Understanding RIS Behavior Step-by-Step
==========================================================
This script walks through the RIS simulation step by step,
explaining each concept and showing visual results.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures

import numpy as np
import matplotlib.pyplot as plt
from MISO_DEMO1 import simulate, params, generate_channels, configure_ris_phase, ula_steering
from ris_models import create_ris_model
import copy


def step1_understand_rice_fading():
    """Step 1: Understand Rice Fading Channel Model."""
    print("\n" + "=" * 70)
    print("STEP 1: Understanding Rice Fading Channel")
    print("=" * 70)
    
    print("\nRice fading combines:")
    print("  - Line-of-Sight (LOS) component: Deterministic, predictable")
    print("  - Scattered component: Random, Rayleigh-distributed")
    print("\nK-factor = Power(LOS) / Power(Scattered)")
    
    # Generate channels with different K-factors
    K_factors_dB = [0, 10, 20, 100]  # dB
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for idx, K_dB in enumerate(K_factors_dB):
        p = copy.deepcopy(params)
        p['links']['direct']['K_dB'] = K_dB
        p['links']['direct']['blocked'] = False
        p['ris_phase_mode'] = 'identity'  # No RIS effect
        
        rng = np.random.default_rng(42)
        h_direct, H_br, h_ru, delays = generate_channels(p, rng)
        
        # Plot magnitude across subcarriers
        freqs_khz = ((np.arange(p['ofdm']['K']) - p['ofdm']['K']//2) * 
                     p['ofdm']['subcarrier_spacing'] / 1e3)
        
        mag_db = 20 * np.log10(np.abs(h_direct[:, 0]) + 1e-12)
        axes[idx].plot(freqs_khz, mag_db, linewidth=1.5)
        axes[idx].set_title(f'K-factor = {K_dB} dB')
        axes[idx].set_xlabel('Frequency offset [kHz]')
        axes[idx].set_ylabel('Channel magnitude [dB]')
        axes[idx].grid(True, alpha=0.3)
        
        # Show statistics
        variation = np.std(mag_db)
        axes[idx].text(0.05, 0.95, f'Std: {variation:.2f} dB',
                      transform=axes[idx].transAxes, 
                      verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Effect of K-factor on Channel Fading', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('step1_rice_fading.png', dpi=150, bbox_inches='tight')
    
    print("\nKey observations:")
    print("  - K = 0 dB: Equal LOS and scattered power → High fading variation")
    print("  - K = 10 dB: Strong LOS → Moderate fading")
    print("  - K = 20 dB: Very strong LOS → Little fading")
    print("  - K = 100 dB: Pure LOS → Flat frequency response (no fading)")
    print("\n✓ Saved: step1_rice_fading.png")


def step2_understand_ofdm():
    """Step 2: Understand OFDM Wideband Transmission."""
    print("\n" + "=" * 70)
    print("STEP 2: Understanding OFDM Wideband Transmission")
    print("=" * 70)
    
    print("\nOFDM divides wideband signal into multiple narrowband subcarriers:")
    print(f"  - Number of subcarriers: K = {params['ofdm']['K']}")
    print(f"  - Subcarrier spacing: Δf = {params['ofdm']['subcarrier_spacing']/1e3} kHz")
    bandwidth = params['ofdm']['K'] * params['ofdm']['subcarrier_spacing']
    print(f"  - Total bandwidth: {bandwidth/1e6:.2f} MHz")
    print(f"  - Carrier frequency: fc = {params['fc']/1e9:.2f} GHz")
    
    # Show OFDM structure
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    
    K = params['ofdm']['K']
    delta_f = params['ofdm']['subcarrier_spacing']
    subcarrier_freqs = (np.arange(K) - K//2) * delta_f + params['fc']
    
    # Frequency domain view
    axes[0].stem(subcarrier_freqs/1e9, np.ones(K), basefmt=' ', linefmt='b-', markerfmt='bo')
    axes[0].set_xlabel('Frequency [GHz]')
    axes[0].set_ylabel('Subcarrier')
    axes[0].set_title('OFDM Subcarriers in Frequency Domain')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([params['fc']/1e9 - bandwidth/2e9 - 0.001, 
                      params['fc']/1e9 + bandwidth/2e9 + 0.001])
    
    # Channel frequency response
    p = copy.deepcopy(params)
    p['ris_phase_mode'] = 'align'
    result = simulate(p)
    
    freqs_khz = result['freqs'] / 1e3
    mag_db = 20 * np.log10(np.abs(result['H_total']) + 1e-12)
    
    axes[1].plot(freqs_khz, mag_db, 'g-', linewidth=2)
    axes[1].set_xlabel('Frequency offset from carrier [kHz]')
    axes[1].set_ylabel('Channel magnitude [dB]')
    axes[1].set_title('Frequency-Selective Channel Response')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=np.mean(mag_db), color='r', linestyle='--', 
                   label=f'Average: {np.mean(mag_db):.2f} dB')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('step2_ofdm.png', dpi=150, bbox_inches='tight')
    
    print("\nEach subcarrier experiences different channel conditions!")
    print("This is called frequency-selective fading.")
    print("\n✓ Saved: step2_ofdm.png")


def step3_constant_phase_ris():
    """Step 3: Constant Phase-Shift RIS."""
    print("\n" + "=" * 70)
    print("STEP 3: Constant Phase-Shift RIS")
    print("=" * 70)
    
    print("\nConstant Phase-Shift RIS:")
    print("  - All elements apply the same phase shift")
    print("  - Phase is independent of frequency")
    print("  - Simple hardware implementation")
    print("  - Formula: φ_n = exp(jθ) for all elements n")
    
    # Create constant phase RIS
    ris = create_ris_model('constant_phase', 
                          num_elements=params['Nr'],
                          constant_phase=np.pi/4)  # 45 degrees
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot 1: Phase coefficients
    element_indices = np.arange(params['Nr'])
    phi = ris.get_phase_response(params['fc'])
    axes[0, 0].plot(element_indices, np.angle(phi, deg=True), 'b-', linewidth=2)
    axes[0, 0].set_xlabel('RIS Element Index')
    axes[0, 0].set_ylabel('Phase Shift [deg]')
    axes[0, 0].set_title('Constant Phase: All Elements Have Same Phase')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=45, color='r', linestyle='--', label='45°')
    axes[0, 0].legend()
    
    # Plot 2: Frequency independence
    frequencies = params['fc'] + np.linspace(-1e6, 1e6, 100)
    phase_at_freqs = []
    for f in frequencies:
        phi_f = ris.get_phase_response(f)
        phase_at_freqs.append(np.angle(phi_f[0], deg=True))  # Check first element
    
    axes[0, 1].plot((frequencies - params['fc'])/1e6, phase_at_freqs, 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Frequency offset [MHz]')
    axes[0, 1].set_ylabel('Phase Shift [deg]')
    axes[0, 1].set_title('Frequency Independence: Phase is Constant')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=45, color='r', linestyle='--', label='45°')
    axes[0, 1].legend()
    
    # Plot 3: Channel response with constant phase RIS
    p = copy.deepcopy(params)
    p['ris_phase_mode'] = 'constant_phase'
    p['ris_constant_phase'] = 0.0
    result = simulate(p)
    
    freqs_khz = result['freqs'] / 1e3
    axes[1, 0].plot(freqs_khz, 20*np.log10(np.abs(result['H_direct'])+1e-12), 
                   '--', label='Direct', linewidth=1.5)
    axes[1, 0].plot(freqs_khz, 20*np.log10(np.abs(result['H_ris'])+1e-12), 
                   ':', label='RIS', linewidth=1.5)
    axes[1, 0].plot(freqs_khz, 20*np.log10(np.abs(result['H_total'])+1e-12), 
                   '-', label='Total', linewidth=2)
    axes[1, 0].set_xlabel('Frequency offset [kHz]')
    axes[1, 0].set_ylabel('Channel magnitude [dB]')
    axes[1, 0].set_title('Channel Response with Constant Phase RIS')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Phase response
    axes[1, 1].plot(freqs_khz, np.angle(result['H_total'], deg=True), 
                   'b-', linewidth=2)
    axes[1, 1].set_xlabel('Frequency offset [kHz]')
    axes[1, 1].set_ylabel('Channel phase [deg]')
    axes[1, 1].set_title('Phase Response (may vary due to channel)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('step3_constant_phase.png', dpi=150, bbox_inches='tight')
    
    print("\nConstant phase RIS applies the same shift to all subcarriers.")
    print("The channel itself can still be frequency-selective.")
    print("\n✓ Saved: step3_constant_phase.png")


def step4_ttd_ris():
    """Step 4: True-Time Delay (TTD) RIS."""
    print("\n" + "=" * 70)
    print("STEP 4: True-Time Delay (TTD) RIS")
    print("=" * 70)
    
    print("\nTrue-Time Delay RIS:")
    print("  - Each element introduces a time delay τ")
    print("  - Phase shift depends on frequency: φ(f) = exp(j2πfτ)")
    print("  - Maintains beam coherence across wideband")
    print("  - Better wideband performance")
    
    # Create TTD RIS
    ris = create_ris_model('ttd', 
                          num_elements=params['Nr'],
                          carrier_frequency=params['fc'])
    
    # Set linearly increasing time delays (beam steering effect)
    time_delays = np.linspace(0, 5e-9, params['Nr'])  # 0 to 5 ns
    ris.configure(time_delays=time_delays)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot 1: Time delays
    element_indices = np.arange(params['Nr'])
    axes[0, 0].plot(element_indices, time_delays * 1e9, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('RIS Element Index')
    axes[0, 0].set_ylabel('Time Delay [ns]')
    axes[0, 0].set_title('TTD: Time Delays per Element')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Frequency dependence
    frequencies = params['fc'] + np.linspace(-1e6, 1e6, 100)
    phase_at_freqs = []
    for f in frequencies:
        phi_f = ris.get_phase_response(f)
        phase_at_freqs.append(np.angle(phi_f[-1], deg=True))  # Check last element
    
    axes[0, 1].plot((frequencies - params['fc'])/1e6, phase_at_freqs, 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Frequency offset [MHz]')
    axes[0, 1].set_ylabel('Phase Shift [deg]')
    axes[0, 1].set_title('Frequency Dependence: Phase Changes with Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Channel response with TTD RIS
    p = copy.deepcopy(params)
    p['ris_phase_mode'] = 'ttd'
    result = simulate(p)
    
    freqs_khz = result['freqs'] / 1e3
    axes[1, 0].plot(freqs_khz, 20*np.log10(np.abs(result['H_direct'])+1e-12), 
                   '--', label='Direct', linewidth=1.5)
    axes[1, 0].plot(freqs_khz, 20*np.log10(np.abs(result['H_ris'])+1e-12), 
                   ':', label='RIS', linewidth=1.5)
    axes[1, 0].plot(freqs_khz, 20*np.log10(np.abs(result['H_total'])+1e-12), 
                   '-', label='Total', linewidth=2)
    axes[1, 0].set_xlabel('Frequency offset [kHz]')
    axes[1, 0].set_ylabel('Channel magnitude [dB]')
    axes[1, 0].set_title('Channel Response with TTD RIS')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Compare phases at different frequencies
    K = params['ofdm']['K']
    k_indices = np.array([0, K//4, K//2, 3*K//4, K-1])
    k_labels = ['Edge-', 'Lower', 'Center', 'Upper', 'Edge+']
    
    for i, k in enumerate(k_indices):
        freq = params['fc'] + result['freqs'][k]
        phi_k = ris.get_phase_response(freq)
        axes[1, 1].plot(element_indices[::8], np.angle(phi_k[::8], deg=True), 
                       'o-', label=k_labels[i], alpha=0.7)
    
    axes[1, 1].set_xlabel('RIS Element Index (sampled)')
    axes[1, 1].set_ylabel('Phase Shift [deg]')
    axes[1, 1].set_title('Phase Patterns at Different Subcarriers')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('step4_ttd.png', dpi=150, bbox_inches='tight')
    
    print("\nTTD RIS adapts phase shifts for each subcarrier frequency.")
    print("This maintains beam coherence across the entire bandwidth.")
    print("\n✓ Saved: step4_ttd.png")


def step5_comparison():
    """Step 5: Direct Comparison of All RIS Modes."""
    print("\n" + "=" * 70)
    print("STEP 5: Comparing All RIS Modes")
    print("=" * 70)
    
    modes = [
        ('identity', 'Identity (No RIS)', 'gray'),
        ('constant_phase', 'Constant Phase', 'blue'),
        ('align', 'Phase-Aligned', 'green'),
        ('ttd', 'True-Time Delay', 'red'),
        ('random', 'Random Phase', 'orange'),
    ]
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    print("\nSimulating all modes...")
    for mode, label, color in modes:
        p = copy.deepcopy(params)
        p['ris_phase_mode'] = mode
        if mode == 'constant_phase':
            p['ris_constant_phase'] = 0.0
        
        result = simulate(p)
        freqs_khz = result['freqs'] / 1e3
        
        # Magnitude
        mag_db = 20 * np.log10(np.abs(result['H_total']) + 1e-12)
        axes[0].plot(freqs_khz, mag_db, label=label, color=color, 
                    linewidth=2 if mode in ['align', 'ttd'] else 1.5,
                    alpha=0.9 if mode in ['align', 'ttd'] else 0.7)
        
        # Phase
        phase_deg = np.angle(result['H_total'], deg=True)
        axes[1].plot(freqs_khz, phase_deg, label=label, color=color,
                    linewidth=2 if mode in ['align', 'ttd'] else 1.5,
                    alpha=0.9 if mode in ['align', 'ttd'] else 0.7)
        
        # Print performance
        avg_power = 10 * np.log10(np.mean(np.abs(result['H_total'])**2))
        print(f"  {label:25s}: Avg power = {avg_power:6.2f} dB")
    
    axes[0].set_ylabel('Channel Magnitude [dB]')
    axes[0].set_title('Comparison of RIS Phase Modes: Frequency Response', 
                     fontsize=12, fontweight='bold')
    axes[0].legend(loc='best', fontsize=9)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Frequency Offset [kHz]')
    axes[1].set_ylabel('Channel Phase [deg]')
    axes[1].legend(loc='best', fontsize=9)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('step5_comparison.png', dpi=150, bbox_inches='tight')
    
    print("\n✓ Saved: step5_comparison.png")


def main():
    """Run all steps."""
    print("\n" + "=" * 70)
    print("COMPLETE RIS TUTORIAL: Step-by-Step Understanding")
    print("=" * 70)
    print("\nThis tutorial will generate 5 figures explaining:")
    print("  1. Rice fading channel model")
    print("  2. OFDM wideband transmission")
    print("  3. Constant phase-shift RIS")
    print("  4. True-time delay (TTD) RIS")
    print("  5. Comparison of all RIS modes")
    
    step1_understand_rice_fading()
    step2_understand_ofdm()
    step3_constant_phase_ris()
    step4_ttd_ris()
    step5_comparison()
    
    print("\n" + "=" * 70)
    print("Tutorial complete! All figures saved.")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - step1_rice_fading.png")
    print("  - step2_ofdm.png")
    print("  - step3_constant_phase.png")
    print("  - step4_ttd.png")
    print("  - step5_comparison.png")
    print("\nYou now understand:")
    print("  ✓ Rice fading and K-factor")
    print("  ✓ OFDM wideband transmission")
    print("  ✓ Constant phase-shift RIS (frequency-flat)")
    print("  ✓ True-time delay RIS (frequency-dependent)")
    print("  ✓ Differences between RIS strategies")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
