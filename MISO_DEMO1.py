# MISO_DEMO1.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import copy

params = {
    "fc": 3.5e9,                 # carrier frequency (Hz)
    "c": 3.0e8,                  # speed of light (m/s)
    "Nt": 8,                     # BS antennas (ULA)
    "Nr": 256,                    # RIS elements (ULA)
    "bs_element_spacing": 0.5,   # spacing in wavelengths at the BS
    "ris_element_spacing": 0.5,  # spacing in wavelengths at the RIS
    "links": {
        "direct": {
            "distance": 50.0,  # BS to UE (m)
            "K_dB": 100.0,        # Rician K-factor (dB) for direct link to more K to more dB
            "aod_deg": -11.0,    # AoD of LOS path at the BS
            "blocked": False,
            "blockage_loss_dB": 0.0,
        },
        "bs_ris": {
            "distance": 60.0,   # BS to RIS (m)
            "K_dB": 100.0,
            "aod_deg": 20.0,   # AoD at the BS towards the RIS
            "aoa_deg": -18.0    # AoA at the RIS from the BS
        },
        "ris_ue": {
            "distance": 30.0,   # RIS to UE (m)
            "K_dB": 100.0,
            "aod_deg": 10.0,    # AoD at the RIS towards the UE
            "aoa_deg": 10.0     # AoA at the UE from the RIS
        }
    },
    "ofdm": {
        "K": 128,                # number of subcarriers
        "subcarrier_spacing": 15e3
    },
    "noise": {
        "snr_dB": 20.0           # SNR per subcarrier (dB)
    },
    "precoder_target_deg": 0.0,
    "ris_phase_mode": "align",
    "rng_seed": 7,
    "tx_power_dBm": 45.0
}
params["precoder_target_deg"] = params["links"]["bs_ris"]["aod_deg"]
params["links"]["direct"]["blocked"] = True  # Partial blockage enabled; set False for clear LOS
params["links"]["direct"]["block_visual"] = {"fraction": 0.45, "width": 14.0, "depth": 6.0}


def db_to_linear(db):
    """Convert dB to linear scale."""
    return 10.0 ** (db / 10.0)


def complex_gaussian(shape, rng):
    """Generate complex Gaussian random variables with unit variance."""
    real = rng.standard_normal(shape)
    imag = rng.standard_normal(shape)
    return (real + 1j * imag) / np.sqrt(2.0)


def free_space_pathloss_amp(distance, wavelength):
    """Compute free-space path loss amplitude (not in dB)."""
    return wavelength / (4.0 * np.pi * distance)


def ula_steering(num_elements, angle_deg, element_spacing_lambda):
    """
    Uniform Linear Array steering vector.
    angle_deg: angle in degrees (broadside is 0)
    element_spacing_lambda: spacing in wavelengths
    """
    angle_rad = np.deg2rad(angle_deg)
    indices = np.arange(num_elements)
    phase_shift = 2.0 * np.pi * element_spacing_lambda * np.sin(angle_rad) * indices
    return np.exp(1j * phase_shift) / np.sqrt(num_elements)


def configure_ris_phase(sim_params, *, H_br=None, h_ru=None, f_tx=None, H_direct=None):
    """
    Configure RIS phase shifts based on the specified mode.
    Supports: 'identity', 'random', 'align', 'constant_phase', 'ttd' (true-time delay)
    """
    Nr = sim_params["Nr"]
    custom = sim_params.get("ris_phase_vector")
    if custom is not None:
        phi = np.asarray(custom, dtype=complex)
        if phi.shape != (Nr,):
            raise ValueError("ris_phase_vector must have length Nr")
        return phi

    mode = sim_params.get("ris_phase_mode", "identity")
    if mode == "random":
        rng = np.random.default_rng(sim_params.get("rng_seed", None))
        phases = rng.uniform(0.0, 2.0 * np.pi, size=Nr)
        return np.exp(1j * phases)

    if mode == "align":
        if H_br is None or h_ru is None or f_tx is None:
            raise ValueError("RIS phase alignment requires channel matrices and BS precoder")
        K = H_br.shape[0]
        k_ref = sim_params.get("ofdm", {}).get("align_reference_subcarrier")
        if k_ref is None:
            k_ref = K // 2
        else:
            k_ref = int(np.clip(k_ref, 0, K - 1))
        incident = H_br[k_ref] @ f_tx
        cascade = h_ru[k_ref] * incident
        epsilon = 1e-12
        if np.all(np.abs(cascade) < epsilon):
            return np.ones(Nr, dtype=complex)
        phases = np.exp(-1j * np.angle(cascade))
        if H_direct is not None:
            direct_ref = H_direct[k_ref]
            if np.abs(direct_ref) >= epsilon:
                phases *= np.exp(1j * np.angle(direct_ref))
        return phases

    if mode == "constant_phase":
        # Constant phase-shift RIS: All elements have the same phase shift
        # This is frequency-independent
        phase_value = sim_params.get("ris_constant_phase", 0.0)
        return np.exp(1j * phase_value) * np.ones(Nr, dtype=complex)
    
    if mode == "ttd":
        # True-Time Delay (TTD) RIS: Introduces frequency-dependent phase shifts
        # This returns the base phase configuration; frequency dependence is handled in simulate()
        if H_br is None or h_ru is None or f_tx is None:
            raise ValueError("TTD RIS mode requires channel matrices and BS precoder")
        K = H_br.shape[0]
        k_ref = sim_params.get("ofdm", {}).get("align_reference_subcarrier")
        if k_ref is None:
            k_ref = K // 2
        else:
            k_ref = int(np.clip(k_ref, 0, K - 1))
        incident = H_br[k_ref] @ f_tx
        cascade = h_ru[k_ref] * incident
        epsilon = 1e-12
        if np.all(np.abs(cascade) < epsilon):
            return np.ones(Nr, dtype=complex)
        # For TTD, we compute time delays instead of just phases
        phases = np.exp(-1j * np.angle(cascade))
        return phases

    return np.ones(Nr, dtype=complex)


def generate_channels(sim_params, rng):
    """
    Generate channel matrices for direct link, BS->RIS, and RIS->UE.
    Uses Rice fading model (LOS + Rayleigh scattering).
    """
    Nt = sim_params["Nt"]
    Nr = sim_params["Nr"]
    K = sim_params["ofdm"]["K"]

    d_direct = sim_params["links"]["direct"]["distance"]
    d_bs_ris = sim_params["links"]["bs_ris"]["distance"]
    d_ris_ue = sim_params["links"]["ris_ue"]["distance"]
    wavelength = sim_params["c"] / sim_params["fc"]
    pl_direct = free_space_pathloss_amp(d_direct, wavelength)
    pl_bs_ris = free_space_pathloss_amp(d_bs_ris, wavelength)
    pl_ris_ue = free_space_pathloss_amp(d_ris_ue, wavelength)


    theta_direct = sim_params["links"]["direct"]["aod_deg"]
    theta_bs_ris = sim_params["links"]["bs_ris"]["aod_deg"]
    psi_bs_ris = sim_params["links"]["bs_ris"]["aoa_deg"]
    theta_ris_ue = sim_params["links"]["ris_ue"]["aod_deg"]

    d_tx = sim_params["bs_element_spacing"]
    d_ris = sim_params["ris_element_spacing"]

    # Direct link (BS -> UE)
    K_direct = db_to_linear(sim_params["links"]["direct"]["K_dB"])
    mu_direct = pl_direct * np.sqrt(K_direct / (K_direct + 1.0)) * ula_steering(Nt, theta_direct, d_tx)
    sigma_direct = pl_direct * np.sqrt(1.0 / (K_direct + 1.0))
    w_direct = complex_gaussian((K, Nt), rng)

    h_direct = mu_direct + sigma_direct * w_direct

    direct_cfg = sim_params["links"]["direct"]
    if direct_cfg.get("blocked", False):
        loss_dB = direct_cfg.get("blockage_loss_dB")
        if loss_dB is None:
            h_direct = np.zeros_like(h_direct)
        else:
            attenuation = 10.0 ** (-loss_dB / 20.0)
            h_direct *= attenuation

    # BS -> RIS matrix
    K_br = db_to_linear(sim_params["links"]["bs_ris"]["K_dB"])
    a_ris_inc = ula_steering(Nr, psi_bs_ris, d_ris)
    a_bs_out = ula_steering(Nt, theta_bs_ris, d_tx)
    mu_br = pl_bs_ris * np.sqrt(K_br / (K_br + 1.0)) * np.outer(a_ris_inc, np.conj(a_bs_out))
    sigma_br = pl_bs_ris * np.sqrt(1.0 / (K_br + 1.0))
    W_br = complex_gaussian((K, Nr, Nt), rng)
    H_br = mu_br[None, :, :] + sigma_br * W_br

    # RIS -> UE row vector
    K_ru = db_to_linear(sim_params["links"]["ris_ue"]["K_dB"])
    a_ris_out = ula_steering(Nr, theta_ris_ue, d_ris)
    mu_ru = pl_ris_ue * np.sqrt(K_ru / (K_ru + 1.0)) * np.conj(a_ris_out)
    sigma_ru = pl_ris_ue * np.sqrt(1.0 / (K_ru + 1.0))
    W_ru = complex_gaussian((K, Nr), rng)
    h_ru = mu_ru + sigma_ru * W_ru

    delays = {
        "direct": d_direct / sim_params["c"],
        "ris": (d_bs_ris + d_ris_ue) / sim_params["c"]
    }

    return h_direct, H_br, h_ru, delays


def simulate(sim_params):
    """
    Main simulation function for wideband OFDM transmission with RIS.
    Supports different RIS phase modes including constant phase-shift and TTD.
    """
    rng = np.random.default_rng(sim_params.get("rng_seed", None))

    f_tx = ula_steering(sim_params["Nt"], sim_params.get("precoder_target_deg", 0.0), sim_params["bs_element_spacing"])
    h_direct, H_br, h_ru, delays = generate_channels(sim_params, rng)

    H_direct = np.einsum("kn,n->k", h_direct, f_tx)
    phi = configure_ris_phase(sim_params, H_br=H_br, h_ru=h_ru, f_tx=f_tx, H_direct=H_direct)

    K = sim_params["ofdm"]["K"]
    delta_f = sim_params["ofdm"]["subcarrier_spacing"]
    freqs = (np.arange(K) - K // 2) * delta_f

    H_ris = np.empty(K, dtype=complex)
    
    mode = sim_params.get("ris_phase_mode", "identity")
    
    if mode == "ttd":
        # True-Time Delay mode: Apply frequency-dependent phase shifts
        fc = sim_params["fc"]
        # Compute the time delays for each RIS element
        # For simplicity, we use the phase at reference frequency to derive delays
        time_delays = -np.angle(phi) / (2.0 * np.pi * fc)
        
        for k in range(K):
            freq_k = fc + freqs[k]
            # Apply frequency-dependent phase shifts based on time delays
            phi_k = np.exp(1j * 2.0 * np.pi * freq_k * time_delays)
            reflected = phi_k * (H_br[k] @ f_tx)
            H_ris[k] = np.dot(h_ru[k], reflected)
    else:
        # Constant phase-shift mode or other modes: frequency-independent
        for k in range(K):
            reflected = phi * (H_br[k] @ f_tx)
            H_ris[k] = np.dot(h_ru[k], reflected)

    H_total = H_direct + H_ris

    snr_dB = sim_params.get("noise", {}).get("snr_dB", 20.0)
    snr_linear = db_to_linear(snr_dB)
    noise_power = np.mean(np.abs(H_total) ** 2) / snr_linear
    
    return {
        "H_direct": H_direct,
        "H_ris": H_ris,
        "H_total": H_total,
        "freqs": freqs,
        "phi": phi,
        "snr_dB": snr_dB,
        "noise_power": noise_power,
        "delays": delays
    }


def simulate_narrowband(sim_params, *, include_scattering=False):
    """Narrowband simulation wrapper."""
    return compute_continuous_channel(sim_params, include_scattering=include_scattering)


def compute_continuous_channel(sim_params, *, include_scattering=False):
    """Compute a continuous-tone (single-frequency) channel snapshot without OFDM machinery."""
    rng = np.random.default_rng(sim_params.get("rng_seed", None))
    
    Nt = sim_params["Nt"]
    Nr = sim_params["Nr"]
    
    # Generate single-frequency channels (K=1)
    temp_params = copy.deepcopy(sim_params)
    temp_params["ofdm"]["K"] = 1
    
    f_tx = ula_steering(Nt, sim_params.get("precoder_target_deg", 0.0), sim_params["bs_element_spacing"])
    h_direct, H_br, h_ru, delays = generate_channels(temp_params, rng)
    
    # Extract single subcarrier
    h_direct = h_direct[0]
    H_br = H_br[0]
    h_ru = h_ru[0]
    
    H_direct = np.dot(h_direct, f_tx)
    
    # Configure RIS phase
    temp_params["ofdm"]["K"] = 1
    phi = configure_ris_phase(temp_params, H_br=H_br[None, :, :], h_ru=h_ru[None, :], 
                               f_tx=f_tx, H_direct=np.array([H_direct]))
    
    incident = H_br @ f_tx
    H_ris = np.dot(h_ru, phi * incident)
    H_total = H_direct + H_ris
    
    return {
        "H_direct": H_direct,
        "H_ris": H_ris,
        "H_total": H_total,
        "phi": phi,
        "incident": incident,
        "f_tx": f_tx,
    }


def plot_frequency_response(freqs_hz, H_direct, H_ris, H_total):
    """Plot magnitude and phase response across subcarriers."""
    freqs_khz = freqs_hz / 1e3
    magnitude_offset = 1e-12

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

    def magnitude_db(H):
        return 20.0 * np.log10(np.maximum(np.abs(H), magnitude_offset))
    
    lines_mag = [
        (H_direct, "Direct", "--"),
        (H_ris, "RIS", ":"),
        (H_total, "Total", "-"),
    ]
    for H, label, style in lines_mag:
        axes[0].plot(freqs_khz, magnitude_db(H), label=label, linestyle=style, linewidth=1.6 if label == "Total" else 1.0)
    axes[0].set_ylabel("|H(f)| [dB]")
    axes[0].grid(True, linestyle=":", linewidth=0.8)
    axes[0].legend()

    lines_phase = [
        (H_direct, "Direct", "--"),
        (H_ris, "RIS", ":"),
        (H_total, "Total", "-"),
    ]
    for H, label, style in lines_phase:
        axes[1].plot(freqs_khz, np.angle(H, deg=True), label=label, linestyle=style, linewidth=1.6 if label == "Total" else 1.0)
    axes[1].set_xlabel("Frequency offset [kHz]")
    axes[1].set_ylabel("∠H(f) [deg]")
    axes[1].grid(True, linestyle=":", linewidth=0.8)
    axes[1].legend()

    plt.tight_layout()
    return fig


def plot_system_layout(sim_params):
    """Visualize the BS-RIS-UE geometry."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Position: BS at origin
    bs_pos = np.array([0.0, 0.0])
    
    # Position: RIS
    d_bs_ris = sim_params["links"]["bs_ris"]["distance"]
    theta_bs_ris = sim_params["links"]["bs_ris"]["aod_deg"]
    ris_pos = bs_pos + d_bs_ris * np.array([np.cos(np.deg2rad(theta_bs_ris)), 
                                             np.sin(np.deg2rad(theta_bs_ris))])
    
    # Position: UE
    d_ris_ue = sim_params["links"]["ris_ue"]["distance"]
    theta_ris_ue = sim_params["links"]["ris_ue"]["aod_deg"]
    ue_pos = ris_pos + d_ris_ue * np.array([np.cos(np.deg2rad(theta_ris_ue + theta_bs_ris)), 
                                             np.sin(np.deg2rad(theta_ris_ue + theta_bs_ris))])
    
    # Plot BS
    ax.plot(bs_pos[0], bs_pos[1], 'bs', markersize=15, label='Base Station')
    ax.text(bs_pos[0], bs_pos[1] - 5, 'BS', ha='center', fontsize=10, fontweight='bold')
    
    # Plot RIS
    ax.plot(ris_pos[0], ris_pos[1], 'ro', markersize=15, label='RIS')
    ax.text(ris_pos[0], ris_pos[1] + 5, 'RIS', ha='center', fontsize=10, fontweight='bold')
    
    # Plot UE
    ax.plot(ue_pos[0], ue_pos[1], 'g^', markersize=15, label='User Equipment')
    ax.text(ue_pos[0], ue_pos[1] - 5, 'UE', ha='center', fontsize=10, fontweight='bold')
    
    # Draw links
    if not sim_params["links"]["direct"].get("blocked", False):
        ax.plot([bs_pos[0], ue_pos[0]], [bs_pos[1], ue_pos[1]], 'k--', linewidth=1.5, alpha=0.5, label='Direct Link')
    else:
        ax.plot([bs_pos[0], ue_pos[0]], [bs_pos[1], ue_pos[1]], 'k:', linewidth=1.5, alpha=0.3, label='Direct Link (blocked)')
        
        # Draw blockage if specified
        if "block_visual" in sim_params["links"]["direct"]:
            block = sim_params["links"]["direct"]["block_visual"]
            fraction = block.get("fraction", 0.5)
            width = block.get("width", 10.0)
            depth = block.get("depth", 5.0)
            
            block_center = bs_pos + fraction * (ue_pos - bs_pos)
            angle = np.arctan2(ue_pos[1] - bs_pos[1], ue_pos[0] - bs_pos[0])
            perpendicular = angle + np.pi / 2
            
            corners = [
                block_center + 0.5 * width * np.array([np.cos(perpendicular), np.sin(perpendicular)]) + 0.5 * depth * np.array([np.cos(angle), np.sin(angle)]),
                block_center - 0.5 * width * np.array([np.cos(perpendicular), np.sin(perpendicular)]) + 0.5 * depth * np.array([np.cos(angle), np.sin(angle)]),
                block_center - 0.5 * width * np.array([np.cos(perpendicular), np.sin(perpendicular)]) - 0.5 * depth * np.array([np.cos(angle), np.sin(angle)]),
                block_center + 0.5 * width * np.array([np.cos(perpendicular), np.sin(perpendicular)]) - 0.5 * depth * np.array([np.cos(angle), np.sin(angle)]),
            ]
            blockage_polygon = Polygon(corners, closed=True, facecolor='gray', edgecolor='black', alpha=0.3)
            ax.add_patch(blockage_polygon)
    
    ax.plot([bs_pos[0], ris_pos[0]], [bs_pos[1], ris_pos[1]], 'b-', linewidth=2, alpha=0.7, label='BS → RIS')
    ax.plot([ris_pos[0], ue_pos[0]], [ris_pos[1], ue_pos[1]], 'r-', linewidth=2, alpha=0.7, label='RIS → UE')
    
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title(f'System Layout: BS-RIS-UE (RIS mode: {sim_params.get("ris_phase_mode", "identity")})')
    ax.legend(loc='best')
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.axis('equal')
    
    return fig


def demo():
    """Run a demonstration of the RIS-assisted MISO system."""
    print("=" * 60)
    print("RIS-Assisted MISO System Simulation")
    print("=" * 60)
    print(f"Carrier frequency: {params['fc']/1e9:.2f} GHz")
    print(f"BS antennas: {params['Nt']}")
    print(f"RIS elements: {params['Nr']}")
    print(f"OFDM subcarriers: {params['ofdm']['K']}")
    print(f"Subcarrier spacing: {params['ofdm']['subcarrier_spacing']/1e3:.1f} kHz")
    print(f"RIS phase mode: {params['ris_phase_mode']}")
    print(f"Direct link blocked: {params['links']['direct']['blocked']}")
    print("=" * 60)
    
    sim_out = simulate(params)
    
    print(f"\nResults:")
    print(f"  Direct channel power: {10*np.log10(np.mean(np.abs(sim_out['H_direct'])**2)):.2f} dB")
    print(f"  RIS channel power: {10*np.log10(np.mean(np.abs(sim_out['H_ris'])**2)):.2f} dB")
    print(f"  Total channel power: {10*np.log10(np.mean(np.abs(sim_out['H_total'])**2)):.2f} dB")
    print(f"  SNR: {sim_out['snr_dB']:.1f} dB")
    
    plot_frequency_response(sim_out["freqs"], sim_out["H_direct"], sim_out["H_ris"], sim_out["H_total"])
    plot_system_layout(params)
    plt.show()


if __name__ == "__main__":
    demo()
