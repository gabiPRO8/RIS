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
    return 10.0 ** (db / 10.0)


def complex_gaussian(shape, rng):
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2.0)


def free_space_pathloss_amp(distance, wavelength):
    distance = max(distance, 1e-3)
    return wavelength / (4.0 * np.pi * distance)

def ula_steering(num_elements, angle_deg, element_spacing_lambda):
    angle_rad = np.deg2rad(angle_deg)
    n = np.arange(num_elements)
    phase = -2j * np.pi * element_spacing_lambda * n * np.sin(angle_rad)
    return np.exp(phase) / np.sqrt(num_elements)


def configure_ris_phase(sim_params, *, H_br=None, h_ru=None, f_tx=None, H_direct=None):
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

    return np.ones(Nr, dtype=complex)


def generate_channels(sim_params, rng):
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
    rng = np.random.default_rng(sim_params.get("rng_seed", None))

    f_tx = ula_steering(sim_params["Nt"], sim_params.get("precoder_target_deg", 0.0), sim_params["bs_element_spacing"])
    h_direct, H_br, h_ru, delays = generate_channels(sim_params, rng)

    H_direct = np.einsum("kn,n->k", h_direct, f_tx)
    phi = configure_ris_phase(sim_params, H_br=H_br, h_ru=h_ru, f_tx=f_tx, H_direct=H_direct)

    K = sim_params["ofdm"]["K"]
    delta_f = sim_params["ofdm"]["subcarrier_spacing"]
    freqs = (np.arange(K) - K // 2) * delta_f

    H_ris = np.empty(K, dtype=complex)
    for k in range(K):
        reflected = phi * (H_br[k] @ f_tx)
        H_ris[k] = np.dot(h_ru[k], reflected)

    H_total = H_direct + H_ris

    rng_bits = np.random.default_rng(sim_params.get("rng_seed", None))
    bits = rng_bits.integers(0, 2, size=(K, 2))
    symbols = (2 * bits[:, 0] - 1) + 1j * (2 * bits[:, 1] - 1)
    X = symbols / np.sqrt(2.0)

    snr_lin = db_to_linear(sim_params["noise"]["snr_dB"])
    noise_var = 1.0 / snr_lin
    noise = np.sqrt(noise_var) * complex_gaussian((K,), rng)

    Y = H_total * X + noise

    return {
        "freqs": freqs,
        "H_direct": H_direct,
        "H_ris": H_ris,
        "H_total": H_total,
        "X": X,
        "Y": Y,
        "delays": delays
    }






def compute_continuous_channel(sim_params, *, include_scattering=False):
    """Compute a continuous-tone (single-frequency) channel snapshot without OFDM machinery."""
    fc = sim_params["fc"]
    wavelength = sim_params["c"] / fc
    Nt = sim_params["Nt"]
    Nr = sim_params["Nr"]

    d_direct = sim_params["links"]["direct"]["distance"]
    d_bs_ris = sim_params["links"]["bs_ris"]["distance"]
    d_ris_ue = sim_params["links"]["ris_ue"]["distance"]

    pl_direct = free_space_pathloss_amp(d_direct, wavelength)
    pl_bs_ris = free_space_pathloss_amp(d_bs_ris, wavelength)
    pl_ris_ue = free_space_pathloss_amp(d_ris_ue, wavelength)

    theta_direct = sim_params["links"]["direct"]["aod_deg"]
    theta_bs_ris = sim_params["links"]["bs_ris"]["aod_deg"]
    psi_bs_ris = sim_params["links"]["bs_ris"]["aoa_deg"]
    theta_ris_ue = sim_params["links"]["ris_ue"]["aod_deg"]

    d_tx = sim_params["bs_element_spacing"]
    d_ris = sim_params["ris_element_spacing"]

    K_direct = db_to_linear(sim_params["links"]["direct"]["K_dB"])
    los_direct = pl_direct * np.sqrt(K_direct / (K_direct + 1.0)) * ula_steering(Nt, theta_direct, d_tx)
    if include_scattering:
        rng = np.random.default_rng(sim_params.get("rng_seed", None))
        scatter_direct = pl_direct * np.sqrt(1.0 / (K_direct + 1.0)) * complex_gaussian((Nt,), rng)
    else:
        scatter_direct = 0.0
    h_direct_vec = los_direct + scatter_direct
    direct_cfg = sim_params["links"]["direct"]
    if direct_cfg.get("blocked", False):
        loss_dB = direct_cfg.get("blockage_loss_dB")
        if loss_dB is None:
            h_direct_vec = np.zeros_like(h_direct_vec)
        else:
            h_direct_vec *= 10.0 ** (-loss_dB / 20.0)

    K_br = db_to_linear(sim_params["links"]["bs_ris"]["K_dB"])
    a_ris_inc = ula_steering(Nr, psi_bs_ris, d_ris)
    a_bs_out = ula_steering(Nt, theta_bs_ris, d_tx)
    los_br = pl_bs_ris * np.sqrt(K_br / (K_br + 1.0)) * np.outer(a_ris_inc, np.conj(a_bs_out))
    if include_scattering:
        scatter_br = pl_bs_ris * np.sqrt(1.0 / (K_br + 1.0)) * complex_gaussian((Nr, Nt), rng)
    else:
        scatter_br = 0.0
    H_br_mat = los_br + scatter_br

    K_ru = db_to_linear(sim_params["links"]["ris_ue"]["K_dB"])
    a_ris_out = ula_steering(Nr, theta_ris_ue, d_ris)
    los_ru = pl_ris_ue * np.sqrt(K_ru / (K_ru + 1.0)) * np.conj(a_ris_out)
    if include_scattering:
        scatter_ru = pl_ris_ue * np.sqrt(1.0 / (K_ru + 1.0)) * complex_gaussian((Nr,), rng)
    else:
        scatter_ru = 0.0
    h_ru_vec = los_ru + scatter_ru

    f_tx = ula_steering(Nt, sim_params.get("precoder_target_deg", 0.0), d_tx)

    H_direct = np.dot(h_direct_vec, f_tx)

    incident = H_br_mat @ f_tx
    cascade = h_ru_vec * incident
    epsilon = 1e-12
    if np.all(np.abs(cascade) < epsilon):
        phi = np.ones(Nr, dtype=complex)
    else:
        phases = np.exp(-1j * np.angle(cascade))
        if np.abs(H_direct) >= epsilon:
            phases *= np.exp(1j * np.angle(H_direct))
        phi = phases

    H_ris = np.dot(h_ru_vec, phi * incident)
    H_total = H_direct + H_ris

    return {
        "H_direct": H_direct,
        "H_ris": H_ris,
        "H_total": H_total,
        "phi": phi,
        "incident": incident,
        "f_tx": f_tx,
    }

def simulate_narrowband(sim_params, *, include_scattering=False):
    return compute_continuous_channel(sim_params, include_scattering=include_scattering)


def summarize_narrowband(sim_params, *, include_scattering=False):
    nb = simulate_narrowband(sim_params, include_scattering=include_scattering)
    tx_power_dBm = sim_params.get("tx_power_dBm", 0.0)
    min_amp = 1e-12

    summary = {}
    for key in ("H_direct", "H_ris", "H_total"):
        value = nb[key]
        amp = np.abs(value)
        summary[key] = {
            "complex": value,
            "amplitude": amp,
            "phase_rad": np.angle(value),
            "power_dBm": tx_power_dBm + 20.0 * np.log10(max(amp, min_amp)),
        }

    summary["phase_gap_total_direct"] = summary["H_total"]["phase_rad"] - summary["H_direct"]["phase_rad"]
    summary["phase_gap_total_ris"] = summary["H_total"]["phase_rad"] - summary["H_ris"]["phase_rad"]
    summary["phi"] = nb["phi"]
    summary["incident"] = nb["incident"]
    summary["f_tx"] = nb["f_tx"]
    return summary


def compute_system_geometry(sim_params):
    bs_pos = np.zeros(2, dtype=float)
    d_bs_ris = sim_params["links"]["bs_ris"]["distance"]
    theta_bs_ris = np.deg2rad(sim_params["links"]["bs_ris"]["aod_deg"])
    ris_pos = np.array([
        d_bs_ris * np.cos(theta_bs_ris),
        d_bs_ris * np.sin(theta_bs_ris),
    ])

    d_direct = sim_params["links"]["direct"]["distance"]
    d_ris_ue = sim_params["links"]["ris_ue"]["distance"]

    d = np.linalg.norm(ris_pos - bs_pos)
    if d < 1e-9:
        raise ValueError("BS and RIS positions coincide; geometry undefined.")

    ex = (ris_pos - bs_pos) / d
    ey = np.array([-ex[1], ex[0]])

    a = (d_direct ** 2 - d_ris_ue ** 2 + d ** 2) / (2.0 * d)
    residual = d_direct ** 2 - a ** 2
    if residual < -1e-6:
        raise ValueError("Inconsistent distance configuration; no intersection found.")
    h = np.sqrt(max(residual, 0.0))

    base_point = bs_pos + a * ex
    candidate1 = base_point + h * ey
    candidate2 = base_point - h * ey

    target_angle = np.deg2rad(sim_params["links"]["direct"]["aod_deg"])

    def angle_from_bs(point):
        return np.arctan2(point[1] - bs_pos[1], point[0] - bs_pos[0])

    def angle_diff(theta):
        return np.arctan2(np.sin(theta), np.cos(theta))

    candidates = [candidate1, candidate2]
    ue_pos = min(
        candidates,
        key=lambda pt: abs(angle_diff(angle_from_bs(pt) - target_angle))
    )

    actual_angle = angle_from_bs(ue_pos)
    rotation = target_angle - actual_angle
    cos_rot = np.cos(rotation)
    sin_rot = np.sin(rotation)
    rotation_matrix = np.array([[cos_rot, -sin_rot], [sin_rot, cos_rot]])

    ris_pos = rotation_matrix @ ris_pos
    ue_pos = rotation_matrix @ ue_pos

    distances = {
        "bs_to_ris": np.linalg.norm(ris_pos - bs_pos),
        "bs_to_ue": np.linalg.norm(ue_pos - bs_pos),
        "ris_to_ue": np.linalg.norm(ue_pos - ris_pos),
    }

    angles_actual = {
        "bs_to_ris": np.degrees(np.arctan2(ris_pos[1] - bs_pos[1], ris_pos[0] - bs_pos[0])),
        "bs_to_ue": np.degrees(np.arctan2(ue_pos[1] - bs_pos[1], ue_pos[0] - bs_pos[0])),
        "ris_to_bs": np.degrees(np.arctan2(bs_pos[1] - ris_pos[1], bs_pos[0] - ris_pos[0])),
        "ris_to_ue": np.degrees(np.arctan2(ue_pos[1] - ris_pos[1], ue_pos[0] - ris_pos[0])),
    }

    angles_declared = {
        "bs_to_ris": sim_params["links"]["bs_ris"]["aod_deg"],
        "bs_to_ue": sim_params["links"]["direct"]["aod_deg"],
        "ris_to_ue": sim_params["links"]["ris_ue"]["aod_deg"],
        "ris_to_bs": sim_params["links"]["bs_ris"]["aoa_deg"],
    }

    return {
        "positions": {
            "bs": bs_pos,
            "ris": ris_pos,
            "ue": ue_pos,
        },
        "distances": distances,
        "angles_actual": angles_actual,
        "angles_declared": angles_declared,
    }



def plot_system_layout(sim_params):
    geometry = compute_system_geometry(sim_params)
    positions = geometry["positions"]
    distances = geometry["distances"]
    angles_actual = geometry["angles_actual"]
    angles_declared = geometry["angles_declared"]

    bs = positions["bs"]
    ris = positions["ris"]
    ue = positions["ue"]

    direct_cfg = sim_params["links"]["direct"]
    direct_blocked = direct_cfg.get("blocked", False)
    block_visual = direct_cfg.get("block_visual", {})
    block_loss_dB = direct_cfg.get("blockage_loss_dB")
    partial_block = direct_blocked and block_loss_dB is not None
    full_block = direct_blocked and block_loss_dB is None

    direct_vector = ue - bs
    direct_distance = np.linalg.norm(direct_vector)
    if direct_distance > 1e-9:
        direction_unit = direct_vector / direct_distance
        normal_unit = np.array([-direction_unit[1], direction_unit[0]])
    else:
        direction_unit = np.array([1.0, 0.0])
        normal_unit = np.array([0.0, 1.0])

    fig, ax = plt.subplots()
    ax.scatter(
        [bs[0], ris[0], ue[0]],
        [bs[1], ris[1], ue[1]],
        color=["C0", "C1", "C2"],
        s=80,
        zorder=3,
    )

    ax.text(bs[0], bs[1] + 5.0, "BS", ha="center", va="bottom")
    ax.text(ris[0], ris[1] + 5.0, "RIS", ha="center", va="bottom")
    ax.text(ue[0], ue[1] + 5.0, "UE", ha="center", va="bottom")

    links = []
    if full_block:
        links.append(("bs_to_ue", bs, ue, "#666666", "--", False))
    elif partial_block:
        links.append(("bs_to_ue", bs, ue, "#888888", "--", True))
    else:
        links.append(("bs_to_ue", bs, ue, "C0", "-.", True))
    links.append(("bs_to_ris", bs, ris, "C1", "--", True))
    links.append(("ris_to_ue", ris, ue, "C2", "-", True))

    for key, start, end, color, style, with_arrow in links:
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            linestyle=style,
            color=color,
            linewidth=1.6,
            zorder=2,
        )
        if with_arrow:
            ax.annotate(
                "",
                xy=end,
                xytext=start,
                arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
            )

    scale = max(distances.values())
    offset = max(4.0, 0.04 * scale)

    if direct_blocked and direct_distance > 1e-9:
        fraction = float(block_visual.get("fraction", 0.5))
        fraction = min(max(fraction, 0.0), 1.0)
        width = float(block_visual.get("width", 12.0))
        depth = float(block_visual.get("depth", 6.0))
        half_depth = depth / 2.0
        half_width = width / 2.0

        center = bs + direction_unit * direct_distance * fraction
        corners = np.array([
            center + direction_unit * half_depth + normal_unit * half_width,
            center - direction_unit * half_depth + normal_unit * half_width,
            center - direction_unit * half_depth - normal_unit * half_width,
            center + direction_unit * half_depth - normal_unit * half_width,
        ])
        ax.add_patch(
            Polygon(
                corners,
                closed=True,
                facecolor="lightgray",
                edgecolor="0.4",
                alpha=0.6,
            )
        )

        if full_block:
            building_caption = "Building (blocking)"
            los_caption = "LOS blocked"
        else:
            loss_text = f"~{block_loss_dB:.0f} dB loss" if block_loss_dB is not None else "blocking"
            building_caption = f"Building ({loss_text})"
            if block_loss_dB is not None:
                los_caption = f"LOS partially blocked (+{block_loss_dB:.0f} dB)"
            else:
                los_caption = "LOS partially blocked"

        building_label_pos = center + normal_unit * max(offset, half_width * 0.6)
        ax.text(
            building_label_pos[0],
            building_label_pos[1],
            building_caption,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85),
        )

        los_midpoint = bs + 0.5 * direct_vector
        los_label_pos = los_midpoint - normal_unit * offset
        ax.text(
            los_label_pos[0],
            los_label_pos[1],
            los_caption,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.75),
        )
    if full_block:
        los_label = "d_BS->UE (blocked)"
    elif partial_block:
        los_label = f"d_BS->UE (partial, +{block_loss_dB:.0f} dB)"
    else:
        los_label = "d_BS->UE"

    distance_labels = {
        "bs_to_ue": los_label,
        "bs_to_ris": "d_BS->RIS",
        "ris_to_ue": "d_RIS->UE",
    }

    def annotate_distance(key, start, end):
        midpoint = 0.5 * (start + end)
        direction = end - start
        norm = np.hypot(direction[0], direction[1])
        if norm > 1e-9:
            normal = np.array([-direction[1], direction[0]]) / norm
        else:
            normal = np.array([0.0, 1.0])
        text_pos = midpoint + normal * offset
        ax.text(
            text_pos[0],
            text_pos[1],
            f"{distance_labels[key]} = {distances[key]:.1f} m",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.75),
        )

    annotate_distance("bs_to_ue", bs, ue)
    annotate_distance("bs_to_ris", bs, ris)
    annotate_distance("ris_to_ue", ris, ue)

    if full_block:
        los_angle_label = "theta_BS->UE (blocked)"
    elif partial_block:
        los_angle_label = f"theta_BS->UE (partial, +{block_loss_dB:.0f} dB)"
    else:
        los_angle_label = "theta_BS->UE"

    angle_annotations = [
        (
            "bs_to_ue",
            los_angle_label,
            bs + np.array([0.0, -2.0 * offset]),
            "center",
            "top",
        ),
        ("bs_to_ris", "theta_BS->RIS", bs + np.array([-2.0 * offset, 0.0]), "right", "center"),
        ("ris_to_ue", "theta_RIS->UE", ris + np.array([2.0 * offset, 0.0]), "left", "center"),
        ("ris_to_bs", "psi_RIS<-BS", ris + np.array([2.0 * offset, -2.0 * offset]), "left", "top"),
    ]

    for key, label, position, ha, va in angle_annotations:
        declared = angles_declared.get(key)
        ax.text(
            position[0],
            position[1],
            f"{label} = {angles_actual[key]:.1f} deg" + (
                f" (param {declared:.1f} deg)" if declared is not None else ""
            ),
            ha=ha,
            va=va,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85),
        )

    coords = np.vstack([bs, ris, ue])
    max_range = np.max(np.ptp(coords, axis=0))
    margin = max(10.0, 0.1 * max_range)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(coords[:, 0].min() - margin, coords[:, 0].max() + margin)
    ax.set_ylim(coords[:, 1].min() - margin, coords[:, 1].max() + margin)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("System geometry with distances and angles")
    ax.grid(True, linestyle=":", linewidth=0.8)

    return geometry




def plot_frequency_response(freqs_hz, H_direct, H_ris, H_total):
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
        axes[1].plot(freqs_khz, np.unwrap(np.angle(H)), label=label, linestyle=style, linewidth=1.6 if label == "Total" else 1.0)
    axes[1].set_ylabel("Phase [rad]")
    axes[1].set_xlabel("Frequency offset [kHz]")
    axes[1].grid(True, linestyle=":", linewidth=0.8)
    axes[1].legend()

    fig.suptitle("Channel frequency response")
    return fig, axes


def plot_power_delay_profile(delays, H_direct, H_ris, tx_power_dBm=0.0):
    direct_power_lin = np.mean(np.abs(H_direct) ** 2)
    ris_power_lin = np.mean(np.abs(H_ris) ** 2)

    delay_ns = np.array([delays["direct"], delays["ris"]]) * 1e9
    powers_lin = np.array([direct_power_lin, ris_power_lin])
    labels = ["Direct", "RIS"]

    min_power_lin = 10 ** (-30)
    powers_dBm = tx_power_dBm + 10.0 * np.log10(np.maximum(powers_lin, min_power_lin))

    fig, ax = plt.subplots()
    marker = ax.stem(delay_ns, powers_dBm, basefmt=" ")
    plt.setp(marker.markerline, markersize=8)
    plt.setp(marker.stemlines, linewidth=1.6)
    for x, y, label in zip(delay_ns, powers_dBm, labels):
        ax.text(x, y + 3, f"{label}", ha="center")

    ax.set_title("Power delay profile (received power)")
    ax.set_xlabel("Delay (ns)")
    ax.set_ylabel("Power [dBm]")
    ax.grid(True, linestyle=":", linewidth=0.8)
    return fig, ax


def demo():
    sim_out = simulate(params)

    nb_summary = summarize_narrowband(params)
    print("Narrowband single-tone snapshot:")
    for key, label in (("H_direct", "Direct"), ("H_ris", "RIS"), ("H_total", "Total")):
        entry = nb_summary[key]
        print(
            f"  {label:<6} |H| = {entry['amplitude']:.3e}, phase = {entry['phase_rad']:.4f} rad,"
            f" Prx ? {entry['power_dBm']:.2f} dBm"
        )
    print(f"  Phase gap Total - Direct: {nb_summary['phase_gap_total_direct']:.3e} rad")
    print(f"  Phase gap Total - RIS:    {nb_summary['phase_gap_total_ris']:.3e} rad")

    plot_power_delay_profile(sim_out["delays"], sim_out["H_direct"], sim_out["H_ris"], params.get("tx_power_dBm", 0.0))
    plot_frequency_response(sim_out["freqs"], sim_out["H_direct"], sim_out["H_ris"], sim_out["H_total"])
    plot_system_layout(params)
    plt.show()


if __name__ == "__main__":
    demo()
