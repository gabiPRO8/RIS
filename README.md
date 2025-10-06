# RIS - Reconfigurable Intelligent Surfaces

Bachelor's Project: RIS-Assisted Wireless Communication System

## Overview

This project implements a comprehensive simulation framework for studying Reconfigurable Intelligent Surfaces (RIS) in wireless communication systems. The implementation focuses on a basic single base station (BS), single RIS, single user equipment (UE) scenario with wideband OFDM transmissions.

## System Model

### Architecture
- **Base Station (BS)**: Multi-antenna transmitter with Nt = 8 antennas (ULA)
- **RIS**: Passive reflecting surface with Nr = 256 elements (ULA)
- **User Equipment (UE)**: Single-antenna receiver
- **Direct Link**: BS → UE (can be blocked/attenuated)
- **Reflected Link**: BS → RIS → UE

### Key Features
1. **Wideband OFDM Transmission**
   - Multiple subcarriers (default: 128)
   - Subcarrier spacing: 15 kHz
   - Carrier frequency: 3.5 GHz

2. **Rice Fading Channel Model**
   - Line-of-Sight (LOS) component
   - Rayleigh scattered component
   - Variable K-factor (dB) to control fading severity
   - Frequency-selective channels across OFDM subcarriers

3. **Multiple RIS Phase-Shift Strategies**
   - **Constant Phase-Shift**: Frequency-independent phase shifts (all elements same phase)
   - **True-Time Delay (TTD)**: Frequency-dependent phase shifts based on time delays
   - **Phase-Aligned**: Optimized for channel conditions using CSI
   - **Random Phase**: Baseline for comparison
   - **Identity**: No phase shift (pass-through)

## File Structure

```
RIS/
├── README.md                  # This file
├── MISO_DEMO1.py             # Main simulation script with demo
├── ris_models.py             # RIS model classes (OOP implementation)
└── ris_comparison.py         # Comparison script for different RIS modes
```

## Technical Concepts

### 1. Constant Phase-Shift RIS
All RIS elements apply the same phase shift, independent of frequency:
```
φ_n = exp(jθ) for all n = 1, ..., Nr
```
- **Pros**: Simple, easy to implement in hardware
- **Cons**: Frequency-flat, may not exploit wideband diversity
- **Use case**: Narrowband systems or when simplicity is required

### 2. True-Time Delay (TTD) RIS
Each element introduces a true time delay τ_n, creating frequency-dependent phase shifts:
```
φ_n(f) = exp(j2πfτ_n)
```
At carrier frequency fc with subcarrier offset Δf_k:
```
φ_n,k = exp(j2π(fc + Δf_k)τ_n)
```
- **Pros**: Maintains beam coherence across wideband, better performance
- **Cons**: More complex hardware, requires precise delay control
- **Use case**: Wideband OFDM systems with large bandwidth

### 3. Phase-Aligned RIS
Optimizes phase shifts to maximize received signal strength using Channel State Information:
```
φ_n = exp(-j∠(h_ru,n × [H_br,n × f_tx]))
```
- **Pros**: Optimal performance with CSI, can align with direct path
- **Cons**: Requires CSI feedback, adaptation overhead
- **Use case**: Systems with reliable CSI feedback

### 4. Rice Fading Channel
Combines LOS and scattered components:
```
h = √(K/(K+1)) · h_LOS + √(1/(K+1)) · h_scattered
```
where:
- K: Rice K-factor (linear scale)
- h_LOS: Deterministic LOS component with phase shift
- h_scattered: Complex Gaussian random variable (Rayleigh fading)

**K-factor interpretation:**
- K = ∞ (K_dB = 100): Pure LOS (no fading)
- K = 10 (K_dB = 10): Strong LOS with some fading
- K = 1 (K_dB = 0): Equal LOS and scattered power
- K = 0 (K_dB = -∞): Pure Rayleigh fading (no LOS)

### 5. OFDM Wideband Transmission
Orthogonal Frequency Division Multiplexing divides the wideband channel into multiple narrowband subcarriers:
```
Subcarrier frequencies: f_k = fc + (k - K/2) × Δf
```
where:
- fc: Carrier frequency
- K: Number of subcarriers
- Δf: Subcarrier spacing
- k: Subcarrier index (0 to K-1)

Each subcarrier experiences a different channel due to frequency selectivity.

## Usage

### Basic Demo
Run the main simulation with default parameters:
```bash
python MISO_DEMO1.py
```

This will:
1. Simulate the RIS-assisted MISO system
2. Display channel power metrics
3. Plot frequency response (magnitude and phase)
4. Show system geometry layout

### RIS Mode Comparison
Compare different RIS phase-shift strategies:
```bash
python ris_comparison.py
```

This will:
1. Explain key technical concepts
2. Simulate all RIS modes (identity, constant_phase, align, ttd, random)
3. Compare performance metrics
4. Generate comparison plots

### Using RIS Models (OOP)
```python
from ris_models import create_ris_model

# Create a True-Time Delay RIS
ris = create_ris_model('ttd', num_elements=256, carrier_frequency=3.5e9)

# Configure with time delays or reference phases
ris.configure(reference_phases=optimal_phases, reference_frequency=3.5e9)

# Get frequency-dependent response
phi_k = ris.get_phase_response(frequency=3.502e9)  # At specific frequency
```

### Customizing Parameters
Modify `params` dictionary in `MISO_DEMO1.py`:

```python
params = {
    "fc": 3.5e9,                 # Carrier frequency
    "Nt": 8,                     # BS antennas
    "Nr": 256,                   # RIS elements
    "ris_phase_mode": "ttd",     # RIS mode
    "links": {
        "direct": {
            "distance": 50.0,     # BS-UE distance
            "K_dB": 100.0,        # K-factor
            "blocked": True,      # Enable/disable direct link
        },
        "bs_ris": {
            "distance": 60.0,     # BS-RIS distance
            "K_dB": 100.0,
            "aod_deg": 20.0,      # Angle of departure
        },
        "ris_ue": {
            "distance": 30.0,     # RIS-UE distance
            "K_dB": 100.0,
        }
    },
    "ofdm": {
        "K": 128,                 # Subcarriers
        "subcarrier_spacing": 15e3
    }
}
```

## Understanding the Results

### Channel Power Metrics
- **Direct channel power**: Power of BS → UE direct link
- **RIS channel power**: Power of BS → RIS → UE reflected link
- **Total channel power**: Combined power (coherent addition)
- **RIS gain**: Improvement over direct link only

### Frequency Response Plots
- **Magnitude plot**: Shows channel gain across subcarriers (frequency-selective fading)
- **Phase plot**: Shows phase variation across subcarriers
  - Flat phase → Frequency-flat channel (good for wideband)
  - Varying phase → Frequency-selective channel (may need equalization)

### System Layout
- Shows spatial arrangement of BS, RIS, and UE
- Indicates link distances and blockage
- Helps visualize the geometric configuration

## Key Parameters to Experiment With

1. **K-factor (K_dB)**:
   - Increase K_dB → Stronger LOS, less fading
   - Decrease K_dB → Weaker LOS, more fading
   - Try: 0 dB (Rayleigh), 10 dB, 20 dB, 100 dB (pure LOS)

2. **Number of RIS elements (Nr)**:
   - More elements → Higher array gain
   - Try: 16, 64, 256, 1024

3. **RIS phase mode**:
   - Compare: "constant_phase" vs "ttd" vs "align"
   - Observe frequency response differences

4. **Direct link blockage**:
   - Enable/disable with `"blocked": True/False`
   - Shows importance of RIS when LOS is blocked

5. **OFDM parameters**:
   - Number of subcarriers (K): 32, 64, 128, 256
   - Subcarrier spacing: 15 kHz, 30 kHz, 60 kHz
   - Observe frequency selectivity effects

## Implementation Details

### Channel Model
- Uses uniform linear array (ULA) geometry
- Free-space path loss: λ/(4πd)
- Rice fading: LOS + complex Gaussian scattering
- Frequency-selective channels for each OFDM subcarrier

### RIS Reflection Model
For constant phase-shift:
```
H_ris[k] = h_ru[k]^T · Φ · H_br[k] · f_tx
```
where Φ = diag(φ_1, ..., φ_Nr) with frequency-independent φ_n.

For TTD:
```
H_ris[k] = h_ru[k]^T · Φ(f_k) · H_br[k] · f_tx
```
where Φ(f_k) = diag(exp(j2πf_k·τ_1), ..., exp(j2πf_k·τ_Nr)).

### Precoding
- Uses ULA steering vector aligned with BS-RIS direction
- Beamforming focuses energy toward RIS

## Dependencies

```bash
pip install numpy matplotlib
```

## Future Extensions

1. **Multi-user scenarios**: Multiple UEs served by one RIS
2. **Dynamic RIS optimization**: Real-time phase adaptation
3. **Hybrid beamforming**: Analog + digital precoding
4. **Channel estimation**: Imperfect CSI effects
5. **Hardware impairments**: Phase quantization, amplitude errors
6. **Mobility**: Time-varying channels
7. **Multi-RIS cooperation**: Multiple RIS surfaces

## References

Key concepts implemented:
- Rice fading channel model (K-factor)
- OFDM wideband transmission
- RIS phase-shift models (constant phase vs TTD)
- ULA beamforming and steering vectors
- Frequency-selective channels

## License

This is a Bachelor's project implementation for educational purposes.

## Author

Bachelor's thesis project on RIS-assisted wireless communications.
