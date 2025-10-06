# Quick Start Guide

## Installation

```bash
# Install required packages
pip install -r requirements.txt
```

## Running the Examples

### 1. Basic Demo
Run the main simulation with default parameters:

```bash
python MISO_DEMO1.py
```

This will display:
- System configuration summary
- Channel power metrics for direct, RIS, and total links
- Frequency response plots (magnitude and phase)
- System geometry visualization

### 2. Compare RIS Modes
Run comprehensive comparison of different RIS modes:

```bash
python ris_comparison.py
```

This will:
- Explain technical concepts (Rice fading, OFDM, RIS types)
- Simulate all RIS modes (identity, constant_phase, align, ttd, random)
- Display performance comparison plots
- Show differences in frequency response

### 3. Run Tests
Verify the implementation:

```bash
python test_ris.py
```

## Understanding the Output

### Console Output Example
```
============================================================
RIS-Assisted MISO System Simulation
============================================================
Carrier frequency: 3.50 GHz
BS antennas: 8
RIS elements: 256
OFDM subcarriers: 128
Subcarrier spacing: 15.0 kHz
RIS phase mode: align
Direct link blocked: True
============================================================

Results:
  Direct channel power: -83.28 dB
  RIS channel power: -151.75 dB
  Total channel power: -83.27 dB
  SNR: 20.0 dB
```

### Key Metrics
- **Direct channel power**: Power received via BS → UE direct path
- **RIS channel power**: Power received via BS → RIS → UE reflected path
- **Total channel power**: Combined power (coherent sum of both paths)
- **RIS gain**: How much the RIS improves the total received power

### Plots

#### Frequency Response
- **Top panel**: Channel magnitude in dB across OFDM subcarriers
  - Shows frequency-selective fading
  - Compare direct, RIS, and total channel responses
- **Bottom panel**: Channel phase in degrees
  - Phase variation indicates frequency selectivity
  - Flat phase = good for wideband transmission

#### System Layout
- Visual representation of BS, RIS, and UE positions
- Shows link distances and angles
- Indicates if direct link is blocked

## Simple Example Code

```python
import numpy as np
from MISO_DEMO1 import simulate, params
import copy

# Modify parameters
custom_params = copy.deepcopy(params)
custom_params['Nr'] = 128              # Use 128 RIS elements
custom_params['ris_phase_mode'] = 'ttd'  # Use True-Time Delay mode

# Run simulation
result = simulate(custom_params)

# Extract results
H_total = result['H_total']
freqs = result['freqs']

# Compute average channel power
avg_power_dB = 10 * np.log10(np.mean(np.abs(H_total)**2))
print(f"Average channel power: {avg_power_dB:.2f} dB")
```

## Using RIS Model Classes

```python
from ris_models import create_ris_model
import numpy as np

# Create a True-Time Delay RIS
ris = create_ris_model('ttd', 
                       num_elements=256,
                       carrier_frequency=3.5e9,
                       element_spacing=0.5)

# Configure with channel-based optimization
# (assuming you have H_br, h_ru, f_tx from channel generation)
# ris.configure(reference_phases=optimal_phases, reference_frequency=3.5e9)

# Or set time delays manually
time_delays = np.linspace(0, 5e-9, 256)  # 0 to 5 ns linear delays
ris.configure(time_delays=time_delays)

# Get phase response at different frequencies
phi_center = ris.get_phase_response(3.5e9)        # At carrier
phi_edge = ris.get_phase_response(3.5e9 + 960e3)  # At edge subcarrier

# Apply to incident signal
incident_signal = np.random.randn(256) + 1j*np.random.randn(256)
reflected_signal = ris.apply_reflection(incident_signal, frequency=3.5e9)
```

## Experimenting with Parameters

### Vary K-factor (LOS strength)
```python
custom_params = copy.deepcopy(params)
custom_params['links']['direct']['K_dB'] = 10.0  # Moderate fading
# Try: 0 dB (Rayleigh), 10 dB, 20 dB, 100 dB (pure LOS)

result = simulate(custom_params)
```

### Change number of RIS elements
```python
custom_params = copy.deepcopy(params)
custom_params['Nr'] = 64  # Fewer elements
# Try: 16, 64, 128, 256, 512

result = simulate(custom_params)
```

### Enable/disable direct link
```python
custom_params = copy.deepcopy(params)
custom_params['links']['direct']['blocked'] = False  # Enable direct link

result = simulate(custom_params)
```

### Compare constant phase vs TTD
```python
import matplotlib.pyplot as plt

# Constant phase
params1 = copy.deepcopy(params)
params1['ris_phase_mode'] = 'constant_phase'
result1 = simulate(params1)

# TTD
params2 = copy.deepcopy(params)
params2['ris_phase_mode'] = 'ttd'
result2 = simulate(params2)

# Plot comparison
plt.figure(figsize=(10, 5))
plt.plot(result1['freqs']/1e3, 20*np.log10(np.abs(result1['H_total'])), 
         label='Constant Phase')
plt.plot(result2['freqs']/1e3, 20*np.log10(np.abs(result2['H_total'])), 
         label='TTD')
plt.xlabel('Frequency offset [kHz]')
plt.ylabel('Channel magnitude [dB]')
plt.legend()
plt.grid(True)
plt.show()
```

## Next Steps

1. **Understand the theory**: Read the technical explanations in README.md
2. **Run examples**: Execute the provided scripts and observe the output
3. **Modify parameters**: Experiment with different configurations
4. **Analyze results**: Compare performance across different RIS modes
5. **Extend the code**: Add your own RIS strategies or channel models

## Common Issues

### Issue: "No module named 'numpy'"
**Solution**: Install dependencies with `pip install -r requirements.txt`

### Issue: Plots don't display
**Solution**: Check if you're using a display environment. For headless systems, save plots instead:
```python
import matplotlib
matplotlib.use('Agg')  # Use before importing pyplot
import matplotlib.pyplot as plt
# ... create plots ...
plt.savefig('output.png')
```

### Issue: Want to save results
**Solution**: Save to file:
```python
import pickle
result = simulate(params)
with open('result.pkl', 'wb') as f:
    pickle.dump(result, f)
```

## Contact

This is a Bachelor's thesis project on RIS-assisted wireless communications.
For questions about the implementation, refer to the code documentation and README.md.
