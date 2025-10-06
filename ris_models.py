"""
RIS Models Module
=================
This module implements different Reconfigurable Intelligent Surface (RIS) models
for wireless communication systems.

RIS Types:
1. Constant Phase-Shift RIS: All elements apply the same frequency-independent phase shift
2. True-Time Delay (TTD) RIS: Elements apply frequency-dependent phase shifts based on time delays
3. Phase-Aligned RIS: Optimizes phases to align with channel conditions
"""

import numpy as np


class RISModel:
    """
    Base class for Reconfigurable Intelligent Surface models.
    
    A RIS is a passive device with multiple reflecting elements that can be configured
    to manipulate electromagnetic waves. Each element introduces a controllable phase shift.
    
    Attributes:
        num_elements (int): Number of RIS elements
        element_spacing (float): Spacing between elements in wavelengths
    """
    
    def __init__(self, num_elements, element_spacing=0.5):
        """
        Initialize RIS model.
        
        Args:
            num_elements: Number of RIS elements
            element_spacing: Element spacing in wavelengths (default: 0.5λ)
        """
        self.num_elements = num_elements
        self.element_spacing = element_spacing
        self.phase_coefficients = np.ones(num_elements, dtype=complex)
        
    def configure(self, **kwargs):
        """
        Configure the RIS phase coefficients.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement configure()")
    
    def get_phase_response(self, frequency):
        """
        Get the phase response at a given frequency.
        
        Args:
            frequency: Operating frequency in Hz
            
        Returns:
            Complex-valued phase coefficients (length = num_elements)
        """
        raise NotImplementedError("Subclasses must implement get_phase_response()")
    
    def apply_reflection(self, incident_signal, frequency=None):
        """
        Apply RIS reflection to incident signal.
        
        Args:
            incident_signal: Complex array of incident signals at each element
            frequency: Operating frequency (required for frequency-dependent models)
            
        Returns:
            Reflected signal with phase shifts applied
        """
        if frequency is not None:
            phi = self.get_phase_response(frequency)
        else:
            phi = self.phase_coefficients
        return phi * incident_signal


class ConstantPhaseRIS(RISModel):
    """
    Constant Phase-Shift RIS.
    
    All elements apply the same phase shift, independent of frequency.
    This is the simplest RIS model and is frequency-flat.
    
    Technical Note:
    - Phase shift is constant across all subcarriers in OFDM
    - Easy to implement in hardware
    - May not fully exploit wideband diversity
    """
    
    def __init__(self, num_elements, element_spacing=0.5, constant_phase=0.0):
        """
        Initialize Constant Phase-Shift RIS.
        
        Args:
            num_elements: Number of RIS elements
            element_spacing: Element spacing in wavelengths
            constant_phase: Phase shift to apply (in radians)
        """
        super().__init__(num_elements, element_spacing)
        self.constant_phase = constant_phase
        self.configure()
    
    def configure(self, phase=None):
        """
        Configure the constant phase shift.
        
        Args:
            phase: Phase shift in radians (default: use existing constant_phase)
        """
        if phase is not None:
            self.constant_phase = phase
        self.phase_coefficients = np.exp(1j * self.constant_phase) * np.ones(self.num_elements, dtype=complex)
    
    def get_phase_response(self, frequency):
        """
        Get phase response (frequency-independent for constant phase RIS).
        
        Args:
            frequency: Operating frequency (unused, for interface compatibility)
            
        Returns:
            Constant phase coefficients
        """
        return self.phase_coefficients


class TrueTimeDelayRIS(RISModel):
    """
    True-Time Delay (TTD) RIS.
    
    Each element introduces a true time delay, resulting in frequency-dependent
    phase shifts. This allows the RIS to maintain beam coherence across wideband signals.
    
    Technical Note:
    - Phase shift = 2π × frequency × time_delay
    - Frequency-dependent: different phase at each subcarrier
    - Better wideband performance than constant phase-shift
    - More complex hardware implementation
    
    In OFDM systems:
    - At center frequency fc: φ_n = exp(j × 2π × fc × τ_n)
    - At subcarrier k with offset Δf_k: φ_n,k = exp(j × 2π × (fc + Δf_k) × τ_n)
    """
    
    def __init__(self, num_elements, element_spacing=0.5, carrier_frequency=3.5e9):
        """
        Initialize True-Time Delay RIS.
        
        Args:
            num_elements: Number of RIS elements
            element_spacing: Element spacing in wavelengths
            carrier_frequency: Carrier frequency in Hz (for reference)
        """
        super().__init__(num_elements, element_spacing)
        self.carrier_frequency = carrier_frequency
        self.time_delays = np.zeros(num_elements)  # Time delays in seconds
        
    def configure(self, time_delays=None, reference_phases=None, reference_frequency=None):
        """
        Configure TTD RIS with time delays or reference phases.
        
        Args:
            time_delays: Array of time delays in seconds (length = num_elements)
            reference_phases: Array of phases at reference frequency (alternative to time_delays)
            reference_frequency: Reference frequency for phase-to-delay conversion
        """
        if time_delays is not None:
            self.time_delays = np.asarray(time_delays)
        elif reference_phases is not None:
            # Convert phases at reference frequency to time delays
            if reference_frequency is None:
                reference_frequency = self.carrier_frequency
            # τ = -φ / (2π × f)
            self.time_delays = -np.angle(reference_phases) / (2.0 * np.pi * reference_frequency)
        
        # Store the phase at carrier frequency as the base coefficients
        self.phase_coefficients = np.exp(1j * 2.0 * np.pi * self.carrier_frequency * self.time_delays)
    
    def get_phase_response(self, frequency):
        """
        Get frequency-dependent phase response.
        
        Args:
            frequency: Operating frequency in Hz
            
        Returns:
            Phase coefficients at the specified frequency
        """
        return np.exp(1j * 2.0 * np.pi * frequency * self.time_delays)
    
    def set_time_delays_from_angles(self, target_angle_deg, wavelength):
        """
        Configure time delays to steer the beam to a target angle.
        
        Args:
            target_angle_deg: Target angle in degrees (broadside = 0)
            wavelength: Wavelength at carrier frequency
        """
        # For ULA, time delay difference between adjacent elements
        # Δτ = d × sin(θ) / c
        c = 3e8  # speed of light
        target_angle_rad = np.deg2rad(target_angle_deg)
        d_meters = self.element_spacing * wavelength
        
        element_indices = np.arange(self.num_elements)
        self.time_delays = element_indices * d_meters * np.sin(target_angle_rad) / c
        
        # Update phase coefficients
        self.phase_coefficients = np.exp(1j * 2.0 * np.pi * self.carrier_frequency * self.time_delays)


class PhaseAlignedRIS(RISModel):
    """
    Phase-Aligned RIS.
    
    Optimizes phase shifts to align the cascaded channel (BS->RIS->UE) for maximum
    received signal strength. The phase of each element is chosen to conjugate
    the channel phase, creating coherent combining at the receiver.
    
    Technical Note:
    - Phase alignment: φ_n = exp(-j × angle(h_ru,n × H_br,n × f_tx))
    - Maximizes |h_ru^H × Φ × H_br × f_tx| where Φ = diag(φ)
    - Can optionally align with direct path for constructive interference
    - Frequency-selective: different phases per subcarrier (or use reference subcarrier)
    """
    
    def __init__(self, num_elements, element_spacing=0.5, use_reference_subcarrier=True):
        """
        Initialize Phase-Aligned RIS.
        
        Args:
            num_elements: Number of RIS elements
            element_spacing: Element spacing in wavelengths
            use_reference_subcarrier: If True, compute phases for one subcarrier only
        """
        super().__init__(num_elements, element_spacing)
        self.use_reference_subcarrier = use_reference_subcarrier
        
    def configure(self, H_br, h_ru, f_tx, H_direct=None, k_ref=None):
        """
        Configure phase alignment based on channel state information.
        
        Args:
            H_br: BS->RIS channel matrix (shape: [K, Nr, Nt] or [Nr, Nt])
            h_ru: RIS->UE channel vector (shape: [K, Nr] or [Nr])
            f_tx: BS precoding vector (shape: [Nt])
            H_direct: Optional direct channel for co-phasing (shape: [K] or scalar)
            k_ref: Reference subcarrier index (if None, use middle subcarrier)
        """
        # Handle single-subcarrier or multi-subcarrier inputs
        if H_br.ndim == 2:
            # Single subcarrier case
            H_br_ref = H_br
            h_ru_ref = h_ru
            H_direct_ref = H_direct if H_direct is not None else None
        else:
            # Multi-subcarrier case: extract reference
            K = H_br.shape[0]
            if k_ref is None:
                k_ref = K // 2
            H_br_ref = H_br[k_ref]
            h_ru_ref = h_ru[k_ref]
            H_direct_ref = H_direct[k_ref] if H_direct is not None else None
        
        # Compute incident signal at RIS
        incident = H_br_ref @ f_tx
        
        # Compute cascaded channel for each element
        cascade = h_ru_ref * incident
        
        # Phase alignment: conjugate the cascade phase
        epsilon = 1e-12
        if np.all(np.abs(cascade) < epsilon):
            self.phase_coefficients = np.ones(self.num_elements, dtype=complex)
        else:
            self.phase_coefficients = np.exp(-1j * np.angle(cascade))
            
            # Optional: align with direct path for constructive interference
            if H_direct_ref is not None and np.abs(H_direct_ref) >= epsilon:
                self.phase_coefficients *= np.exp(1j * np.angle(H_direct_ref))
    
    def get_phase_response(self, frequency):
        """
        Get phase response (frequency-independent in this implementation).
        
        Args:
            frequency: Operating frequency (unused, for interface compatibility)
            
        Returns:
            Phase coefficients
        """
        return self.phase_coefficients


class RandomPhaseRIS(RISModel):
    """
    Random Phase RIS.
    
    Each element has a random phase shift. Useful as a baseline for comparison.
    
    Technical Note:
    - Phases uniformly distributed in [0, 2π)
    - No optimization or channel knowledge required
    - Performance baseline (should be worse than optimized configurations)
    """
    
    def __init__(self, num_elements, element_spacing=0.5, seed=None):
        """
        Initialize Random Phase RIS.
        
        Args:
            num_elements: Number of RIS elements
            element_spacing: Element spacing in wavelengths
            seed: Random seed for reproducibility
        """
        super().__init__(num_elements, element_spacing)
        self.rng = np.random.default_rng(seed)
        self.configure()
    
    def configure(self, seed=None):
        """
        Generate random phase shifts.
        
        Args:
            seed: Optional new random seed
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        phases = self.rng.uniform(0.0, 2.0 * np.pi, size=self.num_elements)
        self.phase_coefficients = np.exp(1j * phases)
    
    def get_phase_response(self, frequency):
        """
        Get phase response (frequency-independent).
        
        Args:
            frequency: Operating frequency (unused, for interface compatibility)
            
        Returns:
            Random phase coefficients
        """
        return self.phase_coefficients


def create_ris_model(ris_type, num_elements, **kwargs):
    """
    Factory function to create RIS models.
    
    Args:
        ris_type: Type of RIS ('constant_phase', 'ttd', 'align', 'random', 'identity')
        num_elements: Number of RIS elements
        **kwargs: Additional arguments for specific RIS types
        
    Returns:
        RISModel instance
    """
    # Extract common parameters
    element_spacing = kwargs.pop('element_spacing', 0.5)
    
    if ris_type == "constant_phase":
        constant_phase = kwargs.pop('constant_phase', 0.0)
        return ConstantPhaseRIS(num_elements, element_spacing=element_spacing, 
                               constant_phase=constant_phase)
    elif ris_type == "ttd":
        carrier_frequency = kwargs.pop('carrier_frequency', 3.5e9)
        return TrueTimeDelayRIS(num_elements, element_spacing=element_spacing,
                               carrier_frequency=carrier_frequency)
    elif ris_type == "align":
        use_reference_subcarrier = kwargs.pop('use_reference_subcarrier', True)
        return PhaseAlignedRIS(num_elements, element_spacing=element_spacing,
                              use_reference_subcarrier=use_reference_subcarrier)
    elif ris_type == "random":
        seed = kwargs.pop('seed', None)
        return RandomPhaseRIS(num_elements, element_spacing=element_spacing, seed=seed)
    elif ris_type == "identity":
        # Identity is just constant phase with phase = 0
        return ConstantPhaseRIS(num_elements, element_spacing=element_spacing, constant_phase=0.0)
    else:
        raise ValueError(f"Unknown RIS type: {ris_type}")
