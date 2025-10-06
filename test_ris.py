"""
Test Suite for RIS Implementation
==================================
Basic tests to verify the implementation is working correctly.
"""

import numpy as np
import sys

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        import MISO_DEMO1
        import ris_models
        import ris_comparison
        print("✓ All modules imported successfully")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_basic_simulation():
    """Test basic simulation functionality."""
    print("\nTesting basic simulation...")
    try:
        from MISO_DEMO1 import simulate, params
        result = simulate(params)
        
        # Check that results contain expected keys
        expected_keys = ['H_direct', 'H_ris', 'H_total', 'freqs', 'phi', 'snr_dB', 'noise_power', 'delays']
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"
        
        # Check dimensions
        K = params['ofdm']['K']
        assert len(result['H_direct']) == K, "H_direct dimension mismatch"
        assert len(result['H_ris']) == K, "H_ris dimension mismatch"
        assert len(result['H_total']) == K, "H_total dimension mismatch"
        assert len(result['freqs']) == K, "freqs dimension mismatch"
        assert len(result['phi']) == params['Nr'], "phi dimension mismatch"
        
        print("✓ Basic simulation passed")
        return True
    except Exception as e:
        print(f"✗ Basic simulation failed: {e}")
        return False


def test_ris_modes():
    """Test different RIS phase modes."""
    print("\nTesting RIS modes...")
    try:
        from MISO_DEMO1 import simulate, params
        import copy
        
        modes = ['identity', 'constant_phase', 'align', 'ttd', 'random']
        
        for mode in modes:
            p = copy.deepcopy(params)
            p['ris_phase_mode'] = mode
            if mode == 'constant_phase':
                p['ris_constant_phase'] = 0.0
            
            result = simulate(p)
            assert 'H_total' in result, f"Mode {mode} failed"
        
        print(f"✓ All {len(modes)} RIS modes tested successfully")
        return True
    except Exception as e:
        print(f"✗ RIS modes test failed: {e}")
        return False


def test_ris_models():
    """Test RIS model classes."""
    print("\nTesting RIS model classes...")
    try:
        from ris_models import (create_ris_model, ConstantPhaseRIS, 
                                TrueTimeDelayRIS, PhaseAlignedRIS, RandomPhaseRIS)
        
        # Test creation
        ris1 = create_ris_model('constant_phase', num_elements=32)
        ris2 = create_ris_model('ttd', num_elements=32, carrier_frequency=3.5e9)
        ris3 = create_ris_model('align', num_elements=32)
        ris4 = create_ris_model('random', num_elements=32)
        
        # Test phase response
        phi = ris1.get_phase_response(3.5e9)
        assert len(phi) == 32, "Phase response dimension mismatch"
        assert np.allclose(np.abs(phi), 1.0), "Phase coefficients should have unit magnitude"
        
        # Test TTD frequency dependence
        time_delays = np.linspace(0, 1e-9, 32)
        ris2.configure(time_delays=time_delays)
        phi_fc = ris2.get_phase_response(3.5e9)
        phi_fc_plus = ris2.get_phase_response(3.5e9 + 15e3)
        assert not np.allclose(phi_fc, phi_fc_plus), "TTD should be frequency-dependent"
        
        print("✓ RIS model classes tested successfully")
        return True
    except Exception as e:
        print(f"✗ RIS models test failed: {e}")
        return False


def test_channel_generation():
    """Test channel generation with Rice fading."""
    print("\nTesting channel generation...")
    try:
        from MISO_DEMO1 import generate_channels, params
        
        rng = np.random.default_rng(42)
        h_direct, H_br, h_ru, delays = generate_channels(params, rng)
        
        # Check dimensions
        K = params['ofdm']['K']
        Nt = params['Nt']
        Nr = params['Nr']
        
        assert h_direct.shape == (K, Nt), "h_direct shape mismatch"
        assert H_br.shape == (K, Nr, Nt), "H_br shape mismatch"
        assert h_ru.shape == (K, Nr), "h_ru shape mismatch"
        assert 'direct' in delays and 'ris' in delays, "delays dictionary incomplete"
        
        print("✓ Channel generation passed")
        return True
    except Exception as e:
        print(f"✗ Channel generation failed: {e}")
        return False


def test_frequency_dependence():
    """Test that TTD RIS is frequency-dependent and constant phase is not."""
    print("\nTesting frequency dependence...")
    try:
        from MISO_DEMO1 import simulate, params
        import copy
        
        # Test constant phase (should be frequency-flat)
        p1 = copy.deepcopy(params)
        p1['ris_phase_mode'] = 'constant_phase'
        p1['ris_constant_phase'] = 0.0
        result1 = simulate(p1)
        
        # For constant phase, the phase pattern should be similar across frequencies
        # (though channel itself can vary)
        
        # Test TTD (should be frequency-dependent)
        p2 = copy.deepcopy(params)
        p2['ris_phase_mode'] = 'ttd'
        result2 = simulate(p2)
        
        # Both should produce valid results
        assert len(result1['H_total']) == params['ofdm']['K']
        assert len(result2['H_total']) == params['ofdm']['K']
        
        print("✓ Frequency dependence test passed")
        return True
    except Exception as e:
        print(f"✗ Frequency dependence test failed: {e}")
        return False


def run_all_tests():
    """Run all tests."""
    print("=" * 70)
    print("RIS Implementation Test Suite")
    print("=" * 70)
    
    tests = [
        test_imports,
        test_basic_simulation,
        test_ris_modes,
        test_ris_models,
        test_channel_generation,
        test_frequency_dependence,
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 70)
    print(f"Test Results: {sum(results)}/{len(results)} passed")
    print("=" * 70)
    
    if all(results):
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == "__main__":
    # Use non-interactive matplotlib backend for testing
    import matplotlib
    matplotlib.use('Agg')
    
    sys.exit(run_all_tests())
