#!/usr/bin/env python3
"""
test_noise_ceiling.py

Simple test script to demonstrate noise ceiling and LOSO functionality
as suggested by the Reddit community.

Usage:
  python test_noise_ceiling.py
"""

import numpy as np
from correlation_analysis import noise_ceiling, loso_lower_bound


def generate_test_data(n_subjects: int = 5, n_conditions: int = 6, noise_level: float = 0.1) -> np.ndarray:
    """
    Generate synthetic data for testing noise ceiling functions.
    
    Args:
        n_subjects: Number of simulated subjects
        n_conditions: Number of experimental conditions  
        noise_level: Amount of subject-to-subject variability
        
    Returns:
        Data matrix of shape (n_subjects, n_conditions)
    """
    # True underlying signal (same for all subjects)
    true_signal = np.linspace(0.3, 0.9, n_conditions)
    
    # Add subject-specific noise
    data_matrix = np.zeros((n_subjects, n_conditions))
    for i in range(n_subjects):
        subject_noise = np.random.normal(0, noise_level, n_conditions)
        data_matrix[i] = true_signal + subject_noise
    
    return data_matrix


def main():
    """Test noise ceiling and LOSO functions with synthetic data."""
    
    print("=== NOISE CEILING & LOSO DEMONSTRATION ===\n")
    
    # Test with different noise levels
    noise_levels = [0.05, 0.1, 0.2, 0.3]
    
    for noise in noise_levels:
        print(f"Noise level: {noise:.2f}")
        
        # Generate test data
        np.random.seed(42)  # For reproducibility
        data = generate_test_data(n_subjects=8, n_conditions=6, noise_level=noise)
        
        # Compute noise ceiling
        upper_nc, lower_nc = noise_ceiling(data)
        
        # Compute LOSO lower bound
        loso_mse = loso_lower_bound(data)
        
        print(f"  Noise ceiling (upper): {upper_nc:.3f}")
        print(f"  Noise ceiling (lower): {lower_nc:.3f}")
        print(f"  LOSO MSE: {loso_mse:.4f}")
        print(f"  Ceiling width: {upper_nc - lower_nc:.3f}")
        print()
    
    print("=== INTERPRETATION ===")
    print("• Noise ceiling bounds represent the best possible model performance")
    print("• Higher noise → wider ceiling, lower maximum achievable correlation")
    print("• LOSO MSE shows prediction error using group average")
    print("• Model correlations should fall within [lower_bound, upper_bound]")
    print("\n=== REAL DATA APPLICATION ===")
    print("Run: python correlation_analysis.py --full <data_dir> --out <output_dir> --noise_ceiling")
    print("This provides context for interpreting model performance relative to data reliability.")


if __name__ == "__main__":
    main()
