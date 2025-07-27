import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import curve_fit
import pandas as pd

class DeurDarkEnergyModel:
    def __init__(self):
        """Initialize parameters for Deur's model"""
        # Cosmological parameters
        self.H0 = 68.0  # Hubble constant in km/s/Mpc
        self.c = 299792.458  # Speed of light in km/s
        
        # Model parameters from the paper
        self.xi = 0.9  # ± 0.1
        self.Rg = 0.15  # ± 0.10 (fraction of baryonic mass in galaxies)
        
        # Galaxy parameters
        self.zg0 = 9.0  # ± 1
        self.tau_g = 3.0  # ± 0.5
        self.Ag = 0.1  # ± 0.1
        self.Bg = 4.0  # ± 1
        
        # Cluster parameters
        self.zc0 = 5.6  # ± 1
        self.tau_c = 2.2  # ± 0.5
        self.Ac = 0.3  # ± 0.15
        self.Bc = 4.8  # ± 1.6
        
    def fermi_dirac_function(self, z, z0, tau, A, B):
        """Fermi-Dirac-like function for depletion factor components"""
        fd_term = 1 / (1 + np.exp((z - z0) / tau))
        exp_term = A * np.exp(-B * z)
        return 1 - fd_term + exp_term
    
    def D_galaxy(self, z):
        """Galaxy contribution to depletion factor"""
        return self.fermi_dirac_function(z, self.zg0, self.tau_g, self.Ag, self.Bg)
    
    def D_cluster(self, z):
        """Cluster contribution to depletion factor"""
        return self.fermi_dirac_function(z, self.zc0, self.tau_c, self.Ac, self.Bc)
    
    def D_M(self, z):
        """Total matter depletion factor D_M(z) - Equation (25)"""
        Rc = 1 - self.Rg  # Fraction in clusters
        return self.xi * (self.Rg * self.D_galaxy(z) + Rc * self.D_cluster(z))
    
    def integrand_luminosity_distance(self, x, z):
        """Integrand for luminosity distance calculation"""
        # x = 1/(1+z')
        z_prime = (1/x) - 1
        
        # Screened density fraction
        Omega_M_star = self.D_M(z_prime)  # Assuming Omega_M = 1 for simplicity
        
        # For flat universe (Omega_K = 0)
        return 1 / (x**2 * np.sqrt(Omega_M_star * x**(-3)))
    
    def luminosity_distance(self, z):
        """Calculate luminosity distance - Equation (23)"""
        # Integrate from x=1/(1+z) to x=1
        x_lower = 1 / (1 + z)
        result, _ = quad(self.integrand_luminosity_distance, x_lower, 1, args=(z,))
        
        # Convert to luminosity distance
        D_L = (self.c / self.H0) * (1 + z) * result
        return D_L
    
    def distance_modulus(self, z):
        """Calculate distance modulus for comparison with supernova data"""
        D_L = self.luminosity_distance(z)
        return 5 * np.log10(D_L) + 25
    
    def plot_depletion_factor(self):
        """Plot the depletion factor D_M(z) as in Figure 1"""
        z_range = np.linspace(0, 20, 1000)
        D_M_values = [self.D_M(z) for z in z_range]
        
        plt.figure(figsize=(10, 6))
        plt.plot(z_range, D_M_values, 'b-', linewidth=2)
        plt.xlabel('z', fontsize=12)
        plt.ylabel('$D_M(z)$', fontsize=12)
        plt.title("Depletion Factor $D_M(z)$ (Deur's Model)", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 20)
        plt.ylim(0, 0.8)
        plt.show()
    
    def plot_hubble_diagram(self, sn_data=None):
        """Plot Hubble diagram comparing with supernova data"""
        z_range = np.logspace(-2, 0.3, 100)
        
        # Calculate predictions
        mu_deur = [self.distance_modulus(z) for z in z_range]
        
        # For comparison: ΛCDM model (Omega_M=0.3, Omega_Lambda=0.7)
        mu_lcdm = self.distance_modulus_lcdm(z_range, 0.3, 0.7)
        
        # Matter-only universe
        mu_matter = self.distance_modulus_lcdm(z_range, 1.0, 0.0)
        
        plt.figure(figsize=(10, 8))
        
        # Plot theoretical curves
        plt.plot(z_range, mu_deur, 'b-', linewidth=2, label="Deur's Model")
        plt.plot(z_range, mu_lcdm, 'r--', linewidth=2, label='ΛCDM')
        plt.plot(z_range, mu_matter, 'g:', linewidth=2, label='Matter only')
        
        # Add simulated supernova data if provided
        if sn_data is not None:
            plt.errorbar(sn_data['z'], sn_data['mu'], yerr=sn_data['err'],
                        fmt='o', color='black', markersize=4, alpha=0.6,
                        label='Supernova data')
        
        plt.xlabel('Redshift z', fontsize=12)
        plt.ylabel('Distance Modulus μ', fontsize=12)
        plt.title('Hubble Diagram Comparison', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0.01, 2)
        plt.ylim(32, 46)
        plt.xscale('log')
        plt.show()
    
    def distance_modulus_lcdm(self, z, Omega_M, Omega_Lambda):
        """Calculate distance modulus for ΛCDM model for comparison"""
        def integrand(z_prime):
            return 1 / np.sqrt(Omega_M * (1 + z_prime)**3 + Omega_Lambda)
        
        D_L = np.zeros_like(z)
        for i, zi in enumerate(z):
            integral, _ = quad(integrand, 0, zi)
            D_L[i] = (self.c / self.H0) * (1 + zi) * integral
        
        return 5 * np.log10(D_L) + 25
    
    def generate_mock_sn_data(self, n_points=50):
        """Generate mock supernova data for testing"""
        z = np.logspace(-2, 0.2, n_points)
        # Use ΛCDM as "true" model for mock data
        mu_true = self.distance_modulus_lcdm(z, 0.3, 0.7)
        # Add realistic errors
        errors = 0.15 + 0.1 * z  # Errors increase with redshift
        mu_obs = mu_true + np.random.normal(0, errors)
        
        return pd.DataFrame({'z': z, 'mu': mu_obs, 'err': errors})

# Example usage
if __name__ == "__main__":
    # Create model instance
    model = DeurDarkEnergyModel()
    
    # Plot the depletion factor
    print("Plotting depletion factor D_M(z)...")
    model.plot_depletion_factor()
    
    # Generate mock supernova data
    print("Generating mock supernova data...")
    sn_data = model.generate_mock_sn_data()
    
    # Plot Hubble diagram
    print("Plotting Hubble diagram...")
    model.plot_hubble_diagram(sn_data)
    
    # Test specific redshift values
    test_z = [0.1, 0.5, 1.0, 1.5]
    print("\nDepletion factor D_M(z) at specific redshifts:")
    for z in test_z:
        print(f"z = {z:.1f}: D_M = {model.D_M(z):.3f}")
    
    # Compare distance moduli
    print("\nDistance modulus comparison at z=1:")
    z_test = 1.0
    mu_deur = model.distance_modulus(z_test)
    mu_lcdm = model.distance_modulus_lcdm(np.array([z_test]), 0.3, 0.7)[0]
    print(f"Deur's model: μ = {mu_deur:.2f}")
    print(f"ΛCDM model: μ = {mu_lcdm:.2f}")
    print(f"Difference: Δμ = {mu_deur - mu_lcdm:.2f}")
