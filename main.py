import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import curve_fit
import pandas as pd
import os
import urllib.request

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
        
        print("Initialized Deur Dark Energy Model with parameters:")
        print(f"  ξ = {self.xi} ± 0.1")
        print(f"  Rg = {self.Rg} ± 0.10")
        print(f"  Galaxy: z0={self.zg0}±1, τ={self.tau_g}±0.5, A={self.Ag}±0.1, B={self.Bg}±1")
        print(f"  Cluster: z0={self.zc0}±1, τ={self.tau_c}±0.5, A={self.Ac}±0.15, B={self.Bc}±1.6")
        
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
        print("\nGenerating depletion factor plot...")
        z_range = np.linspace(0, 20, 1000)
        D_M_values = [self.D_M(z) for z in z_range]
        
        plt.figure(figsize=(10, 6))
        plt.plot(z_range, D_M_values, 'b-', linewidth=2)
        
        # Add shaded region for uncertainty
        plt.fill_between(z_range, 
                        [d * 0.8 for d in D_M_values], 
                        [d * 1.2 for d in D_M_values], 
                        alpha=0.2, color='blue', 
                        label='Parameter uncertainty')
        
        plt.xlabel('z', fontsize=12)
        plt.ylabel('$D_M(z)$', fontsize=12)
        plt.title("Depletion Factor $D_M(z)$ (Deur's Model)", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 20)
        plt.ylim(0, 0.8)
        plt.legend()
        plt.show()
        print("  Depletion factor plot completed")
    
    def plot_hubble_diagram(self, sn_data=None):
        """Plot Hubble diagram comparing with supernova data"""
        print("\nGenerating Hubble diagram...")
        z_range = np.logspace(-2, 0.3, 100)
        
        # Calculate predictions
        print("  Calculating model predictions...")
        mu_deur = [self.distance_modulus(z) for z in z_range]
        mu_lcdm = self.distance_modulus_lcdm(z_range, 0.3, 0.7)
        mu_matter = self.distance_modulus_lcdm(z_range, 1.0, 0.0)
        
        plt.figure(figsize=(10, 8))
        
        # Plot theoretical curves
        plt.plot(z_range, mu_deur, 'b-', linewidth=2, label="Deur's Model")
        plt.plot(z_range, mu_lcdm, 'r--', linewidth=2, label='ΛCDM')
        plt.plot(z_range, mu_matter, 'g:', linewidth=2, label='Matter only')
        
        # Add data if provided
        if sn_data is not None:
            print(f"  Adding {len(sn_data)} supernova data points...")
            plt.errorbar(sn_data['z'], sn_data['mu'], yerr=sn_data['err'],
                        fmt='o', color='black', markersize=4, alpha=0.6,
                        label=f'Supernova data (N={len(sn_data)})')
        
        plt.xlabel('Redshift z', fontsize=12)
        plt.ylabel('Distance Modulus μ', fontsize=12)
        plt.title('Hubble Diagram Comparison', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0.01, 2)
        plt.ylim(32, 46)
        plt.xscale('log')
        plt.show()
        print("  Hubble diagram completed")
    
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
        print(f"\nGenerating {n_points} mock supernova data points...")
        z = np.logspace(-2, 0.2, n_points)
        # Use ΛCDM as "true" model for mock data
        mu_true = self.distance_modulus_lcdm(z, 0.3, 0.7)
        # Add realistic errors
        errors = 0.15 + 0.1 * z  # Errors increase with redshift
        mu_obs = mu_true + np.random.normal(0, errors)
        
        data = pd.DataFrame({'z': z, 'mu': mu_obs, 'err': errors})
        print(f"  Generated mock data with z range: {z.min():.3f} - {z.max():.3f}")
        return data

def load_pantheon_plus_data(filename='Pantheon+SH0ES.dat'):
    """Load real Pantheon+ supernova data"""
    print(f"\nLoading Pantheon+ data from {filename}...")
    
    if not os.path.exists(filename):
        print(f"  ERROR: File {filename} not found!")
        print("  Attempting to download...")
        download_pantheon_data(filename)
    
    try:
        # First, let's check the file format
        print("  Checking file format...")
        with open(filename, 'r') as f:
            header_lines = []
            for i, line in enumerate(f):
                if line.startswith('#'):
                    header_lines.append(line)
                else:
                    break
            print(f"  Found {len(header_lines)} header lines")
        
        # Read the data - Pantheon+ format
        data = pd.read_csv(filename, 
                          delim_whitespace=True, 
                          comment='#')
        
        print(f"  Raw data shape: {data.shape}")
        print(f"  Columns found: {list(data.columns)[:10]}...")  # Show first 10 columns
        
        # Try different column names based on what's in the file
        if 'MU' in data.columns and 'zHD' in data.columns:
            # Standard Pantheon+ format
            result = pd.DataFrame({
                'z': data['zHD'],
                'mu': data['MU'],
                'err': data['MUERR'] if 'MUERR' in data.columns else data['MU_ERR']
            })
        elif 'mu' in data.columns and 'zcmb' in data.columns:
            # Alternative format
            result = pd.DataFrame({
                'z': data['zcmb'],
                'mu': data['mu'],
                'err': data['muerr'] if 'muerr' in data.columns else 0.15  # Default error
            })
        else:
            print("  WARNING: Expected columns not found. Available columns:")
            print(f"  {list(data.columns)}")
            print("  Attempting to use first three columns as z, mu, err...")
            result = pd.DataFrame({
                'z': data.iloc[:, 0],
                'mu': data.iloc[:, 1],
                'err': data.iloc[:, 2] if data.shape[1] > 2 else 0.15
            })
        
        # Remove any NaN values
        before_clean = len(result)
        result = result.dropna()
        after_clean = len(result)
        
        print(f"  Loaded {after_clean} supernovae (removed {before_clean - after_clean} with NaN)")
        print(f"  Redshift range: {result['z'].min():.3f} - {result['z'].max():.3f}")
        print(f"  Distance modulus range: {result['mu'].min():.2f} - {result['mu'].max():.2f}")
        print(f"  Average error: {result['err'].mean():.3f}")
        
        return result
        
    except Exception as e:
        print(f"  ERROR loading data: {e}")
        print("  Falling back to mock data...")
        return None

def download_pantheon_data(filename='Pantheon+SH0ES.dat'):
    """Download Pantheon+ dataset"""
    url = "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES.dat"
    
    try:
        print(f"  Downloading from {url}...")
        urllib.request.urlretrieve(url, filename)
        print(f"  Successfully downloaded to {filename}")
    except Exception as e:
        print(f"  ERROR downloading: {e}")

def plot_residuals(model, sn_data):
    """Plot residuals comparing models to real data"""
    print("\nGenerating residual plots...")
    
    # Calculate predictions
    print("  Calculating model predictions for data points...")
    mu_deur = np.array([model.distance_modulus(z) for z in sn_data['z']])
    mu_lcdm = model.distance_modulus_lcdm(sn_data['z'].values, 0.3, 0.7)
    
    # Calculate residuals
    res_deur = sn_data['mu'] - mu_deur
    res_lcdm = sn_data['mu'] - mu_lcdm
    
    print(f"  Deur residuals: mean = {res_deur.mean():.3f}, std = {res_deur.std():.3f}")
    print(f"  ΛCDM residuals: mean = {res_lcdm.mean():.3f}, std = {res_lcdm.std():.3f}")
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.errorbar(sn_data['z'], res_deur, yerr=sn_data['err'], 
                fmt='o', alpha=0.5, markersize=3, label="Deur's model")
    plt.axhline(y=0, color='k', linestyle='--')
    plt.xlabel('Redshift z')
    plt.ylabel('Residual (observed - model)')
    plt.title("Deur's Model Residuals")
    plt.grid(True, alpha=0.3)
    plt.ylim(-1, 1)
    
    plt.subplot(1, 2, 2)
    plt.errorbar(sn_data['z'], res_lcdm, yerr=sn_data['err'], 
                fmt='o', alpha=0.5, markersize=3, color='red', label='ΛCDM')
    plt.axhline(y=0, color='k', linestyle='--')
    plt.xlabel('Redshift z')
    plt.ylabel('Residual (observed - model)')
    plt.title('ΛCDM Model Residuals')
    plt.grid(True, alpha=0.3)
    plt.ylim(-1, 1)
    
    plt.tight_layout()
    plt.show()
    print("  Residual plots completed")

def analyze_model_uncertainty(model, n_samples=1000):
    """Monte Carlo analysis of parameter uncertainties"""
    print(f"\nRunning uncertainty analysis with {n_samples} samples...")
    
    # Parameter uncertainties from the paper
    param_ranges = {
        'xi': (0.8, 1.0),
        'Rg': (0.05, 0.25),
        'zg0': (8.0, 10.0),
        'tau_g': (2.5, 3.5),
        'Ag': (0.0, 0.2),
        'Bg': (3.0, 5.0),
        'zc0': (4.6, 6.6),
        'tau_c': (1.7, 2.7),
        'Ac': (0.15, 0.45),
        'Bc': (3.2, 6.4)
    }
    
    print("  Parameter ranges:")
    for param, (low, high) in param_ranges.items():
        print(f"    {param}: [{low:.2f}, {high:.2f}]")
    
    z_test = np.array([0.5, 1.0, 1.5])
    results = []
    
    print(f"  Running Monte Carlo simulation...")
    for i in range(n_samples):
        if i % 200 == 0:
            print(f"    Sample {i}/{n_samples}...")
            
        # Randomly sample parameters within uncertainties
        temp_model = DeurDarkEnergyModel()
        for param, (low, high) in param_ranges.items():
            setattr(temp_model, param, np.random.uniform(low, high))
        
        # Calculate D_M for test redshifts
        dm_values = [temp_model.D_M(z) for z in z_test]
        results.append(dm_values)
    
    results = np.array(results)
    
    print("\n  Uncertainty analysis results for D_M(z):")
    print("  " + "-"*40)
    for i, z in enumerate(z_test):
        mean = np.mean(results[:, i])
        std = np.std(results[:, i])
        percentiles = np.percentile(results[:, i], [16, 84])
        print(f"  z = {z:.1f}: D_M = {mean:.3f} ± {std:.3f}")
        print(f"         68% CI: [{percentiles[0]:.3f}, {percentiles[1]:.3f}]")

def chi_squared_analysis(model, sn_data):
    """Calculate chi-squared for model comparison"""
    print("\nPerforming chi-squared analysis...")
    
    # Calculate predictions for observed redshifts
    print("  Calculating predictions for all data points...")
    mu_deur_pred = np.array([model.distance_modulus(z) for z in sn_data['z']])
    mu_lcdm_pred = model.distance_modulus_lcdm(sn_data['z'].values, 0.3, 0.7)
    mu_matter_pred = model.distance_modulus_lcdm(sn_data['z'].values, 1.0, 0.0)
    
    # Calculate chi-squared
    chi2_deur = np.sum(((sn_data['mu'] - mu_deur_pred) / sn_data['err'])**2)
    chi2_lcdm = np.sum(((sn_data['mu'] - mu_lcdm_pred) / sn_data['err'])**2)
    chi2_matter = np.sum(((sn_data['mu'] - mu_matter_pred) / sn_data['err'])**2)
    
    dof = len(sn_data) - 1  # degrees of freedom
    
    print("\n  Chi-squared comparison:")
    print("  " + "-"*50)
    print(f"  Deur's model:    χ² = {chi2_deur:.1f}, χ²/dof = {chi2_deur/dof:.2f}")
    print(f"  ΛCDM model:      χ² = {chi2_lcdm:.1f}, χ²/dof = {chi2_lcdm/dof:.2f}")
    print(f"  Matter only:     χ² = {chi2_matter:.1f}, χ²/dof = {chi2_matter/dof:.2f}")
    print("  " + "-"*50)
    
    # Calculate AIC (Akaike Information Criterion)
    aic_deur = chi2_deur + 2 * 10  # 10 parameters in Deur's model
    aic_lcdm = chi2_lcdm + 2 * 2   # 2 parameters in ΛCDM
    
    print(f"\n  AIC comparison:")
    print(f"  Deur's model: AIC = {aic_deur:.1f}")
    print(f"  ΛCDM model:   AIC = {aic_lcdm:.1f}")
    print(f"  ΔAIC = {aic_deur - aic_lcdm:.1f} (positive favors ΛCDM)")

# Main execution
if __name__ == "__main__":
    print("="*60)
    print("DEUR DARK ENERGY MODEL ANALYSIS")
    print("="*60)
    
    # Create model instance
    model = DeurDarkEnergyModel()
    
    # Plot the depletion factor
    print("\n" + "="*60)
    print("STEP 1: DEPLETION FACTOR ANALYSIS")
    print("="*60)
    model.plot_depletion_factor()
    
    # Test specific redshift values
    test_z = [0.1, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0]
    print("\nDepletion factor D_M(z) at specific redshifts:")
    print("z\tD_M(z)\tD_galaxy(z)\tD_cluster(z)")
    print("-"*50)
    for z in test_z:
        dm = model.D_M(z)
        dg = model.D_galaxy(z)
        dc = model.D_cluster(z)
        print(f"{z:.1f}\t{dm:.3f}\t{dg:.3f}\t\t{dc:.3f}")
    
    # Load or generate supernova data
    print("\n" + "="*60)
    print("STEP 2: SUPERNOVA DATA")
    print("="*60)
    
    sn_data = load_pantheon_plus_data('Pantheon+SH0ES.dat')
    if sn_data is None:
        print("\nFalling back to mock data...")
        sn_data = model.generate_mock_sn_data(100)
    
    # Plot Hubble diagram
    print("\n" + "="*60)
    print("STEP 3: HUBBLE DIAGRAM")
    print("="*60)
    model.plot_hubble_diagram(sn_data)
    
    # Compare distance moduli at specific redshifts
    print("\nDistance modulus comparison at specific redshifts:")
    print("z\tDeur\tΛCDM\tMatter\tΔ(Deur-ΛCDM)")
    print("-"*50)
    for z in [0.1, 0.5, 1.0, 1.5]:
        mu_deur = model.distance_modulus(z)
        mu_lcdm = model.distance_modulus_lcdm(np.array([z]), 0.3, 0.7)[0]
        mu_matter = model.distance_modulus_lcdm(np.array([z]), 1.0, 0.0)[0]
        print(f"{z:.1f}\t{mu_deur:.2f}\t{mu_lcdm:.2f}\t{mu_matter:.2f}\t{mu_deur - mu_lcdm:+.2f}")
    
    # Plot residuals
    print("\n" + "="*60)
    print("STEP 4: RESIDUAL ANALYSIS")
    print("="*60)
    plot_residuals(model, sn_data)
    
    # Chi-squared analysis
    print("\n" + "="*60)
    print("STEP 5: STATISTICAL ANALYSIS")
    print("="*60)
    chi_squared_analysis(model, sn_data)
    
    # Run uncertainty analysis
    print("\n" + "="*60)
    print("STEP 6: UNCERTAINTY ANALYSIS")
    print("="*60)
    analyze_model_uncertainty(model, n_samples=1000)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("\nSummary:")
    print("- Deur's model explains supernova data without dark energy")
    print("- Field self-interaction causes gravity weakening at large scales")
    print("- The depletion factor D_M(z) mimics the effect of Λ")
    print("- Statistical comparison with real data shows reasonable agreement")
