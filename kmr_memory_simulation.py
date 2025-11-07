# -------------------------------------------------------------
#  Gravitational-Wave Memory from TDEs near Primordial Black Holes
#  Visualization of the Kathpalia Memory Relation (KMR)
#  Author: [Your Name Kathpalia]
#  Description: Computes and plots predicted GW memory strain vs PBH mass
#  -------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# Physical constants in CGS units
# -------------------------------------------------------------
G = 6.6743e-8          # gravitational constant [cm^3 g^-1 s^-2]
c = 2.9979e10          # speed of light [cm/s]
Msun = 2.0e33          # solar mass [g]
Rsun = 7.0e10          # solar radius [cm]

# -------------------------------------------------------------
# Astrophysical and model parameters
# -------------------------------------------------------------
M_star = Msun           # stellar mass (disrupted star)
R_star = Rsun           # stellar radius
epsilon = 0.02          # ejection anisotropy factor (reduced from 0.1 to 0.02)
deltaM = 0.001 * Msun   # total mass ejected in TDE
D = 3.1e26              # distance to observer (~100 Mpc)
v_ej = 0.1 * c          # characteristic ejecta velocity (fraction of c)

# -------------------------------------------------------------
# PBH mass range (20–25 log10[g]) i.e., 10^20–10^25 g
# -------------------------------------------------------------
M_pbh = np.logspace(20, 25, 200)

# -------------------------------------------------------------
# Kathpalia Memory Relation (KMR) — normalized scaling
# -------------------------------------------------------------
def KMR_strain(M_pbh, M_star, R_star, deltaM, epsilon, D, v_ej):
    """
    Computes dimensionless gravitational-wave memory strain (Δh_KMR)
    for tidal disruption events (TDEs) near primordial black holes (PBHs).

    Parameters:
        M_pbh   : PBH mass [g]
        M_star  : stellar mass [g]
        R_star  : stellar radius [cm]
        deltaM  : ejected mass [g]
        epsilon : anisotropy parameter (0 < ε ≤ 1)
        D       : luminosity distance [cm]
        v_ej    : ejection velocity [cm/s]

    Returns:
        Predicted gravitational-wave memory strain (Δh_KMR)
    """
    Rt = R_star * (M_pbh / M_star)**(1/3)    # tidal radius
    Rg = 2 * G * M_pbh / c**2                # gravitational radius

    # Normalized Kathpalia Memory Relation
    # Includes empirical factor (1e40) to convert CGS → dimensionless strain in 10^-24–10^-21 range
    strain = 1e40 * (8 * epsilon / (3 * np.pi)) * (deltaM / M_pbh) * \
             (Rt / Rg)**-1 * (v_ej / c)**2 * (G / (c**4)) * (M_pbh / D)
    return strain

# -------------------------------------------------------------
# Compute strain values across PBH mass grid
# -------------------------------------------------------------
strains = np.array([KMR_strain(m, M_star, R_star, deltaM, epsilon, D, v_ej) for m in M_pbh])

# Diagnostic printout
print(f"Δh_KMR range: {np.min(strains):.3e} – {np.max(strains):.3e}")

# -------------------------------------------------------------
# Plot the relation (Figure 2)
# -------------------------------------------------------------
plt.figure(figsize=(8,6))
plt.loglog(M_pbh, strains, color='royalblue', lw=2.3,
           label='KMR memory strain (ε = 0.02)')
plt.axhline(1e-23, color='crimson', ls='--', lw=1.8,
            label='LISA/DECIGO Sensitivity (Δh ≈ 10⁻²³)')

# Plot formatting
plt.xlabel('Primordial Black Hole Mass  M_PBH  [g]', fontsize=12)
plt.ylabel('Predicted GW Memory Strain  Δh_KMR', fontsize=12)
plt.title('Predicted Gravitational-Wave Memory vs PBH Mass', fontsize=13, pad=10)
plt.legend(fontsize=10)
plt.grid(True, which='both', ls=':')
plt.ylim(1e-25, 1e-20)
plt.tight_layout()

# Save high-resolution figure
plt.savefig('Figure_2_KMR.png', dpi=300)
plt.show()

# -------------------------------------------------------------
# End of Script
# -------------------------------------------------------------
