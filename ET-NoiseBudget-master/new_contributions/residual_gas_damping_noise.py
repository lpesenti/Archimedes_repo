# Plot of residual gas pressure on 4 mirrors of a FPMI. 
# Rai Weiss's equation from LIGO Document T0900509-v1
# Included Cavalleri et al equation (https://doi.org/10.1016/j.physleta.2010.06.041)
# more details in logbook ("Residual gas pressure: molecule impact on mirrors (revisited)")

import numpy as np
import matplotlib.pyplot as plt

c = 2.99792458e8        # speed of light [m / s]
kb = 1.38064852e-23  # Boltzmann cosntant [m2 kg s-2 K-1]
NA = 6.0221409e+23   # Avogadro's number
pi = np.pi

def S_F_weiss(M, r, p, T, m):
    """
    Returns the freq. independent force PSD on a single test mass due to impinging residual gas particles as given by T0900509
    r = mirror radius [m]
    p = pressure of gas [Pa]
    m = molecular mass of particle [kg] (default = 2.99e-26 for water)
    T= temperaure [K] (default = 300K)
    """
    S_F = 8 * p * np.sqrt(kb*T*m) * np.pi * r**2
    return S_F


def S_F_cavalleri(M, r, p, T, m):
    """
    Returns the freq. independent force PSD on a single test mass due to impinging residual gas particles as given by  Cavalleri et al 2009 (https://doi.org/10.1016/j.physleta.2010.06.041)
    Assumes radius to thickness ratio of mirror to be 1:1
    r = mirror radius [m]
    p = pressure of gas [Pa]
    m = molecular mass of particle [kg] (default = 2.99e-26 for water)
    T= temperaure [K] (default = 300K)
    """
    S_F = p * np.sqrt((128/np.pi)*m*kb*T) * np.pi * r**2 * (1 + r/(2*r) + np.pi/4)
    return S_F

def calc_x_noise(f, S_F, M):
    """
    Returns the displacement noise for a single test mass for a given force PSD. 
    S_F = PSD of force noise
    M = mass of test mass [kg]
    f = frequency [Hz]
    """  
    x = np.sqrt(S_F) / (M * (2*np.pi*f)**2)
    return x




if __name__ == "__main__":
    # Plot comparision between Weiss and Cavalleri equation.
    m = 2.99e-26/18        #  mass of water moleecule [kg]
    T = 300             # temperature [K]
    M = 200          # 100g mirrors [kg]
    r = 0.31           # mirror radius [m]

    f = np.logspace(0,2.5,int(1e5))    # freqeuncy [Hz]
    

# =============================================================================
#     M_ligo = 200
#     r_ligo = 0.275
#     L_ligo = 10e3
#     p_LIGO = 1e-10*100  # [Pa]
#     m_H2 = 2*1.66e-27       #[kg]
# =============================================================================

# =============================================================================
# AEI 10m Prototype
#     M_pt = 200      # 100g mirrors [kg]
#     r_pt = 0.31   # 24.4 mm 
#     L_pt = 1       # 10 m  
#     p_pt = 1e-9*100   # [Pa] (1mbar = 100Pa)
# =============================================================================
    
###################
# ET
###################
    M_ET = 200
    r_ET = 0.31
    L_ET = 1e4
    p_ET = 1e-10*100  # [Pa]  1e-10 mbar
    m_H2 = 2*1.66e-27       #[kg]
    x_ET_weiss = calc_x_noise(f, \
                           S_F_weiss(M_ET, r_ET, p_ET, T, m=m_H2)\
                           , M_ET)
    
    x_ET_cavalleri = calc_x_noise(f, \
                           S_F_cavalleri(M_ET, r_ET, p_ET, T, m=m_H2)\
                           , M_ET)
    
    fig1 = plt.figure(1, figsize=(6,6))
    fig1.clf()
    ax1 = fig1.gca()
    ax1.loglog(f, 2*x_ET_weiss/L_ET, lw=3, label='Weiss equation')
    #ax1.loglog(f, (1.6e-20)/f**2, '--' ,color='C0', lw=2, label='plot from Weiss paper \n (includes squeeze damping)')
    #ax1.loglog(f, (100*4*1.5e-24)/f**2, '--' ,color='C3', lw=2, label='plot from Weiss paper')
    ax1.loglog(f, 2*x_ET_cavalleri/L_ET, lw=3, label='Cavalleri equation')
    ax1.set_xlabel('Frequency [Hz]')
    ax1.set_ylabel(r'h [1 /$\sqrt{Hz}$]')
    ax1.set_title('ET: H2 at {:.0e} mbar'.format(p_ET/100))
    ax1.grid(which='both')
    ax1.autoscale(enable=True, axis='both', tight=True)
    ax1.legend()
# =============================================================================
# 
#     
#     x_pt_weiss = calc_x_noise(f, \
#                            S_F_weiss(M_ET, r_ET, p_ET)\
#                            , M_ET)
#     
#     x_pt_cavalleri = calc_x_noise(f, \
#                            S_F_cavalleri(M_ET, r_ET, p_ET)\
#                            , M_ET)
# =============================================================================
    
    fig2 = plt.figure(2, figsize=(6,6))
    fig2.clf()
    ax2 = fig2.gca()
    ax2.loglog(f, 2*x_ET_weiss, lw=3, label='Weiss equation')
    ax2.loglog(f, 2*x_ET_cavalleri, lw=3, label='Cavalleri equation')
    ax2.set_xlabel('Frequency [Hz]')
    ax2.set_ylabel(r'Displacement noise (4 mirrors) [m /$\sqrt{Hz}$]')
    ax2.set_title('ET displacement P_H2 at {:.0e} mbar'.format(p_ET/100))
    ax2.grid(which='both')
    ax2.autoscale(enable=True, axis='both', tight=True)
    ax2.legend()
    
    plt.show() 


