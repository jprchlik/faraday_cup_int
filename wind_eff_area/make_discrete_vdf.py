import numpy as np

def make_discrete_vdf(pls_par,mag_par,pres=0.5,qres=0.5,clip=4.):
    """
    Returns Discrete Velocity distribution function given a set of input parameters.

    Parameters:
    -----------
    pls_par: np.array
        A numpy array of plasma parameters in the following order: Vx,Vy,Vz,Wper,Wpar,Np.
        That is the proton velocity in the solar wind in X GSE, Y GSE, and Z GSE in km/s,
        followed by the thermal width perpendicular and paralle to the magnetic field normal,
        and finally the proton number density in cm^{-3}
    
    mag_par: np.array
        A numpy array of the magnetic field normal in the following order : bx, by, and bz.
        The normal vectors should be defined in GSE coordinates.

    pres: float, optional
        The resolution of the instrument in the parallel direction in km/s (Default = 0.5).

    qres: float, optional
        The resolution of the instrument in the perpendicular direction in km/s (Default = 0.5).

    clip: float, optional
        The measurement width beyond input velocity width as a sigma value (Default = 4). 
    """

    #Set up names for easy call
    u_gse = pls_par[:3]
    wper  = pls_par[3]
    wpar  = pls_par[4]
    n     = pls_par[5]

    
    
    #distribution of velocities in the parallel direction
    p = np.arange(-wpar*msig,(wpar*msig)+pres,pres)
    #distribution of velocities in the perpendicular direction
    q = np.arange(0,(wper*msig)+qres,qres)
    
    
    #created 2D grid of velocities in the X and y direction
    pgrid, qgrid = np.meshgrid(p,q)
    
    #Get VDF constance
    a = n/(np.sqrt(np.pi**3.)*(wpar*wper**2.)) # 1/cm^3 * s^3 /km^3 
    
    #compute the raw vdf
    rawvdf = a*np.exp(- (pgrid/wpar)**2. - (qgrid/wper)**2.)
    
    dis_vdf = {'vdf':rawvdf,'pgrid':pgrid,'qgrid':qgrid,'u_gse':u_gse,'b_gse':mag_par}
    return dis_vdf


def plot_vdf(dis_vdf):
    from matplotlib.pyplot import pcolormesh

    fig, ax = plt.subplots()

    ax.pcolormesh(dis_vdf['pgrid'],dis_vdf['qgrid'],np.log10(dis_vdf['vdf']))

    return fig,ax


