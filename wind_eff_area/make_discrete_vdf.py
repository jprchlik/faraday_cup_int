import numpy as np
from scipy.integrate import tplquad
from functools import partial
import scipy.interpolate as interp
from scipy.interpolate import RectBivariateSpline
from scipy.special import erf
import monte_carlo_int as mci
from scipy.stats import gennorm
import time


def solve_sing_decomp(phi,theta):
    """
    Returns singular decomposition matrix solutions to use when computing multi-spacecraft solutions to Vgse, Wper/par, and Np

    Parameters
    ----------

    phi: np.arary or float
        Phi angle values of FC in radians (must be the same length as theta)
    theta: np.array or float
        Theta angle values of FC in radians (must be the same length as phi)
    
    Returns
    -------

    u_svdc: np.array
        A three column by size of phi (i.e. number of FC) orthoginal array used for decomposition.
    w_svdc: np.array
        Three element vector containing the singular values
    v_svdc: np.array
        A three column by three row orthoginal array used for decomposition.
    wp_svdc: np.array
        A 3x3 diagonal matrix with the diagonal containing values of 1./w_svdc.  
    """

    #number of faraday cup angles
    ncup = phi.size
    
    #number of parameters in decomp
    npar=3
    #compute singular value decomp based on Wind observing pipeline
    #/crater/observatories/wind/code/dvapbimax/sub_bimax_moments.pro
    #create vector of FC observaitons
    r_vec = np.zeros((ncup,npar))
    #Include - sign in phi because of opposite of GSE phi (which is how phi is defined) when FC normal points towards the Sun
    #removed because I now fixed the conversion in GSE coordinates. It had to do with how Arfken, Weber, and Harris defined
    #phi and theta
    r_vec[:,0] = np.cos(phi) * np.cos(theta)
    r_vec[:,1] = np.sin(phi) * np.cos(theta)
    r_vec[:,2] = np.sin(theta)

    
    #compute singular value matrix unitary array
    u_svdc,w_svdc,v_svdc = np.linalg.svd(r_vec,compute_uv=True,full_matrices=False)
    
    #create identity matrix for calculating velocity in GSE
    wp_svdc = (1./w_svdc)*np.identity(npar)


    return u_svdc,w_svdc,v_svdc,wp_svdc

def compute_gse_from_fit(phi,theta,fit):
    """
    Returns singular decomposition matrix solutions to use when computing multi-spacecraft solutions to Vgse, Wper/par, and Np

    Parameters
    ----------
    phi: np.arary or float
        Phi angle values of FC in radians (must be the same length as theta)
    theta: np.array or float
        Theta angle values of FC in radians (must be the same length as phi)
    fit: float
        The fit parameter in FC normal coordinates 

    Returns
    ------
    best_vgse: np.array
        Array of best fit coordinates in GSE [x,y,z]     
    """

    #get decomp matrices
    u_svdc,w_svdc,v_svdc,wp_svdc = solve_sing_decomp(phi,theta)

    #Get "best fit" values in FC cooridnates
    best_vgse =  np.dot(np.dot(np.dot(v_svdc.T,wp_svdc),u_svdc.T),fit)
   
    return best_vgse

def make_discrete_vdf(pls_par,mag_par,pres=0.5,qres=0.5,clip=300.,add_ring=[]):
    """
    Returns Discrete Velocity distribution function given a set of input parameters.

    Parameters
    -----------
    pls_par: np.array
        A numpy array of plasma parameters in the following order: Vx,Vy,Vz,Wper,Wpar,Np.
        That is the proton velocity in the solar wind in X GSE, Y GSE, and Z GSE in km/s,
        followed by the thermal width perpendicular and parallel to the magnetic field normal,
        and finally the proton number density in cm^{-3}. Wper and Wpar are assumed to be 
        Sigma values.
    
    mag_par: np.array
        A numpy array of the magnetic field normal in the following order : bx, by, and bz.
        The normal vectors should be defined in GSE coordinates.

    pres: float, optional
        The resolution of the instrument in the parallel direction in km/s (Default = 0.5).

    qres: float, optional
        The resolution of the instrument in the perpendicular direction in km/s (Default = 0.5).

    clip: float, optional
        The measurement width in vper and vpar (Default = 300.). 

    add_ring: np.array or list, optional
        The list must contain the following variables in the following order:
        q_r,p_r,wper_r,wpar_r,peak_r (Default = []). The variables q_r and p_r
        are the locations of a ring respectively in Vperp and Vpar, wper_r and wpar_r
        are the perpendicular and parallel thermal velocity widths, peak_r is the 
        maximum value of the added ring. The variable must be 5 elements long 
        or the code will not add a ring distribution.

    Returns
    ---------
    dis_vdf: dictionary
        Dictionary containing a discrete VDF function ['vdf'], the propogating direction grid ['pgrid'],
        the perpendicular to the propogating direction grid ['qgrid'], velocity vector in gse ['u_gse'],
        the normal magnetic field vector ['b_gse'] and a BiVariateSpline interpolation function ['vdf_func'].   
    """

    #Set up names for easy call
    u_gse = pls_par[:3]
    wper  = pls_par[3]
    wpar  = pls_par[4]
    n     = pls_par[5]


    #convert FWHM to sigma
    ftos = 1./(2.*np.sqrt(2.*np.log(2.)))
    
    #distribution of velocities in the parallel direction
    #p = np.arange(-wpar*clip,(wpar*clip)+pres,pres)
    #updated to fixed range 2018/10/12 J. Prchlik
    p = np.arange(-clip,clip+pres,pres)
    #distribution of velocities in the perpendicular direction
    #q = np.arange(0,(wper*clip)+qres,qres)
    #updated to fixed range 2018/10/12 J. Prchlik
    q = np.arange(0,clip+qres,qres)
    
    
    #created 2D grid of velocities in the X and y direction
    pgrid, qgrid = np.meshgrid(p,q)
    pgrid, qgrid = pgrid.T,qgrid.T
    
    #Get VDF peak from density
    a = n/(np.pi**(3./2.)*wpar*wper**2)
    
    #compute the raw vdf 
    rawvdf = a*np.exp(- (pgrid/(wpar))**2. - (qgrid/(wper))**2.)

    #Add ring to fit
    if len(add_ring) == 5:
        #parameters specificed in add_ring
        q_r,p_r,wper_r,wpar_r,peak_r = add_ring
        #Add a positive Gaussian Kernal to "Measured" VDF
        rawvdf += peak_r*np.exp(- ((pgrid-(p_r))/wpar_r)**2. - ((qgrid-(q_r))/wper_r)**2.)

    #create an interpolating function for the vdf
    f = RectBivariateSpline(p,q,rawvdf)
    
    #create dictionary
    dis_vdf = {'vdf':rawvdf,'pgrid':pgrid,'qgrid':qgrid,'u_gse':u_gse,'b_gse':mag_par,'vdf_func':f}
    return dis_vdf

def make_discrete_gennorm_vdf(pls_par,mag_par,pres=0.5,qres=0.5,clip=300.,add_ring=[]):
    """
    Returns Discrete Velocity distribution function given a set of input parameters.

    Parameters
    -----------
    pls_par: np.array
        A numpy array of plasma parameters in the following order: Vx,Vy,Vz,Wper,Wpar,The Peak (not Density),sper, and spar.
        That is the proton velocity in the solar wind in X GSE, Y GSE, and Z GSE in km/s,
        followed by the thermal width perpendicular and parallel to the magnetic field normal,
        and finally the proton number density in cm^{-3}. Respectively, sper and spar correspond to
        the s parameter of a generalized noramal distribution (Nadarajah 2005), where s=2 is a Gaussian.
        Wper and Wpar are assumed to be sigma values.
    
    mag_par: np.array
        A numpy array of the magnetic field normal in the following order : bx, by, and bz.
        The normal vectors should be defined in GSE coordinates.

    pres: float, optional
        The resolution of the instrument in the parallel direction in km/s (Default = 0.5).

    qres: float, optional
        The resolution of the instrument in the perpendicular direction in km/s (Default = 0.5).

    clip: float, optional
        The measurement width in vper and vpar (Default = 300.). 

    add_ring: np.array or list, optional
        The list must contain the following variables in the following order:
        q_r,p_r,wper_r,wpar_r,peak_r,sper_r,spar_r (Default = []). The variables q_r and p_r
        are the locations of a ring respectively in Vperp and Vpar, wper_r and wpar_r
        are the perpendicular and parallel thermal velocity widths, peak_r is the 
        maximum value of the added ring, sper_r and spar_r are the s parameter of a generalized
        normal distribution. The variable must be 7 elements long or the code will not add a ring distribution.

    Returns
    ---------
    dis_vdf: dictionary
        Dictionary containing a discrete VDF function ['vdf'], the propogating direction grid ['pgrid'],
        the perpendicular to the propogating direction grid ['qgrid'], velocity vector in gse ['u_gse'],
        the normal magnetic field vector ['b_gse'] and a BiVariateSpline interpolation function ['vdf_func'].   
    """

    #Set up names for easy call
    u_gse = pls_par[:3]
    wper  = pls_par[3]
    wpar  = pls_par[4]
    n     = pls_par[5]
    sper  = pls_par[6]
    spar  = pls_par[7]

    #Automatically add ring if plasma parameter is longer than 8 elements long 2019/03/25 J. Prchlik
    if len(pls_par) > 8:
        add_ring = pls_par[8:]



    #convert FWHM to sigma
    ftos = 1./(2.*np.sqrt(2.*np.log(2.)))
    
    #distribution of velocities in the parallel direction
    #p = np.arange(-wpar*clip,(wpar*clip)+pres,pres)
    #updated to fixed range 2018/10/12 J. Prchlik
    p = np.arange(-clip,clip+pres,pres)
    #distribution of velocities in the perpendicular direction
    #q = np.arange(0,(wper*clip)+qres,qres)
    #updated to fixed range 2018/10/12 J. Prchlik
    q = np.arange(0,clip+qres,qres)
    
    
    #created 2D grid of velocities in the X and y direction
    pgrid, qgrid = np.meshgrid(p,q)
    pgrid, qgrid = pgrid.T,qgrid.T
    
    #Get VDF constance
    #Added addition 2^{3/2)  based on definition 2018/10/05 J. Prchlik
    #Can no longer use traditional density in gennorm distribution 2019/02/08
    a = n#/(np.sqrt((np.pi)**3.)*(wpar*wper**2.)) # 1/cm^3 * s^3 /km^3 
    
    #compute the raw vdf (ftos comes from transformation from FWHM to 2Sigma)
    #rawvdf = a*np.exp(- (pgrid/wpar)**2. - (qgrid/wper)**2.)
    #Factor of 2 comes from definition of qgrid which is only positive and p=0 is the origin
    rawvdf = a*2.*gennorm.pdf(pgrid,spar,scale=np.sqrt(2.)*wpar)*gennorm.pdf(qgrid,sper,scale=np.sqrt(2.)*wper)

    #Add ring to fit
    if len(add_ring) == 7:
        #parameters specificed in add_ring
        q_r,p_r,wper_r,wpar_r,peak_r,sper_r,spar_r = add_ring
        #Add a positive Gaussian Kernal to "Measured" VDF
        #rawvdf += peak_r*np.exp(- ((pgrid-(p_r))/wpar_r)**2. - ((qgrid-(q_r))/wper_r)**2.)
        #Positive and negative pgrid comes from the p(arellel) value being a magnitude quantity 2019/02/08 J. Prchlik
        rawvdf += (peak_r*
                  gennorm.pdf( pgrid,spar_r,scale=(np.sqrt(2.)*wpar_r),loc=p_r)*
                  (gennorm.pdf(-qgrid,spar_r,scale=(np.sqrt(2.)*wper_r),loc=q_r)
                  +gennorm.pdf( qgrid,sper_r,scale=(np.sqrt(2.)*wper_r),loc=q_r)))

    #create an interpolating function for the vdf
    f = RectBivariateSpline(p,q,rawvdf)
    
    #create dictionary
    dis_vdf = {'vdf':rawvdf,'pgrid':pgrid,'qgrid':qgrid,'u_gse':u_gse,'b_gse':mag_par,'vdf_func':f}

    return dis_vdf

def ring_vdf(dis_vdf,vx,vy,vz,wper,wpar,den,q_r,p_r,wper_r,wpar_r,peak_r):
        #                   Vx,Vy,Vz,Wper,Wpar, Np
    pls_par = np.array([vx,vy,vz,wper, wpar, den]) 
    
    #Get static variables from cur_vdf to add to creation of new guess VDF
    vel_clip = cur_vdf['pgrid'].max()
    pres     = np.mean(np.diff(cur_vdf['pgrid'][:,0]))
    qres     = np.mean(np.diff(cur_vdf['qgrid'][0,:]))
    
    #Create new VDF guess based on input parameters
    dis_vdf = mdv.make_discrete_vdf(pls_par,cur_vdf['b_gse'],pres=pres,qres=qres,clip=vel_clip) 
    
    #Add ring to fit
    #Add a positive Gaussian Kernal to "Measured" VDF
    dis_vdf['vdf'] += peak_r*np.exp(- ((dis_vdf['pgrid']-(p_r))/wpar_r)**2. - ((dis_vdf['qgrid']-(q_r))/wper_r)**2.)
    #update the interpolator function
    dis_vdf['vdf_func'] =  RectBivariateSpline(dis_vdf['pgrid'][:,0],dis_vdf['qgrid'][0,:],dis_vdf['vdf'])
    
    return dis_vdf


def make_discrete_vdf_random(dis_vdf,normval,improved=False,sc_range=0.1,p_sig=10.,q_sig=10.,n_p_prob=[0.5,0.5],ip=0.,iq=0.):
    """
    Returns Discrete Velocity distribution function given a set of input parameters. With random variations 
    in the raw vdf

    Parameters
    -----------
    
    dis_vdf: dictionary
        Dictionary containing a discrete VDF function ['vdf'], the propogating direction grid ['pgrid'],
        the perpendicular to the propogating direction grid ['qgrid'], velocity vector in gse ['u_gse'],
        the normal magnetic field vector ['b_gse'] and a BiVariateSpline interpolation function ['vdf_func'].   
    normval: np.array
        A numpy array with the same shape as the p and q grids, which is the probability of selecting a point 
        any where on the grid. The grid should be updated in a way that guesses that improve the fit are favored 
        over guess that do not.
    improved: boolean, optional
        Whether the previous iteration improved the fit (Default = False)
    sc_range: float,optional
        Range to vary the input VDF as a fraction (Default = 0.1)
    p_sig: float,optional
        Sigma of the added gaussian in p space in km/s (Default = 10)
    q_sig: float,optional
        Sigma of the added gaussian in q space in km/s (Default = 10)
    n_p_prob: list or np.array, optional
        The probability of selecting a positive or negative gaussian. The first element is the probability
        of selecting a gaussian that removes from the vdf, while the second element is the probability of 
        selecting a gaussian that adds to the vdf. The total probability must sum to 1 (default = [0.5,0.5]).
    ip: float, optional
        Location of the last Gaussian kernel guess in the P coordinate (Default = 0.). If improved is true then
        this coordinate improved the fit and will be used for the next guess.
    iq: float, optional
        Location of the last Gaussian kernel guess in the Q coordinate (Default = 0.). If improved is true then
        this coordinate improved the fit and will be used for the next guess.

    Returns
    ---------
    ran_vdf: dictionary
        Dictionary containing a discrete VDF function ['vdf'], the propogating direction grid ['pgrid'],
        the perpendicular to the propogating direction grid ['qgrid'], velocity vector in gse ['u_gse'],
        the normal magnetic field vector ['b_gse'] and a BiVariateSpline interpolation function ['vdf_func'].   
    p_grab:  float
        The center of the new Gaussian in the p coordinate
    q_grab:  float
        The center of the new Gaussian in the q coordinate
    a_scale: float
        The scale factor of the Gaussian kernel applied to 2D velocity distribution that is either +/-1
    """

    #distribution of velocities in the parallel direction
    p = dis_vdf['pgrid'][:,0]
    #distribution of velocities in the perpendicular direction
    q = dis_vdf['qgrid'][0,:]
    

    #copy previous variables
    pgrid = dis_vdf['pgrid']
    qgrid = dis_vdf['qgrid']
    u_gse = dis_vdf['u_gse']
    mag_par = dis_vdf['b_gse']

     
    local_state = np.random.RandomState()
    
    #grab the raw vdf
    rawvdf = dis_vdf['vdf']

    #normalized probabilities to vary
    #normval = np.log10(rawvdf)/np.sum(np.log10(rawvdf))
   

    #grab some value on the q,p grid use the input VDF to inform the prior
    #normalizing creates preference to cut down middle of velocity distribution 2018/08/23 J. Prchlik
    #Try weighting by the predicted VDF 2018/10/03 J. Prchlik
    #Grad point in the same area if previous gaussian improved fit
    #Stop guessing around a positive point instead updating a ring structure around a point 2018/11/16 J. Prchlik
    ####if improved:
    ####    #Added reduced Gaussian width 2018/10/24 J. Prchlik
    ####    p_grab = float(local_state.normal(loc=ip,scale=p_sig/2.,size=1))#,p=normval.ravel()))
    ####    q_grab = float(local_state.normal(loc=iq,scale=q_sig/2.,size=1))#,p=normval.ravel()))
    #####Do no guess in same area if adding a Gaussian did not improve the fit 2018/10/18 J. Prchlik
    ####else:
    p_grab = float(local_state.choice(pgrid.ravel(),size=1,p=normval.ravel()))
    q_grab = float(local_state.choice(qgrid.ravel(),size=1,p=normval.ravel()))

    
    #try either adding or subtracting
    a_scale = float(local_state.choice([-1.,1],size=1,p=n_p_prob))


    #calculate amplitude of vdf at p_grab, q_grab
    a = float(dis_vdf['vdf_func'](p_grab,q_grab,grid=False))

   
    #vary the amplitude of the guassian by a small amount
    low_off = 1.-sc_range
    hgh_off = 1.+sc_range

    #adjust the height of the peak
    var = float(local_state.uniform(low=low_off,high=hgh_off,size=1))

    #add varience to a_scale
    a_scale *= var

    #Correct for the cylinderical coordinates in q system (refect a negative Gaussian into the postive axis) 2018/11/16 J, Prchlik
    p_g = np.exp(- ((pgrid-p_grab)/p_sig)**2. - ((qgrid-q_grab)/q_sig)**2.)
    n_g = np.exp(- ((pgrid-p_grab)/p_sig)**2. - ((qgrid-(-q_grab))/q_sig)**2.)

    #add variation guassian to rawvdf
    ranvdf = a_scale*a*(p_g+n_g)+rawvdf

    #replace with original values if something is less than 0
    ranvdf[ranvdf < 0] = rawvdf[ranvdf < 0]

    #create an interpolating function for the vdf
    f = RectBivariateSpline(p,q,ranvdf)


    #####
    #Testing kernel addition 2018/09/14 J. Prchlik
    ####
    #ranvdf = a_scale*a*np.exp(- ((pgrid-p_grab)/p_sig)**2. - ((qgrid-q_grab)/q_sig)**2.)
    
    #create dictionary
    ran_vdf = {'vdf':ranvdf,'pgrid':pgrid,'qgrid':qgrid,'u_gse':u_gse,'b_gse':mag_par,'vdf_func':f}
    return ran_vdf,p_grab,q_grab,a_scale

def convert_fc_gse(fc_cor,phi_ang,theta_ang):
    """
    Convert GSE coordinates to faraday cup coordinates
    The - sign in phi_ang comes from the definition used in Arfken, Weber, and Harris  compared to what
    I meant when I set up the problem

    Parameters
    ----------
    fc_cor: float
        3D Measurements made in the FC rest frame  
    phi_ang: float or np.array
        Phi angle between GSE and FC (radians)
    theta_ang: float or np.array
        Phi angle between GSE and FC (radians)
 
    Returns
    ---------
    xyz: np.array
        Components of input in GSE
    """


    ##########invert rotation matrix and apply to solution from gaussian fit  
    #######Brought back 2018/09/12 J. Prchlik
    #simplified with two rotation matrix 2018/10/04 J. Prchlik
    ######get rotation matrix
    rot_mat = rotation_matrix(-phi_ang,theta_ang,psi_ang=0.)
    rot_cor = rotation_matrix(-np.pi/2.,-np.pi/2.,psi_ang=0.)
    #removed Transpose 2018/10/03 J. Prchlik because of changed rotation matrix definition
    #convert back to Right handed definition in GSE coordinates
    x_gse,y_gse,z_gse = rot_mat.dot(rot_cor).dot(fc_cor)


    return np.array([x_gse,y_gse,z_gse])


def rotation_matrix(phi_ang,theta_ang,psi_ang=0.):
    """   
    For a single phi, theta, and phi compute the Euler rotation matrix
   
    Example
    --------
    rot_mat = rotation_matrix(phi_ang,theta_ang,psi_ang=0.)

    Parameters
    -----------
    phi_ang: float
        The phi angle (yaw) between two coordinate systems in radians
    theta_ang: float
        The theta angle (pitch) between two coordinate systems in radians
    psi_ang: float,optional 
        The psi angle (roll) between two coordinate systems in radians (Default = 0.)

    Returns
    ---------
    rot_mat: np.array
        A 3x3 rotation matrix from the input phi, theta, and psi angles
    """
    #get euler angles
    #updating euler angles to a more generic definition 2018/10/04 J. Prchlik
    a11 = np.cos(psi_ang)*np.cos(theta_ang)*np.cos(phi_ang)-np.sin(psi_ang)*np.sin(phi_ang)
    a12 = np.cos(psi_ang)*np.cos(theta_ang)*np.sin(phi_ang)+np.sin(psi_ang)*np.cos(phi_ang)
    a13 = -np.cos(psi_ang)*np.sin(theta_ang)
    a21 = -np.sin(psi_ang)*np.cos(theta_ang)*np.cos(phi_ang)-np.cos(psi_ang)*np.sin(phi_ang)
    a22 = -np.sin(psi_ang)*np.cos(theta_ang)*np.sin(phi_ang)+np.cos(psi_ang)*np.cos(phi_ang)
    a23 = np.sin(psi_ang)*np.sin(theta_ang)
    a31 = np.sin(theta_ang)*np.cos(phi_ang)
    a32 = np.sin(theta_ang)*np.sin(phi_ang)
    a33 = np.cos(theta_ang)


    #create rotation matrix
    rot_mat = np.array([[a11,a12,a13],
                         [a21,a22,a23],
                         [a31,a32,a33]],dtype=np.double)

    return rot_mat

#### Redundent 2018/10/12
####jdef rotation_matrix(phi_ang,theta_ang,psi_ang=0.):
####j    #get the stardard euler angles and rotation matrix
####j    rot_mat = euler_angles(phi_ang,theta_ang,psi_ang=psi_ang)
####j
####j    
####j    return rot_mat
####j

def find_best_vths(wa,we,pls_par,mag_par,rea_cur,inpt_x,pres=1.00,qres=1.00,warange=3.,werange=3.,wasamp=10,wesamp=4):
    """
    Varies the thermal width on one faraday cup to get the best fit initial guess solution. 
    Based on fit values of components
   
    Example
    ----------
    wa, we =  mdv.find_best_vths(wa,we,pls_par_bad,rea_cur,inpt_x)

    Parameters
    ----------
    wa: float
        Initial guess of Wpar in km/s
    we: float 
        Inital guess of Wper in km/s
    pls_par: np.array
        Fit plasma parameters using SVD method in this module
    mag_par: np.array
        Observed magnetic field normal vector, which Wpar and Wper are parallel and perpendicular to, respectively
    rea_cur: np.array
        A 1D array of the current measured in the FC
    inpt_x: np.array
        A 2D array of FC cup measurements (Created x_meas array in this library).
    warange: float, optional
        Multiplied by wa to give the range of values to consider (Default = 3)
    werange: float, optional
        Multiplied by we to give the range of values to consider (Default = 3)
    wasamp: int,optional
        Number of gridspace to use in X^2 fitting in parallel direction (Default = 10)
    wesamp: int,optional
        Number of gridspace to use in X^2 fitting in perpendicular direction (Default = 4)


    Return
    --------
    out_a: float
        The updated Wpar parameter
    out_e: float
        The updated Wper parameter

    """
    import itertools as it

    #Run through X^2 space to find the best widths
    was = np.linspace(warange,warange*wa,wasamp)
    wes = np.linspace(werange,werange*we,wesamp)

    #Append the original fit values 
    was = np.concatenate((was,np.array([wa])))
    wes = np.concatenate((wes,np.array([we])))

    #set the output values
    out_a = wa
    out_e = we

    #set the init X^2 value
    chi2 = 1.e30

    #get all combinations of wa's and we's
    combs = it.product(was,wes)
    for a,e in combs:
        #Make a copy of the input plasma array
        tmp_pls_par = pls_par.copy() 
        #Update wit the grid of thermal speed parameters
        tmp_pls_par[3] = e
        tmp_pls_par[4] = a
        #Get the 2D velocity distribution in X^2 space
        vdf_f_chi = make_discrete_vdf(tmp_pls_par,mag_par,pres=1.00,qres=1.00,clip=5.)
        
        #integrate over given X (spacecraft) values
        test_cur = arb_p_response(inpt_x,vdf_f_chi,30)

        #compute Chi2 stat
        chi2_tmp = np.sum(((test_cur-rea_cur)/rea_cur)**2)

        #update value if new parameter is better
        if chi2_tmp < chi2: 
            chi2 = chi2_tmp
            out_a = a
            out_e = e

    #return best values over loop
    return out_a,out_e

       
     

   


def convert_gse_fc(gse_cor,phi_ang,theta_ang):
    """
    Convert GSE coordinates to faraday cup coordinates. Faraday cup coordiantes have positive Z pointing toward the receiving cup normal
    and positive X to the right in a right handed coordinate system.
    The - sign in phi_ang comes from the definition used in Arfken, Weber, and Harris  compared to what
    I meant when I set up the problem

    Parameters
    ----------
    gse_cor: np.array
        X,Y, and X GSE coordinates to convert to  faraday cup coordiantes
    
    phi_ang: float or np.array
        Phi angle between GSE and FC (radians)

    theta_ang: float or np.array
        Phi angle between GSE and FC (radians)
    
    Returns
    ---------
    p_grid: np.array
        Array of coordinates converted into from X, Y, Z GSE to faraday cup X, Y, Z
    """


    #switched to simplier euler angle transformation 2018/10/03 J. Prchlik
    rot_mat = rotation_matrix(-phi_ang,theta_ang)
    rot_cor = rotation_matrix(-np.pi/2.,-np.pi/2.)
    p_grid  = rot_mat.dot(rot_cor).T.dot(gse_cor)

    

    return p_grid

def make_fc_meas(dis_vdf,fc_spd=np.arange(300,600,15),fc_phi=-15.,fc_theta=-15):
    """
    Creates measurement parameters for a given faraday cup

    Parameters
    ----------
    dis_vdf: dictionary
        A dictionary returned from make_discrete_vdf
    
    fc_spd: np.array, optional
        A 1D numpy array containing speed bins for the faraday cup 
        (Default = np.arange(300,600,15)). 

    fc_phi: float, optional
        The Phi angle between the faraday cup center and GSE (Default = -15)

    fc_theta: float, optional
        The Phi angle between the faraday cup center and GSE (Default = 15)

    Returns
    --------
    x_meas: np.array
               x[0,:]          v_window  [km/s] 
               x[1,:]          v_delta   [km/s] 
               x[2,:]          phi_ang (yaw) [rad] 
               x[3,:]          theta_ang (pitch) [rad] 
               x[4,:]      b in FC "x" pointing right
               x[5,:]      b in FC "y"
               x[6,:]      b in FC "z" normal to cup
    """

    #Create array to populate with measurement geometry and range
    x_meas = np.zeros((7,fc_spd.shape[0]),dtype=np.double)

    #get the gse values of the magnetic field
    b_gse = dis_vdf['b_gse']

    #Populate measesurement values
    x_meas[0,:] = fc_spd #speed bins
    x_meas[1,:] = np.concatenate([np.diff(fc_spd),[fc_spd[-1]-fc_spd[-2]]]) #speed bin sizes
    x_meas[2,:] = np.radians(fc_phi+np.zeros(fc_spd.shape[0]))
    x_meas[3,:] = np.radians(fc_theta+np.zeros(fc_spd.shape[0]))


    #get transformed coordinates from GSE to FC
    #only need the first element because the angle are consistent throughout
    #Will now fail if theta and phi are arrays base on updates on 2018/10/04 J. Prchlik
    out_xyz = convert_gse_fc(b_gse,x_meas[2,0],x_meas[3,0])

    #print(out_xyz)
    #print(out_xyz.shape)

    #populate out_xyz into x_meas
    #It is just a repeat value so just fill entire array with value
    #2018/10/04 J. Prchlik
    x_meas[4,:] = out_xyz[0]
    x_meas[5,:] = out_xyz[1]
    x_meas[6,:] = out_xyz[2]

    return x_meas

def plot_vdf(dis_vdf):
    """
    Plots VDF in dictionary dis_vdf

    Parameters
    ----------
    dis_vdf: dictionary
         A dictionary returned from make_discrete_vdf

    Returns
    ---------
        fig, ax objects
    """
    import matplotlib.pyplot as plt

    #Added static Contour levels 2018/10/12 J. Prchlik
    n_levels = 5
    #update contour levels to be n$\sigma$ from peak 2018/11/16 J. Prchlik
    contour_levels = np.exp(-np.arange(n_levels,0,-1))
    #contour_levels = np.arange(0,2*n_levels+1,2)-4-n_levels*2
    plot_v = dis_vdf['vdf']/dis_vdf['vdf'].max()

    #get values above counter levels
    good_vals = np.where(plot_v > min(contour_levels))

    #min, max Vperp
    q_max = np.max(np.abs(dis_vdf['qgrid'][good_vals]))
    #max Vpar
    p_max = np.max(np.abs(dis_vdf['pgrid'][good_vals]))

    fig, ax = plt.subplots(figsize=(8,6))

    plotc = ax.pcolormesh(dis_vdf['pgrid'],dis_vdf['qgrid'],plot_v,vmin=min(contour_levels),vmax=1)
    ax.contour(dis_vdf['pgrid'],dis_vdf['qgrid'],plot_v, contour_levels,colors='black',linestyles='dashed',linewidths=3 )
    cbar = fig.colorbar(plotc)
    cbar.set_label('Normalized [Fraction of Max.]',fontsize=18)

    #square everything up
    p_max = np.sqrt(q_max**2+p_max**2)
    q_max = p_max
    ax.set_xlim([-p_max,p_max])
    ax.set_ylim([0,q_max])

    #Add labels to plots
    ax.set_xlabel(r'V$_\parallel$ [km/s]',fontsize=22)
    ax.set_ylabel(r'V$_\perp$ [km/s]',fontsize=22)

    return fig,ax

def plot_pq_lim_fc(x_meas,dis_vdf,fov_ang=45.):
    """
    Plots the limits of the FC in pq space

    Parameters
    ----------
    x_meas: np.array
               x[0,:]          v_window  [km/s] 
               x[1,:]          v_delta   [km/s] 
               x[2,:]          phi_ang (yaw) [rad] 
               x[3,:]          theta_ang (pitch) [rad] 
               x[4,:]      b in FC "x" pointing right
               x[5,:]      b in FC "y"
               x[6,:]      b in FC "z" normal to cup
    dis_vdf: dictionary
         A dictionary returned from make_discrete_vdf
    fov_ang: float, optional
        The limiting FoV angle for the FC (Default = 45)

    Returns
    ---------
        fig, ax objects
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8,6))
    #Added static Contour levels 2018/10/12 J. Prchlik
    n_levels = 10
    contour_levels = np.arange(0,2*n_levels+1,2)-4-n_levels*2


    plotc = ax.pcolormesh(dis_vdf['pgrid'],dis_vdf['qgrid'],np.log10(dis_vdf['vdf']),vmin=-19,vmax=-5)
    ax.contour(dis_vdf['pgrid'],dis_vdf['qgrid'],np.log10(dis_vdf['vdf']), contour_levels,colors='black',linestyles='dashed',linewidths=3 )
    cbar = fig.colorbar(plotc)
    cbar.set_label('Normalized Dist. [s$^{3}$cm$^{-3}$km$^{-3}$]',fontsize=18)

    ax.set_xlabel(r'V$_\parallel$ [km/s]',fontsize=22)
    ax.set_ylabel(r'V$_\perp$ [km/s]',fontsize=22)

    #set state of the faraday cup
    fc_state_vlo = x_meas[0,:]-.5*x_meas[1,:] 
    fc_state_vhi = x_meas[0,:]+.5*x_meas[1,:] 
    fc_state_phi = x_meas[2,:]
    fc_state_tht = x_meas[3,:]

    #"Measured" Bx,By, and Bz values in FC
    hold_bfc = x_meas[4:7,0]#convert_gse_fc(dis_vdf['b_gse'],phi_ang,theta_ang)

    #"Measured" Vx,Vy, and Vz values in FC
    hold_ufc = convert_gse_fc(dis_vdf['u_gse'],fc_state_phi[0],fc_state_tht[0])

    #create rotation matrix between FC and magnetic field normal
    #So that the magnetic field normal in FC reference frame maps
    #to (p',q',r') = (1,0,0) in the magnetic field frame
    rot_mat = arb_rotation_matrix(hold_bfc,[1.,0.,0.])

    #for i in range( x_meas[0,:].size):
    for i in (0, x_meas[0,:].size-1):
        
        #get parameters for each faraday cup value
        fc_vlo    = fc_state_vlo[i]
        fc_vhi    = fc_state_vhi[i]
        phi_ang   = fc_state_phi[i]
        theta_ang = fc_state_tht[i]

        #radius in velocity space of the fc measurement
        fc_rad = fc_vhi

        #Get sample of Xfc,Yfc coordinats
        phis = np.linspace(0.,2.*np.pi,200)
        fc_x, fc_y = fc_vhi*np.cos(phis),fc_vhi*np.sin(phis)
        fc_z_lo = 1.*fc_vlo+np.zeros(fc_x.size)
        fc_z_hi = 1.*fc_vhi+np.zeros(fc_x.size)

        ####print(fc_vhi)
        ####print(max(fc_x),min(fc_x))
        ####print(max(fc_x),min(fc_y))
        ####print(dis_vdf['u_gse'])
        ####print(np.degrees([phi_ang,theta_ang]))
        ####print(hold_ufc)
        #create Nx3 array for sampling
        #fc_x = np.concatenate([fc_x,fc_x])
        #fc_y = np.concatenate([fc_y,fc_y])
        #fc_z = np.concatenate([fc_z_lo,fc_z_hi])
        fc_xyz_hi = np.vstack([fc_x-hold_ufc[0],fc_y-hold_ufc[1],fc_z_hi+hold_ufc[2]])
        fc_xyz_lo = np.vstack([fc_x-hold_ufc[0],fc_y-hold_ufc[1],fc_z_lo+hold_ufc[2]])

        #convert faraday cup coordinates into p',q',r' coordinates
        fc_pqr_hi = rot_mat.dot(fc_xyz_hi)
        fc_pqr_lo = rot_mat.dot(fc_xyz_lo)
      
        #convert from p',q',r' to p,q
        fc_p_hi = fc_pqr_hi[0]
        fc_q_hi = np.sqrt(fc_pqr_hi[2]**2+fc_pqr_hi[1]**2)
        #convert from p',q',r' to p,q
        fc_p_lo = fc_pqr_lo[0]
        fc_q_lo = np.sqrt(fc_pqr_lo[2]**2+fc_pqr_lo[1]**2)

        #test dot product
        dot_test_p_hi = hold_bfc.dot(fc_xyz_hi)
        dot_test_p_lo = hold_bfc.dot(fc_xyz_lo)
        dot_test_q_hi = np.sqrt(np.sum(fc_xyz_hi**2,axis=0)-dot_test_p_hi**2)
        dot_test_q_lo = np.sqrt(np.sum(fc_xyz_lo**2,axis=0)-dot_test_p_lo**2)

        ####print('#########################')
        ####print('Pvals')
        ####print(max(fc_p_hi),min(fc_p_hi))
        ####print(max(fc_p_lo),min(fc_p_lo))
        ####print(max(dot_test_p_hi),min(dot_test_p_hi))
        ####print(max(dot_test_p_lo),min(dot_test_p_lo))
        ####print('#########################')
        ####print('#########################')
        ####print('Qvals')
        ####print(max(fc_q_hi),min(fc_q_hi))
        ####print(max(fc_q_lo),min(fc_q_lo))
        ####print(max(dot_test_q_hi),min(dot_test_q_hi))
        ####print(max(dot_test_q_lo),min(dot_test_q_lo))
        ####print('#########################')
        #plot the new p and q limits
        #plot the new p and q limits
        if i == 0:
            #plot lower limit
            ax.plot(fc_p_lo,fc_q_lo,'-r')
            #store first p,q values of lower limit
            fp_p,fp_q = fc_p_lo[0],fc_q_lo[0]
            #store last p,q values of lower limit
            lp_p,lp_q = fc_p_lo[-1],fc_q_lo[-1]
        else:
            #plot upper limit
            ax.plot(fc_p_hi,fc_q_hi,'-r')

    #complete pbox of FC measurements
    #ax.plot([fp_p,fc_p_hi[0]],[fp_q,fc_q_hi[0]],'-r')
    #ax.plot([lp_p,fc_p_hi[-1]],[lp_q,fc_q_hi[-1]],'-r')


    return fig,ax


def arb_rotation_matrix(a,b):
    """
    Creates a rotation matrix between two unit vectors a and b.
    U.dot(a) = b
 
    Parameters
    ------------
    a,b: np.array
        Two 3 element numpy array unit vectors

    Returns
    -----------
    rot_mat: np.array
        3x3 rotation matrix between unit vectors
    
    """
   
    v = np.cross(a,b)
    s = np.linalg.norm(v)
    c = np.dot(a,b)
 
    I = np.identity(3)

    #skew-symmetric cross product of matrix v
    vx = np.array([[0,-v[2],v[1]],
                   [v[2],0,-v[0]],
                   [-v[1],v[0],0]])

    rot_mat = I+vx+vx.dot(vx)/(1+c)
    return rot_mat
                  

def sample_function(z_min,z_max,z_peak,max_s=45,min_s=30,wid_s=100):
    """
    Parameters
    ----------
    z_min: float
        The minimum solar wind speed accepted by the FC. 
    z_max: float
        The maximum solar wind speed accepted by the FC. 
    z_peak: float
        The peak of the solar wind in FC coordinates.
    max_s: int, optional
        Maximum number of samples near peak (Default = 45)
    min_s: int, optional
        Minimum number of samples near peak (Default = 30)
    wid_s: int, optional
        Width of Gaussian near peak in same units as z_min, z_max, and z_peak (Default = 10)
    Returns
    -------
    samp: int
        Number of samples to run in current bin.
    """
    z = (z_min+(z_max-z_min)/2.)

    #Gaussian peak
    A = (max_s-min_s)
  

    #calculate the number of samples 
    samp = A*np.exp(-((z-z_peak)**2/(2.*wid_s**2)))+min_s
    #convert to an integer
    samp = int(round(samp))

    return samp


def arb_p_response_dyn_samp(x_meas,dis_vdf,z_peak,
                            samp_func = sample_function,sc='wind'):
    """
    Parameters
    -----------
    x_meas: np.array
         expect x_meas array in the same format as the wind code
         x[0,:]          v_window  [km/s] 
         x[1,:]          v_delta   [km/s] 
         x[2,:]          phi_ang [rad] 
         x[3,:]          theta_ang [rad] 
         x[4,:]      b in FC "x"
         x[5,:]      b in FC "y"
         x[6,:]      b in FC "z" normal to cup
        
    dis_vdf: dictionary
        A dictionary returned from make_discrete_vdf
    z_peak: float
        The location for peak sampling 
    samp_func: function, optional
        Dynamic function used to determine the sampling for a given set up x,y,z in FC coordinates (Default = sample_function).
    sc: string, optional
        Spacecraft effective area to use (Default = 'wind')

    Returns
    --------
    dis_cur: np.array
        Discrete values of the measured current in the FC at x[0,:] velocity windows
    
    """

    #number of spectral observations in cup
    nmeas = x_meas.shape[1]

    #simulated measured current
    f_p_curr = np.zeros(nmeas)

    #set state of the faraday cup
    fc_state_vlo = x_meas[0,:]-.5*x_meas[1,:] 
    fc_state_vhi = x_meas[0,:]+.5*x_meas[1,:] 
    fc_state_phi = x_meas[2,:]
    fc_state_tht = x_meas[3,:]
 
    

    #get faraday cup measurement at each FC measurement velocity
    dis_cur = []
    for i in range( x_meas[0,:].size):
        
        #get parameters for each faraday cup value
        fc_vlo    = fc_state_vlo[i]
        fc_vhi    = fc_state_vhi[i]
        phi_ang   = fc_state_phi[i]
        theta_ang = fc_state_tht[i]
        
        #get sampling for current bin
        samp = samp_func(fc_vlo,fc_vhi,z_peak)
        #print(samp,fc_vlo,fc_vhi,z_peak,phi_ang,theta_ang)

        inp = np.array([fc_vlo,fc_vhi,phi_ang,theta_ang])

        out = fc_meas(dis_vdf,inp,samp=samp,sc=sc)
        #print(out)
        dis_cur.append(out)

    dis_cur = np.array(dis_cur)

    return dis_cur


def arb_p_response(x_meas,dis_vdf,samp,sc='wind'):
    """
    Parameters
    -----------
    x_meas: np.array
         expect x_meas array in the same format as the Wind code
         x[0,:]          v_window  [km/s] 
         x[1,:]          v_delta   [km/s] 
         x[2,:]          phi_ang [rad] 
         x[3,:]          theta_ang [rad] 
         x[4,:]      b in FC "x"
         x[5,:]      b in FC "y"
         x[6,:]      b in FC "z" normal to cup
        
    dis_vdf: dictionary
         A dictionary returned from make_discrete_vdf

    samp: int
        Velocity bins to use when doing the Mid-Point Integration. 10 km/s Seems to be a good mixture
        of rapid integration and high precision for Wind observations. 

    sc: string, optional
        Spacecraft effective area to use (Default = 'wind')

    Returns
    --------
    dis_cur: np.array
        Discrete values of the measured current in the FC at x[0,:] velocity windows
    
    """

    #number of spectral observations in cup
    nmeas = x_meas.shape[1]

    #simulated measured current
    f_p_curr = np.zeros(nmeas)

    #set state of the faraday cup
    fc_state_vlo = x_meas[0,:]-.5*x_meas[1,:] 
    fc_state_vhi = x_meas[0,:]+.5*x_meas[1,:] 
    fc_state_phi = x_meas[2,:]
    fc_state_tht = x_meas[3,:]
 
    

    #get faraday cup measurement at each FC measurement velocity
    dis_cur = []
    for i in range( x_meas[0,:].size):
        
        #get parameters for each faraday cup value
        fc_vlo    = fc_state_vlo[i]
        fc_vhi    = fc_state_vhi[i]
        phi_ang   = fc_state_phi[i]
        theta_ang = fc_state_tht[i]

        inp = np.array([fc_vlo,fc_vhi,phi_ang,theta_ang])

        #Do the integration for a given cup velocity range
        out = fc_meas(dis_vdf,inp,samp=samp,sc=sc)
        #print(out)
        dis_cur.append(out)

    dis_cur = np.array(dis_cur)

    return dis_cur

#
def fc_meas(vdf,fc,fov_ang=45.,sc ='wind',samp=10):
    """
    Get the spacecraft measurement of the VDF

    Parameters
    ----------
    vdf: dictionary
        A dictionary returned from make_discrete_vdf.
   
    fc: np.array
        A numpy containing properties of the faraday cup in lo, hi, phi, theta

    sc: string, optional
        Spacecraft effective area to use (Default = 'wind')
  
    samp: int, optional
       Sampling frequency in km/s

    Returns
    ---------
    meas: float or np.array
        Measured current in FC
    """


    #break up faraday cup parameters
    fc_vlo    = fc[0]
    fc_vhi    = fc[1]
    phi_ang   = fc[2]
    theta_ang = fc[3]

    #"Measured" Vx,Vy, and Vz values in FC
    hold_ufc = convert_gse_fc(vdf['u_gse'],phi_ang,theta_ang)
    #"Measured" Bx,By, and Bz values in FC
    hold_bfc = convert_gse_fc(vdf['b_gse'],phi_ang,theta_ang)
    #"Measured" VDF
    hold_vdf = vdf['vdf']
 
    #interpolating function 
    hold_ifunc = vdf['vdf_func']


    #get the angle GSE and magnetic field
    phi_mag_fc = np.arctan2(hold_bfc[1],hold_bfc[0])
    the_mag_fc = np.arcsin(hold_bfc[2])


    #create vx limits
    vx_lim_min = lambda vz: -vz*np.tan(np.radians(fov_ang))
    vx_lim_max = lambda vz: vz*np.tan(np.radians(fov_ang))
    #create vy limits
    vy_lim_min = lambda vz,vx: -np.sqrt((vz*np.tan(np.radians(fov_ang)))**2- vx**2)
    vy_lim_max = lambda vz,vx: np.sqrt((vz*np.tan(np.radians(fov_ang)))**2- vx**2)



    #Get angle between magnetic field normal and the FC
    ###bn_phi,bn_theta = np.arctan2(hold_bfc[1],hold_bfc[0]),np.arcsin(hold_bfc[2])

    ####create rotation matrix from fc normal to b normal
    ####need - sign due to Arfken, Webber, and Harris definition
    ###fc_bn_rot = rotation_matrix(-bn_phi,bn_theta)
    ####flip so z FC is the normal
    ###fc_cr_rot = rotation_matrix(-np.pi/2.,np.pi/2.)

    #Switch to more efficent matrix rotation
    #create rotation matrix between FC and magnetic field normal
    #So that the magnetic field normal in FC reference frame maps
    #to (p',q',r') = (1,0,0) in the magnetic field frame
    rot_mat = arb_rotation_matrix(hold_bfc,[1.,0.,0.])
    #Use set velocity resolution to find change values in GSE coordinates while trying to sample 
    #consistent values in p',q',r' coordinates
    dx,dy,dz = np.abs(rot_mat.dot(np.zeros(3)+samp))


    #get maximum number of samples for array creation
    z_samp = (fc_vhi-fc_vhi)/dz
    x_samp = (vx_lim_max(fc_vhi)-vx_lim_min(fc_vhi))/dx
    y_samp = (vy_lim_max(fc_vhi,0)-vy_lim_min(fc_vhi,0))/dy

    #max number of samples at 1000.
    z_samp,x_samp,y_samp = 400.*np.array([z_samp,x_samp,y_samp])/(z_samp+x_samp+y_samp)

    #make sure at least 1 sample in a given direction
    if z_samp < 2:
        z_samp = 2
    if x_samp < 2:
        x_samp = 2
    if y_samp < 2:
        y_samp = 2


    #create function with input parameters for int_3d
    #int_3d_inp = partial(int_3d,spacecraft=sc,ufc=hold_ufc,bfc=hold_bfc,qgrid=hold_qgrid,pgrid=hold_pgrid,vdf=hold_vdf)
    args = (sc,hold_ufc,hold_bfc,hold_ifunc)
        
    #Updated with constant p',q', and r' coordinates
    #meas = mci.mp_trip_cython(int_3d, fc_vlo, fc_vhi, vx_lim_min, vx_lim_max, vy_lim_min, vy_lim_max,args=args,samp=samp)
    meas = mci.mp_trip_cython_pqr(int_3d, fc_vlo, fc_vhi, vx_lim_min, vx_lim_max, vy_lim_min, vy_lim_max,
                                  int(z_samp),int(x_samp),int(y_samp),args=args)

    return meas


#Note unlike IDL PYTHON expects VZ to be last not first
def int_3d(vy,vx,vz,spacecraft='wind',ufc=[1],bfc=[1],ifunc=lambda p,q: p*q): 
    """
    3D function to integrate. Vz is defined to be normal to the cup sensor

    Parameters
    ----------
    vz: np.array
        The velocity of the solar wind normal FC in km/s
    vx: np.array
        The velocity of the solar wind in the x direction with respect to the FC in km/s
    vy: np.array                                                                 
        The velocity of the solar wind in the y direction with respect to the FC in km/s
    bfc: np.array
        An array of magnetic field vectors in the faraday cup corrdinates  [Bx,By,Bz]
    ufc: np.array                                                                   
        An array of plasma velocity vectors in the faraday cup corrdinates [Vx,Vy,Vz]
    hold_ifunc: ND interpolation function
        Function of a RectBivariateSpline (https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RectBivariateSpline.html).
        Doing the interpolation this way is much faster than reinterpolating from a grid everytime as early version of the code required.
    """
    e =  1.60217646e-19   # coulombs

##3    eff_area_inp = partial(eff_area,spacecraft=spacecraft)
##3    vdf_inp = partial(vdf_calc,hold_bfc=bfc,hold_ufc=ufc,
##3                      hold_qgrid=qgrid,hold_pgrid=pgrid,
##3                      hold_vdf=vdf)
##3

    #Calculate effective area
    test_area = eff_area(vx,vy,vz,spacecraft=spacecraft)
    #if ((test_area < 1.e-16) | (not np.isfinite(test_area))):
    #    return 0


    #Get observed VDF
    test_vdf  =  vdf_calc(vx,vy,-vz,hold_bfc=bfc,hold_ufc=ufc,
                      hold_ifunc=ifunc)
    #if ((test_vdf < 1.e-16) | (not np.isfinite(test_vdf))):
    #    return 0
#####    print('TEST INT')
#####    print(min(test_vdf))
#####    print(min(test_area))
#####    print(min(vz),max(vz))
#####    print(min(vx),max(vx))
#####    print(min(vy),max(vy))
#####    print(e)
#####    print('TEST INT')
#####
    #print('############')
    #print('EFF AREA')
    #print(test_area)
    #print("VDF calc")
    #print(test_vdf)
    #val = e*(vz)*eff_area_inp(vx,vy,vz)*vdf_inp(-vx,vy,vz)
    #print(vz)
    val =  e*(vz)*test_area*test_vdf #eff_area_inp(vz,vx,vy)*vdf_inp(vz,vx,vy)

    #remove nan values
    #val = val[np.isfinite(val)]
    #print(val)
    return val


def eff_area(vx,vy,vz,spacecraft='wind'):
    """
    Calculates effective area for a given set of velocities

    Parameters
    ----------
    vx: np.array
        The velocity of the solar wind in the x direction with respect to the FC in km/s
    vy: np.array                                                                 
        The velocity of the solar wind in the y direction with respect to the FC in km/s
    vz: np.array
        The velocity of the solar wind normal FC in km/s
    
    Returns
    -------
    calc_eff_area: np.array
        The effective area for each set of velocites
    """

    #get angle onto the cup
    alpha = np.degrees(np.arctan2(np.sqrt(vy**2 + vx**2),vz))

    #Get effective area for give spacecraft
    eff_area = return_space_craft_ef(spacecraft)
    
    #alpha as an index for wind
    i_alpha = (alpha*10.) #.astype('int')

    #get interpolated effective area and fill out of range values with 0
    calc_eff_area = np.interp(i_alpha,np.arange(eff_area.size),eff_area,left=0,right=0)

    #Get compute effective area at closest value 
    return calc_eff_area



# for a VDF that is defined on a grid, get an interpolate
# at any desired location in phase space
#
def vdf_calc(vx,vy,vz,hold_bfc=[1,1,1],hold_ufc=[1,1,1],hold_ifunc=lambda p,q: p*q):
    """
    Calculates measured VDF

    Parameters
    ----------
    vx: np.array
        The velocity of the solar wind in the x direction with respect to the FC in km/s
    vy: np.array                                                                 
        The velocity of the solar wind in the y direction with respect to the FC in km/s
    vz: np.array
        The velocity of the solar wind normal FC in km/s
    hold_bfc: np.array
        An array of magnetic field vectors in the faraday cup corrdinates  [Bx,By,Bz]
    hold_ufc: np.array                                                              
        An array of plasma velocity vectors in the faraday cup corrdinates [Vx,Vy,Vz]
    hold_ifunc: ND interpolation function
        Function of a RectBivariateSpline (https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RectBivariateSpline.html).
        Doing the interpolation this way is much faster than reinterpolating from a grid everytime as early version of the code required.
        The spline function is calculated earlier at in this code at make_discrete_vdf.

    Returns
    -------
    vals: np.array
        An array of interpolated detection currents along the detector
    """
#COMMON measurement_params, hold_vdf, hold_pgrid, hold_qgrid, hold_ufc, hold_bfc


    #Use the magnitude of the input vector
    #vz *= -1

    #break up velocity and magnetic field components
    #These coordinates are in FC coordiante system 
    bx = hold_bfc[0]
    by = hold_bfc[1]
    bz = hold_bfc[2]
    ux = hold_ufc[0]
    uy = hold_ufc[1] 
    uz = hold_ufc[2]

    #print('INPUT VX')
    #print(vx)
    #print('##################')
    #Get measured p and q values
    #print('XXXXXXXXXXXXXXXXXXXXXXX')
    #print(hold_ufc)
    #print((vx-ux),(vy-uy),(vz-uz))
    #Convert FC coordinates to Wper and Wpar coordinates
    #print(vx,vy,vz)
    p = (vx-ux)*bx + (vy-uy)*by + (vz-uz)*bz
    #print(max(p),min(p))
    q = np.sqrt( (vx-ux)**2 + (vy-uy)**2 + (vz-uz)**2 - p**2)
    ###print('start ps and qs')
    ###print(ux,uy,uz)
    ###print(bx,by,bz)
    ####print(p.min(),q.min())
    ####print(p.mean(),q.mean())
    ####print(p.max(),q.max())
    ####print('end ps and qs')



    ####vals = interp.griddata( (hold_pgrid[mind_pq_x,mind_pq_y].ravel(),hold_qgrid[mind_pq_x,mind_pq_y].ravel()),hold_vdf[mind_pq_x,mind_pq_y].ravel(), (p,q),method='nearest', fill_value=0.0)
    #nearest is faster and probably okay with a 0.5km/s grid
    #time_start = time.time()
    #vals = interp.griddata( (hold_pgrid.ravel(),hold_qgrid.ravel()),hold_vdf.ravel(), (p.ravel(),q.ravel()),method='nearest',fill_value=0.0)
    #vals = []
    #for i,j in zip(p,q):
    #    vals.append(hold_ifunc(i,j))
    #turn of grid so it does not calculation all combinations of p and q 2018/08/02 J. Prchlik
    #intepolate measrued values along the grid
    vals = hold_ifunc(p,q,grid=False)
    #code run time
    ##time_elapsed = (time.time() - time_start)
    ####Time elapsed pring
    ##print('Time to run Interpolation  is {0:5.1f}s'.format(time_elapsed))
    ###print(vals)
    return np.array(vals)

def p_bimax_response(x_meas, p_solpar,spacecraft='wind'):
    """
    Parameters
    ----------- 
    x_meas: np.array
               x[0,:]          v_window  [km/s] 
               x[1,:]          v_delta   [km/s] 
               x[2,:]          phi_ang [rad] 
               x[3,:]          theta_ang [rad] 
               x[4,:]      b in FC "x"
               x[5,:]      b in FC "y"
               x[6,:]      b in FC "z" normal to cup
    p_solarpar: np.array
        The solar wind parameters in GSE coordinates [Vx,Vy,Vz,Bx,By,Bz] in units of [km/s,km/s,km/s,nT,nT,nT]

    spacecraft: string, optional
        The spacecraft to use for effective area calculations (Default = 'wind).

    Returns
    ------
    f_p_curr: np.array
        The observed current density per velocity bin assuming a cold plasma 
    """

    co_charge =  1.60217646e-19   # coulombs
    # solar wind proton parameters
    ux  =np.double(p_solpar[0])
    uy  =np.double(p_solpar[1])
    uz  =np.double(p_solpar[2])
    wper=np.double(p_solpar[3])
    wpar=np.double(p_solpar[4])
    Np  =np.double(p_solpar[5])
    # FC measurement parameters
    vn	= np.double(x_meas[0,:])# speed window to measure
    dv	= np.double(x_meas[1,:])# width of speed window to measure
    Pc 	= np.double(x_meas[2,:])# orientation angle of cup (from X to Y) in radians
    Tc 	= np.double(x_meas[3,:])# orientation angle of cup (from XY plane to Z) 
    bxn = np.double(x_meas[4,:])# Magnetic field unit vector
    byn = np.double(x_meas[5,:])
    bzn = np.double(x_meas[6,:])
    bz2	= np.double(bzn**2.)
    # components of bulk speed for each measurement in FC frame
    ###uxfc	=	 ux*np.sin(Pc)         + uy*np.cos(Pc)
    ###uyfc	=	-ux*np.cos(Pc)*np.sin(Tc) + uy*np.sin(Pc)*np.sin(Tc) + uz*np.cos(Tc)
    ###uzfc	=	 ux*np.cos(Pc)*np.cos(Tc) - uy*np.sin(Pc)*np.cos(Tc) + uz*np.sin(Tc)
    ###vxmax = uxfc
    ###vymax = uyfc
    ###vzmax = uzfc
    #only need to send 1 phi and theta measurment 2018/10/04 J. Prchlik
    ufc = convert_gse_fc(p_solpar[:3],Pc[0],Tc[0])
    #print([vxmax,vymax,vzmax])
    #print(ufc)
    vxmax = np.double(ufc[0])
    vymax = np.double(ufc[1])
    vzmax = np.double(ufc[2])
    # get effective area for flow upon the sensor
    # using the approximation that the flow is all coming in at the bulk
    # flow angle (i.e. the cold plasma approximation that the the thermal
    # speed is negligible)
    a_eff_p = np.double(eff_area( vxmax, vymax, vzmax,spacecraft=spacecraft))
    
    # calculate modified gaussian integral of v*f(v). Limits in
    # transverse directions are infinity, i.e. approximate that the whole distribution is
    # within the field of view. This allows us to reduce the integral to Erf() expressions
    vb_norm_p = -np.double(vzmax)
    w_eff_p = np.double(np.sqrt((bz2)*(wpar**2.) + (1.0 - bz2)*(wper**2.) ))
    upper_p     = np.double((1.0/w_eff_p) * (     	vn +   
                                      	0.5 * dv - vb_norm_p ) )
    lower_p     = np.double((1.0/w_eff_p) * (     	vn - 
    					0.5 * dv - vb_norm_p )) 
    f_p_curr_a  = np.double(0.5 * co_charge  * a_eff_p * Np )   
    f_p_curr_b_a  = np.double( vb_norm_p * ( erf( upper_p ) - erf(( lower_p ) )))
    f_p_curr_b_b  = np.double(-(w_eff_p/np.sqrt(np.pi)) *( np.exp( - upper_p**2.) - np.exp( - lower_p**2 ) ) )

    f_p_curr = f_p_curr_a*(f_p_curr_b_a+f_p_curr_b_b)
    
    return f_p_curr



def return_space_craft_ef(spacecraft):
    """
    Returns effective area for a give spacecraft
    Currently set up so the index corresponds to 0.1 degrees on the cup because that is how 
    the wind effected area is defined.
    
    Parameters
    ----------
    spacecraft: str
        String corresponding to the spacecraft you want the effective area for. Currently,
        the only option is wind

    Returns
    -------
    eff_area = np.array
        Array of effective area values where each index corresponds to 0.1 degrees
    """



    #Get wind effective area
    if spacecraft.lower() == 'wind':
        eff_area = np.zeros(910)
        eff_area[  0]=3.3820000e+06 
        eff_area[  1]=3.3821000e+06 
        eff_area[  2]=3.3822000e+06
        eff_area[  3]=3.3823000e+06 
        eff_area[  4]=3.3824000e+06 
        eff_area[  5]=3.3825000e+06
        eff_area[  6]=3.3826000e+06 
        eff_area[  7]=3.3827000e+06 
        eff_area[  8]=3.3828000e+06
        eff_area[  9]=3.3829000e+06 
        eff_area[ 10]=3.3830000e+06 
        eff_area[ 11]=3.3830000e+06
        eff_area[ 12]=3.3830000e+06 
        eff_area[ 13]=3.3830000e+06 
        eff_area[ 14]=3.3830000e+06
        eff_area[ 15]=3.3830000e+06 
        eff_area[ 16]=3.3830000e+06 
        eff_area[ 17]=3.3830000e+06
        eff_area[ 18]=3.3830000e+06 
        eff_area[ 19]=3.3830000e+06 
        eff_area[ 20]=3.3830000e+06
        eff_area[ 21]=3.3829000e+06 
        eff_area[ 22]=3.3828000e+06 
        eff_area[ 23]=3.3827000e+06
        eff_area[ 24]=3.3826000e+06 
        eff_area[ 25]=3.3825000e+06 
        eff_area[ 26]=3.3824000e+06
        eff_area[ 27]=3.3823000e+06 
        eff_area[ 28]=3.3822000e+06 
        eff_area[ 29]=3.3821000e+06
        eff_area[ 30]=3.3820000e+06 
        eff_area[ 31]=3.3819000e+06 
        eff_area[ 32]=3.3818000e+06
        eff_area[ 33]=3.3817000e+06 
        eff_area[ 34]=3.3816000e+06 
        eff_area[ 35]=3.3815000e+06
        eff_area[ 36]=3.3814000e+06 
        eff_area[ 37]=3.3813000e+06 
        eff_area[ 38]=3.3812000e+06
        eff_area[ 39]=3.3811000e+06 
        eff_area[ 40]=3.3810000e+06 
        eff_area[ 41]=3.3809000e+06
        eff_area[ 42]=3.3808000e+06 
        eff_area[ 43]=3.3807000e+06 
        eff_area[ 44]=3.3806000e+06
        eff_area[ 45]=3.3805000e+06 
        eff_area[ 46]=3.3804000e+06 
        eff_area[ 47]=3.3803000e+06
        eff_area[ 48]=3.3802000e+06 
        eff_area[ 49]=3.3801000e+06 
        eff_area[ 50]=3.3800000e+06
        eff_area[ 51]=3.3798000e+06 
        eff_area[ 52]=3.3796000e+06 
        eff_area[ 53]=3.3794000e+06
        eff_area[ 54]=3.3792000e+06 
        eff_area[ 55]=3.3790000e+06 
        eff_area[ 56]=3.3788000e+06
        eff_area[ 57]=3.3786000e+06 
        eff_area[ 58]=3.3784000e+06 
        eff_area[ 59]=3.3782000e+06
        eff_area[ 60]=3.3780000e+06 
        eff_area[ 61]=3.3779000e+06 
        eff_area[ 62]=3.3778000e+06
        eff_area[ 63]=3.3777000e+06 
        eff_area[ 64]=3.3776000e+06 
        eff_area[ 65]=3.3775000e+06
        eff_area[ 66]=3.3774000e+06 
        eff_area[ 67]=3.3773000e+06 
        eff_area[ 68]=3.3772000e+06
        eff_area[ 69]=3.3771000e+06 
        eff_area[ 70]=3.3770000e+06 
        eff_area[ 71]=3.3769000e+06
        eff_area[ 72]=3.3768000e+06 
        eff_area[ 73]=3.3767000e+06 
        eff_area[ 74]=3.3766000e+06
        eff_area[ 75]=3.3765000e+06 
        eff_area[ 76]=3.3764000e+06 
        eff_area[ 77]=3.3763000e+06
        eff_area[ 78]=3.3762000e+06 
        eff_area[ 79]=3.3761000e+06 
        eff_area[ 80]=3.3760000e+06
        eff_area[ 81]=3.3758000e+06 
        eff_area[ 82]=3.3756000e+06 
        eff_area[ 83]=3.3754000e+06
        eff_area[ 84]=3.3752000e+06 
        eff_area[ 85]=3.3750000e+06 
        eff_area[ 86]=3.3748000e+06
        eff_area[ 87]=3.3746000e+06 
        eff_area[ 88]=3.3744000e+06 
        eff_area[ 89]=3.3742000e+06
        eff_area[ 90]=3.3740000e+06 
        eff_area[ 91]=3.3738000e+06 
        eff_area[ 92]=3.3736000e+06
        eff_area[ 93]=3.3734000e+06 
        eff_area[ 94]=3.3732000e+06 
        eff_area[ 95]=3.3730000e+06
        eff_area[ 96]=3.3728000e+06 
        eff_area[ 97]=3.3726000e+06 
        eff_area[ 98]=3.3724000e+06
        eff_area[ 99]=3.3722000e+06 
        eff_area[100]=3.3720000e+06 
        eff_area[101]=3.3717000e+06
        eff_area[102]=3.3714000e+06 
        eff_area[103]=3.3711000e+06 
        eff_area[104]=3.3708000e+06
        eff_area[105]=3.3705000e+06 
        eff_area[106]=3.3702000e+06 
        eff_area[107]=3.3699000e+06
        eff_area[108]=3.3696000e+06 
        eff_area[109]=3.3693000e+06 
        eff_area[110]=3.3690000e+06
        eff_area[111]=3.3689000e+06 
        eff_area[112]=3.3688000e+06 
        eff_area[113]=3.3687000e+06
        eff_area[114]=3.3686000e+06 
        eff_area[115]=3.3685000e+06 
        eff_area[116]=3.3684000e+06
        eff_area[117]=3.3683000e+06 
        eff_area[118]=3.3682000e+06 
        eff_area[119]=3.3681000e+06
        eff_area[120]=3.3680000e+06 
        eff_area[121]=3.3676000e+06 
        eff_area[122]=3.3672000e+06
        eff_area[123]=3.3668000e+06 
        eff_area[124]=3.3664000e+06 
        eff_area[125]=3.3660000e+06
        eff_area[126]=3.3656000e+06 
        eff_area[127]=3.3652000e+06 
        eff_area[128]=3.3648000e+06
        eff_area[129]=3.3644000e+06 
        eff_area[130]=3.3640000e+06 
        eff_area[131]=3.3638000e+06
        eff_area[132]=3.3636000e+06 
        eff_area[133]=3.3634000e+06 
        eff_area[134]=3.3632000e+06
        eff_area[135]=3.3630000e+06 
        eff_area[136]=3.3628000e+06 
        eff_area[137]=3.3626000e+06
        eff_area[138]=3.3624000e+06 
        eff_area[139]=3.3622000e+06 
        eff_area[140]=3.3620000e+06
        eff_area[141]=3.3617000e+06 
        eff_area[142]=3.3614000e+06 
        eff_area[143]=3.3611000e+06
        eff_area[144]=3.3608000e+06 
        eff_area[145]=3.3605000e+06 
        eff_area[146]=3.3602000e+06
        eff_area[147]=3.3599000e+06 
        eff_area[148]=3.3596000e+06 
        eff_area[149]=3.3593000e+06
        eff_area[150]=3.3590000e+06 
        eff_area[151]=3.3586000e+06 
        eff_area[152]=3.3582000e+06
        eff_area[153]=3.3578000e+06 
        eff_area[154]=3.3574000e+06 
        eff_area[155]=3.3570000e+06
        eff_area[156]=3.3566000e+06 
        eff_area[157]=3.3562000e+06 
        eff_area[158]=3.3558000e+06
        eff_area[159]=3.3554000e+06 
        eff_area[160]=3.3550000e+06 
        eff_area[161]=3.3546000e+06
        eff_area[162]=3.3542000e+06 
        eff_area[163]=3.3538000e+06 
        eff_area[164]=3.3534000e+06
        eff_area[165]=3.3530000e+06 
        eff_area[166]=3.3526000e+06 
        eff_area[167]=3.3522000e+06
        eff_area[168]=3.3518000e+06 
        eff_area[169]=3.3514000e+06 
        eff_area[170]=3.3510000e+06
        eff_area[171]=3.3506000e+06 
        eff_area[172]=3.3502000e+06 
        eff_area[173]=3.3498000e+06
        eff_area[174]=3.3494000e+06 
        eff_area[175]=3.3490000e+06 
        eff_area[176]=3.3486000e+06
        eff_area[177]=3.3482000e+06 
        eff_area[178]=3.3478000e+06 
        eff_area[179]=3.3474000e+06
        eff_area[180]=3.3470000e+06 
        eff_area[181]=3.3466000e+06 
        eff_area[182]=3.3462000e+06
        eff_area[183]=3.3458000e+06 
        eff_area[184]=3.3454000e+06 
        eff_area[185]=3.3450000e+06
        eff_area[186]=3.3446000e+06 
        eff_area[187]=3.3442000e+06 
        eff_area[188]=3.3438000e+06
        eff_area[189]=3.3434000e+06 
        eff_area[190]=3.3430000e+06 
        eff_area[191]=3.3425700e+06
        eff_area[192]=3.3421400e+06 
        eff_area[193]=3.3417100e+06 
        eff_area[194]=3.3412800e+06
        eff_area[195]=3.3408500e+06 
        eff_area[196]=3.3404200e+06 
        eff_area[197]=3.3399900e+06
        eff_area[198]=3.3395600e+06 
        eff_area[199]=3.3391300e+06 
        eff_area[200]=3.3387000e+06
        eff_area[201]=3.3382400e+06 
        eff_area[202]=3.3377800e+06 
        eff_area[203]=3.3373200e+06
        eff_area[204]=3.3368600e+06 
        eff_area[205]=3.3364000e+06 
        eff_area[206]=3.3359400e+06
        eff_area[207]=3.3354800e+06 
        eff_area[208]=3.3350200e+06 
        eff_area[209]=3.3345600e+06
        eff_area[210]=3.3341000e+06 
        eff_area[211]=3.3336200e+06 
        eff_area[212]=3.3331400e+06
        eff_area[213]=3.3326600e+06 
        eff_area[214]=3.3321800e+06 
        eff_area[215]=3.3317000e+06
        eff_area[216]=3.3312200e+06 
        eff_area[217]=3.3307400e+06 
        eff_area[218]=3.3302600e+06
        eff_area[219]=3.3297800e+06 
        eff_area[220]=3.3293000e+06 
        eff_area[221]=3.3288000e+06
        eff_area[222]=3.3283000e+06 
        eff_area[223]=3.3278000e+06 
        eff_area[224]=3.3273000e+06
        eff_area[225]=3.3268000e+06 
        eff_area[226]=3.3263000e+06 
        eff_area[227]=3.3258000e+06
        eff_area[228]=3.3253000e+06 
        eff_area[229]=3.3248000e+06 
        eff_area[230]=3.3243000e+06
        eff_area[231]=3.3236900e+06 
        eff_area[232]=3.3230800e+06 
        eff_area[233]=3.3224700e+06
        eff_area[234]=3.3218600e+06 
        eff_area[235]=3.3212500e+06 
        eff_area[236]=3.3206400e+06
        eff_area[237]=3.3200300e+06 
        eff_area[238]=3.3194200e+06 
        eff_area[239]=3.3188100e+06
        eff_area[240]=3.3182000e+06 
        eff_area[241]=3.3176600e+06 
        eff_area[242]=3.3171200e+06
        eff_area[243]=3.3165800e+06 
        eff_area[244]=3.3160400e+06 
        eff_area[245]=3.3155000e+06
        eff_area[246]=3.3149600e+06 
        eff_area[247]=3.3144200e+06 
        eff_area[248]=3.3138800e+06
        eff_area[249]=3.3133400e+06 
        eff_area[250]=3.3128000e+06 
        eff_area[251]=3.3121500e+06
        eff_area[252]=3.3115000e+06 
        eff_area[253]=3.3108500e+06 
        eff_area[254]=3.3102000e+06
        eff_area[255]=3.3095500e+06 
        eff_area[256]=3.3089000e+06 
        eff_area[257]=3.3082500e+06
        eff_area[258]=3.3076000e+06 
        eff_area[259]=3.3069500e+06 
        eff_area[260]=3.3063000e+06
        eff_area[261]=3.3056300e+06 
        eff_area[262]=3.3049600e+06 
        eff_area[263]=3.3042900e+06
        eff_area[264]=3.3036200e+06 
        eff_area[265]=3.3029500e+06 
        eff_area[266]=3.3022800e+06
        eff_area[267]=3.3016100e+06 
        eff_area[268]=3.3009400e+06 
        eff_area[269]=3.3002700e+06
        eff_area[270]=3.2996000e+06 
        eff_area[271]=3.2989200e+06 
        eff_area[272]=3.2982400e+06
        eff_area[273]=3.2975600e+06 
        eff_area[274]=3.2968800e+06 
        eff_area[275]=3.2962000e+06
        eff_area[276]=3.2955200e+06 
        eff_area[277]=3.2948400e+06 
        eff_area[278]=3.2941600e+06
        eff_area[279]=3.2934800e+06 
        eff_area[280]=3.2928000e+06 
        eff_area[281]=3.2921100e+06
        eff_area[282]=3.2914200e+06 
        eff_area[283]=3.2907300e+06 
        eff_area[284]=3.2900400e+06
        eff_area[285]=3.2893500e+06 
        eff_area[286]=3.2886600e+06 
        eff_area[287]=3.2879700e+06
        eff_area[288]=3.2872800e+06 
        eff_area[289]=3.2865900e+06 
        eff_area[290]=3.2859000e+06
        eff_area[291]=3.2850900e+06 
        eff_area[292]=3.2842800e+06 
        eff_area[293]=3.2834700e+06
        eff_area[294]=3.2826600e+06 
        eff_area[295]=3.2818500e+06 
        eff_area[296]=3.2810400e+06
        eff_area[297]=3.2802300e+06 
        eff_area[298]=3.2794200e+06 
        eff_area[299]=3.2786100e+06
        eff_area[300]=3.2778000e+06 
        eff_area[301]=3.2770900e+06 
        eff_area[302]=3.2763800e+06
        eff_area[303]=3.2756700e+06 
        eff_area[304]=3.2749600e+06 
        eff_area[305]=3.2742500e+06
        eff_area[306]=3.2735400e+06 
        eff_area[307]=3.2728300e+06 
        eff_area[308]=3.2721200e+06
        eff_area[309]=3.2714100e+06 
        eff_area[310]=3.2707000e+06 
        eff_area[311]=3.2697900e+06
        eff_area[312]=3.2688800e+06 
        eff_area[313]=3.2679700e+06 
        eff_area[314]=3.2670600e+06
        eff_area[315]=3.2661500e+06 
        eff_area[316]=3.2652400e+06 
        eff_area[317]=3.2643300e+06
        eff_area[318]=3.2634200e+06 
        eff_area[319]=3.2625100e+06 
        eff_area[320]=3.2616000e+06
        eff_area[321]=3.2607900e+06 
        eff_area[322]=3.2599800e+06 
        eff_area[323]=3.2591700e+06
        eff_area[324]=3.2583600e+06 
        eff_area[325]=3.2575500e+06 
        eff_area[326]=3.2567400e+06
        eff_area[327]=3.2559300e+06 
        eff_area[328]=3.2551200e+06 
        eff_area[329]=3.2543100e+06
        eff_area[330]=3.2535000e+06 
        eff_area[331]=3.2526000e+06 
        eff_area[332]=3.2517000e+06
        eff_area[333]=3.2508000e+06 
        eff_area[334]=3.2499000e+06 
        eff_area[335]=3.2490000e+06
        eff_area[336]=3.2481000e+06 
        eff_area[337]=3.2472000e+06 
        eff_area[338]=3.2463000e+06
        eff_area[339]=3.2454000e+06 
        eff_area[340]=3.2445000e+06 
        eff_area[341]=3.2435100e+06
        eff_area[342]=3.2425200e+06 
        eff_area[343]=3.2415300e+06 
        eff_area[344]=3.2405400e+06
        eff_area[345]=3.2395500e+06 
        eff_area[346]=3.2385600e+06 
        eff_area[347]=3.2375700e+06
        eff_area[348]=3.2365800e+06 
        eff_area[349]=3.2355900e+06 
        eff_area[350]=3.2346000e+06
        eff_area[351]=3.2336300e+06 
        eff_area[352]=3.2326600e+06 
        eff_area[353]=3.2316900e+06
        eff_area[354]=3.2307200e+06 
        eff_area[355]=3.2297500e+06 
        eff_area[356]=3.2287800e+06
        eff_area[357]=3.2278100e+06 
        eff_area[358]=3.2268400e+06 
        eff_area[359]=3.2258700e+06
        eff_area[360]=3.2249000e+06 
        eff_area[361]=3.2224200e+06 
        eff_area[362]=3.2199400e+06
        eff_area[363]=3.2174600e+06 
        eff_area[364]=3.2149800e+06 
        eff_area[365]=3.2125000e+06
        eff_area[366]=3.2100200e+06 
        eff_area[367]=3.2075400e+06 
        eff_area[368]=3.2050600e+06
        eff_area[369]=3.2025800e+06 
        eff_area[370]=3.2001000e+06 
        eff_area[371]=3.1962400e+06
        eff_area[372]=3.1923800e+06 
        eff_area[373]=3.1885200e+06 
        eff_area[374]=3.1846600e+06
        eff_area[375]=3.1808000e+06 
        eff_area[376]=3.1769400e+06 
        eff_area[377]=3.1730800e+06
        eff_area[378]=3.1692200e+06 
        eff_area[379]=3.1653600e+06 
        eff_area[380]=3.1615000e+06
        eff_area[381]=3.1567500e+06 
        eff_area[382]=3.1520000e+06 
        eff_area[383]=3.1472500e+06
        eff_area[384]=3.1425000e+06 
        eff_area[385]=3.1377500e+06 
        eff_area[386]=3.1330000e+06
        eff_area[387]=3.1282500e+06 
        eff_area[388]=3.1235000e+06 
        eff_area[389]=3.1187500e+06
        eff_area[390]=3.1140000e+06 
        eff_area[391]=3.1084820e+06 
        eff_area[392]=3.1029640e+06
        eff_area[393]=3.0974460e+06 
        eff_area[394]=3.0919280e+06 
        eff_area[395]=3.0864100e+06
        eff_area[396]=3.0808920e+06 
        eff_area[397]=3.0753740e+06 
        eff_area[398]=3.0698560e+06
        eff_area[399]=3.0643380e+06 
        eff_area[400]=3.0588200e+06 
        eff_area[401]=3.0526550e+06
        eff_area[402]=3.0464900e+06 
        eff_area[403]=3.0403250e+06 
        eff_area[404]=3.0341600e+06
        eff_area[405]=3.0279950e+06 
        eff_area[406]=3.0218300e+06 
        eff_area[407]=3.0156650e+06
        eff_area[408]=3.0095000e+06 
        eff_area[409]=3.0033350e+06 
        eff_area[410]=2.9971700e+06
        eff_area[411]=2.9904530e+06 
        eff_area[412]=2.9837360e+06 
        eff_area[413]=2.9770190e+06
        eff_area[414]=2.9703020e+06 
        eff_area[415]=2.9635850e+06 
        eff_area[416]=2.9568680e+06
        eff_area[417]=2.9501510e+06 
        eff_area[418]=2.9434340e+06 
        eff_area[419]=2.9367170e+06
        eff_area[420]=2.9300000e+06 
        eff_area[421]=2.9227000e+06 
        eff_area[422]=2.9154000e+06
        eff_area[423]=2.9081000e+06 
        eff_area[424]=2.9008000e+06 
        eff_area[425]=2.8935000e+06
        eff_area[426]=2.8862000e+06 
        eff_area[427]=2.8789000e+06 
        eff_area[428]=2.8716000e+06
        eff_area[429]=2.8643000e+06 
        eff_area[430]=2.8570000e+06 
        eff_area[431]=2.8492000e+06
        eff_area[432]=2.8414000e+06 
        eff_area[433]=2.8336000e+06 
        eff_area[434]=2.8258000e+06
        eff_area[435]=2.8180000e+06 
        eff_area[436]=2.8102000e+06 
        eff_area[437]=2.8024000e+06
        eff_area[438]=2.7946000e+06 
        eff_area[439]=2.7868000e+06 
        eff_area[440]=2.7790000e+06
        eff_area[441]=2.7705000e+06 
        eff_area[442]=2.7620000e+06 
        eff_area[443]=2.7535000e+06
        eff_area[444]=2.7450000e+06 
        eff_area[445]=2.7365000e+06 
        eff_area[446]=2.7280000e+06
        eff_area[447]=2.7195000e+06 
        eff_area[448]=2.7110000e+06 
        eff_area[449]=2.7025000e+06
        eff_area[450]=2.6940000e+06 
        eff_area[451]=2.6833000e+06 
        eff_area[452]=2.6725999e+06
        eff_area[453]=2.6618999e+06 
        eff_area[454]=2.6511999e+06 
        eff_area[455]=2.6404999e+06
        eff_area[456]=2.6297998e+06 
        eff_area[457]=2.6190998e+06 
        eff_area[458]=2.6083998e+06
        eff_area[459]=2.5976998e+06 
        eff_area[460]=2.5869998e+06 
        eff_area[461]=2.5747998e+06
        eff_area[462]=2.5625998e+06 
        eff_area[463]=2.5503998e+06 
        eff_area[464]=2.5381999e+06
        eff_area[465]=2.5259999e+06 
        eff_area[466]=2.5137999e+06 
        eff_area[467]=2.5015999e+06
        eff_area[468]=2.4894000e+06 
        eff_area[469]=2.4772000e+06 
        eff_area[470]=2.4650000e+06
        eff_area[471]=2.4514999e+06 
        eff_area[472]=2.4379999e+06 
        eff_area[473]=2.4244999e+06
        eff_area[474]=2.4109998e+06 
        eff_area[475]=2.3974998e+06 
        eff_area[476]=2.3839997e+06
        eff_area[477]=2.3704996e+06 
        eff_area[478]=2.3569996e+06 
        eff_area[479]=2.3434996e+06
        eff_area[480]=2.3299995e+06 
        eff_area[481]=2.3152995e+06 
        eff_area[482]=2.3005996e+06
        eff_area[483]=2.2858997e+06 
        eff_area[484]=2.2711997e+06 
        eff_area[485]=2.2564998e+06
        eff_area[486]=2.2417998e+06 
        eff_area[487]=2.2270998e+06 
        eff_area[488]=2.2123999e+06
        eff_area[489]=2.1977000e+06 
        eff_area[490]=2.1830000e+06 
        eff_area[491]=2.1673000e+06
        eff_area[492]=2.1515999e+06 
        eff_area[493]=2.1358999e+06 
        eff_area[494]=2.1201999e+06
        eff_area[495]=2.1044998e+06 
        eff_area[496]=2.0887998e+06 
        eff_area[497]=2.0730997e+06
        eff_area[498]=2.0573997e+06 
        eff_area[499]=2.0416997e+06 
        eff_area[500]=2.0259996e+06
        eff_area[501]=2.0092997e+06 
        eff_area[502]=1.9925997e+06 
        eff_area[503]=1.9758998e+06
        eff_area[504]=1.9591998e+06 
        eff_area[505]=1.9424999e+06 
        eff_area[506]=1.9257999e+06
        eff_area[507]=1.9091000e+06 
        eff_area[508]=1.8924000e+06 
        eff_area[509]=1.8757001e+06
        eff_area[510]=1.8590001e+06 
        eff_area[511]=1.8414001e+06 
        eff_area[512]=1.8238000e+06
        eff_area[513]=1.8062000e+06 
        eff_area[514]=1.7885999e+06 
        eff_area[515]=1.7709999e+06
        eff_area[516]=1.7533998e+06 
        eff_area[517]=1.7357998e+06 
        eff_area[518]=1.7181997e+06
        eff_area[519]=1.7005997e+06 
        eff_area[520]=1.6829996e+06 
        eff_area[521]=1.6643997e+06
        eff_area[522]=1.6457997e+06 
        eff_area[523]=1.6271998e+06 
        eff_area[524]=1.6085998e+06
        eff_area[525]=1.5899999e+06 
        eff_area[526]=1.5713999e+06 
        eff_area[527]=1.5528000e+06
        eff_area[528]=1.5342000e+06 
        eff_area[529]=1.5156001e+06 
        eff_area[530]=1.4970001e+06
        eff_area[531]=1.4775001e+06 
        eff_area[532]=1.4580000e+06 
        eff_area[533]=1.4385000e+06
        eff_area[534]=1.4189999e+06 
        eff_area[535]=1.3994999e+06 
        eff_area[536]=1.3799998e+06
        eff_area[537]=1.3604998e+06 
        eff_area[538]=1.3409997e+06 
        eff_area[539]=1.3214997e+06
        eff_area[540]=1.3019996e+06 
        eff_area[541]=1.2816997e+06 
        eff_area[542]=1.2613997e+06
        eff_area[543]=1.2410998e+06 
        eff_area[544]=1.2207998e+06 
        eff_area[545]=1.2004999e+06
        eff_area[546]=1.1801999e+06 
        eff_area[547]=1.1599000e+06 
        eff_area[548]=1.1396000e+06
        eff_area[549]=1.1193001e+06 
        eff_area[550]=1.0990001e+06 
        eff_area[551]=1.0778801e+06
        eff_area[552]=1.0567600e+06 
        eff_area[553]=1.0356400e+06 
        eff_area[554]=1.0145199e+06
        eff_area[555]=9.9339984e+05 
        eff_area[556]=9.7227979e+05 
        eff_area[557]=9.5115973e+05
        eff_area[558]=9.3003968e+05 
        eff_area[559]=9.0891962e+05 
        eff_area[560]=8.8779956e+05
        eff_area[561]=8.6586962e+05 
        eff_area[562]=8.4393969e+05 
        eff_area[563]=8.2200975e+05
        eff_area[564]=8.0007981e+05 
        eff_area[565]=7.7814988e+05 
        eff_area[566]=7.5621994e+05
        eff_area[567]=7.3429000e+05 
        eff_area[568]=7.1236006e+05 
        eff_area[569]=6.9043013e+05
        eff_area[570]=6.6850019e+05 
        eff_area[571]=6.4686013e+05 
        eff_area[572]=6.2522007e+05
        eff_area[573]=6.0358002e+05 
        eff_area[574]=5.8193996e+05 
        eff_area[575]=5.6029991e+05
        eff_area[576]=5.3865985e+05 
        eff_area[577]=5.1701979e+05 
        eff_area[578]=4.9537974e+05
        eff_area[579]=4.7373968e+05 
        eff_area[580]=4.5209962e+05 
        eff_area[581]=4.3263968e+05
        eff_area[582]=4.1317973e+05 
        eff_area[583]=3.9371978e+05 
        eff_area[584]=3.7425984e+05
        eff_area[585]=3.5479989e+05 
        eff_area[586]=3.3533994e+05 
        eff_area[587]=3.1588000e+05
        eff_area[588]=2.9642005e+05 
        eff_area[589]=2.7696010e+05 
        eff_area[590]=2.5750016e+05
        eff_area[591]=2.4143012e+05 
        eff_area[592]=2.2536008e+05 
        eff_area[593]=2.0929004e+05
        eff_area[594]=1.9322001e+05 
        eff_area[595]=1.7714997e+05 
        eff_area[596]=1.6107993e+05
        eff_area[597]=1.4500989e+05 
        eff_area[598]=1.2893986e+05 
        eff_area[599]=1.1286982e+05
        eff_area[600]=9.6799781e+04 
        eff_area[601]=8.7173800e+04 
        eff_area[602]=7.7547819e+04
        eff_area[603]=6.7921837e+04 
        eff_area[604]=5.8295856e+04 
        eff_area[605]=4.8669875e+04
        eff_area[606]=3.9043894e+04 
        eff_area[607]=2.9417912e+04 
        eff_area[608]=1.9791931e+04
        eff_area[609]=1.0165950e+04 
        eff_area[610]=5.3996863e+02 
        eff_area[611]=4.8597177e+02
        eff_area[612]=4.3197490e+02 
        eff_area[613]=3.7797804e+02 
        eff_area[614]=3.2398118e+02
        eff_area[615]=2.6998431e+02 
        eff_area[616]=2.1598745e+02 
        eff_area[617]=1.6199059e+02
        eff_area[618]=1.0799373e+02 
        eff_area[619]=5.3996863e+01 
        eff_area[620]=0.0000000e+00
        eff_area[621:909] = 0.0



    return eff_area
