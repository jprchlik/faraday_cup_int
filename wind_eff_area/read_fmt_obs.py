#reads cdf files in python
from spacepy import pycdf
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import make_discrete_vdf as mdv



def convert_e_to_kms(eng):
    """
    Converts energy bin measurements to velocity measurements in km/s by assuming the 
    incoming particles are protons.

    Paramters
    ---------
    eng: float or np.array
        The energy of an FC bin in eV       

    Returns
    -------
    vel: float or np.array
        The velocity of the energy bins in km/s by assuming the particles are protons
    """
    #some constants
    mp = 9.382720813E8 # mass of proton in eV/c^2
    c  = 2.99792458E5  # speed of light in km/s
    vel = c*np.sqrt(2.*eng/mp)
    return vel


def fmt_wind_spec(date,spec_dir='test_obs/test_wind_spectra/',parm_dir=None,
                  spec_fmt = 'wi_sw-ion-dist_swe-faraday_{0:%Y%m%d}_v01.cdf',
                  parm_fmt = 'wi_h1_swe_{0:%Y%m%d}_v01.cdf'):

    """
    Reads Wind/SWE cdf files and formats the for use in reconstruction of 
    the velocity distribution.


    Parameters:
    -----------
    date: str
        The date of the observation you are interested in. This date string 
        must be able to be read by pd.to_datetime().
    spec_dir: str, optional
        The directory containing the Wind ionspectra (Default = 'test_obs/test_wind_spectra/')
    parm_dir: str, optional
        The directory containing the Wind parameter. If None then parm_dir is
        the same as spec_dir (Default = None).
    spec_fmt: str, optional
        A formatted string containing containing the file name format of the ionspectra.
        (Default = 'wi_sw-ion-dist_swe-faraday_{0:%Y$m%d}_v01.cdf').
    parm_fmt: str, optional
        A formatted string containing the file name format of the parameters derived from 
        the spectra (Default = 'wi_h1_swe_{0:%Y%m%d}_v01.cdf').

    Returns
    -------
    fcs: dictionary
        A dictionary containing information on all the Faraday cup measures in Wind. This dictionary
        has the same format as create_grid_vals_multi_fc in the multi_fc_functions.py module. 
    vdf_inpt:dictionary
        A dictionary as created in the make_discrete_vdf function in the make_discrete_vdf.py module.
        It uses the initial parameters from the Non-Linear Wind ion spectra fits to create this dictionary.
    """


    #If the parameter directory is not set, set it to the value in the ion spectrum directory
    if parm_dir is None:
        parm_dir = spec_dir

    #Add / to the end of parm_dir and spec_dir if not already added
    if parm_dir[-1] != '/':
        parm_dir += '/'
    if spec_dir[-1] != '/':
        spec_dir += '/'

    #The observation time
    obs_time = pd.to_datetime(date)

    #full name of the ion spectrum file
    spec_fil = spec_dir+spec_fmt.format(obs_time)

    #full name of the derived parameters file
    parm_fil = parm_dir+parm_fmt.format(obs_time)



    #read in ion spectrum and parameter file
    wind_spec = pycdf.CDF(spec_fil)
    wind_parm = pycdf.CDF(parm_fil)

    #get parameter epoch time closest to requested observation time
    parm_diff = np.abs(wind_parm['Epoch'][...]-obs_time)
    parm_ind, = np.where(parm_diff == parm_diff.min())


    #get velocity components for that time using the nonlinear fitting technique
    vx = wind_parm['Proton_VX_nonlin'][...][parm_ind]
    vy = wind_parm['Proton_VY_nonlin'][...][parm_ind]
    vz = wind_parm['Proton_VZ_nonlin'][...][parm_ind]

    #Get plasma parameters
    wper = wind_parm['Proton_Wperp_nonlin'][...][parm_ind]
    wpar = wind_parm['Proton_Wpar_nonlin'][...][parm_ind]
    dens = wind_parm['Proton_Np_nonlin'][...][parm_ind]


    #replace values greater than 9999
    if wpar > 9990:
        wpar = wper


    #Create a plasma parameter array
    pls_vals = np.array([vx,vy,vz,wper,wpar,dens])



    #store the velocity in an array and find the magnitude of the velocity
    pls_par = np.array([vx,vy,vz]).ravel()
    v_mag = np.sqrt(np.sum(pls_par**2))
    
    #get the gse values of the magnetic field
    bx = wind_parm['BX'][...][parm_ind]
    by = wind_parm['BY'][...][parm_ind]
    bz = wind_parm['BZ'][...][parm_ind]


    #store in a vector
    b_gse = np.array([bx,by,bz]).ravel()
    #convert to a normal vector, which is required for rotations
    b_gse /= np.sqrt(np.sum(b_gse**2))



    #Create a 2D VDF with fit parameters
    vel_clip = 200.
    vdf_inpt = mdv.make_discrete_vdf(pls_vals,b_gse,pres=1.00,qres=1.00,clip=vel_clip)

    #The sampling rate in of the VDF in km/s
    samp = 15.

    #Get effective area calculation from the FC spec cdf
    cup_area = wind_spec['calibration_effArea'][...]
    cup_angl = wind_spec['calibration_angle'][...]
    #create an interpolator function to calculate following effective areas
    #Angles larger than 90 fill with 0
    cup_intp = interp1d(cup_angl, cup_area, kind='linear',fill_value=0.0,bounds_error=False)

    #loop over both faraday cups
    far_cups = [1,2]
    #store all faraday cup information in a structured dictionary
    fcs = {}
    for i,fc in enumerate(far_cups): 
        #store all values in a seperate variable
        cup = 'cup{0:1d}'.format(fc)

        #Get time variable of measurements
        cup_time = wind_spec['Epoch'][...]

        #get the spectroscopic index nearest to input date
        spec_diff = np.abs(cup_time-obs_time)
        spec_ind, = np.where(spec_diff == spec_diff.min())

        #put pertinent variables in np.arrays
        cup_flux = wind_spec[cup+'_qflux'][...]
        cup_azim = wind_spec[cup+'_azimuth'][...]
        cup_incl = wind_spec['inclination_angle'][...][fc-1]+np.zeros(cup_azim.shape[1])
        cup_eprq = wind_spec[cup+'_EperQ'][...]
        cup_edel = wind_spec[cup+'_EperQ_DEL'][...]
        cup_vkms = convert_e_to_kms(cup_eprq) 
        cup_dkms = convert_e_to_kms(cup_eprq+cup_edel)-cup_vkms 




        #loop over all spectral observations at a differnt angle 1 cup at a time
        for j,(phi,theta) in enumerate(zip(cup_azim[spec_ind].ravel(),cup_incl)):
            #set a number for this particular FC
            k = len(fcs.keys())

            #rotate velocity into FC coordinates
            pls_fc = mdv.convert_gse_fc(pls_par,phi,theta)

            #find angle between FC and bulk flow
            fc_bf = np.abs(np.degrees(np.arccos(np.dot([0.,0.,-1.],pls_fc/v_mag))))
        
            #########################################
            #Set up observering condidtions before making any VDFs
            #veloity grid
            #########################################
            ######################################
            #get effective area of wind and other coversion parameters
            #get effective area by interpolating the angle between solar wind and bulk flow
            waeff = cup_intp(fc_bf) #cm^3/km
            q0    = 1.6021892e-7 # picocoulombs
            #velocity grid is static for each epoch
            grid_v= cup_vkms[spec_ind]
            dv    = cup_dkms[spec_ind]
            cont  = 1.e12/(waeff*q0*dv*grid_v)
        
        
            #calculate x_meas array
            #Create array to populate with measurement geometry and range
            x_meas = np.zeros((7,grid_v.size),dtype=np.double)
        
            #current in Amps
            rea_cur = cup_flux[spec_ind,j,:]*1E-12 
            
            #Populate measesurement values
            x_meas[0,:] = grid_v #speed bins
            x_meas[1,:] = dv #speed bin sizes
            x_meas[2,:] = np.radians(phi)
            x_meas[3,:] = np.radians(theta)
        
        
            #get transformed coordinates from GSE to FC
            #only need the first element because the angle are consistent throughout
            #Will now fail if theta and phi are arrays base on updates on 2018/10/04 J. Prchlik
            out_xyz = mdv.convert_gse_fc(b_gse,x_meas[2,0],x_meas[3,0])
        
        
            #populate out_xyz into x_meas
            #It is just a repeat value so just fill entire array with value
            #2018/10/04 J. Prchlik
            x_meas[4,:] = out_xyz[0]
            x_meas[5,:] = out_xyz[1]
            x_meas[6,:] = out_xyz[2]
            
            #convert degree phi and theta to radians
            rad_phi,rad_theta = np.radians((phi,theta))
        
            #create key for input fc
            key = 'fc_{0:1d}'.format(k)
            fcs[key] = {}

        
            #populate key with measurements and parameter 
            fcs[key]['x_meas']  = x_meas
            fcs[key]['rea_cur'] = rea_cur
            fcs[key]['peak']    = v_mag
            fcs[key]['cont']    = cont
            #Add initial guess based on measured Wind parameters
            fcs[key]['init_guess'] = mdv.arb_p_response(fcs[key]['x_meas'],vdf_inpt,samp)


    return fcs,vdf_inpt