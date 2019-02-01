import make_discrete_vdf as mdv
from scipy.interpolate import RectBivariateSpline
import numpy as np
from multiprocessing import Pool
from scipy.optimize import curve_fit, minimize
import matplotlib.pyplot as plt
from fancy_plot import fancy_plot


def proc_wrap(arg):
    return [mdv.arb_p_response(*arg[:-1]),arg[-1]]

def make_discrete_vdf_add_fixed_kernel(dis_vdf,p,q,a_scale=0.1,p_sig=10.,q_sig=10.):
    """
    Returns Discrete Velocity distribution function given a set of input parameters. With fixed variations 
    in the raw vdf

    Parameters:
    -----------
    
    dis_vdf: dictionary
        Dictionary containing a discrete VDF function ['vdf'], the propogating direction grid ['pgrid'],
        the perpendicular to the propogating direction grid ['qgrid'], velocity vector in gse ['u_gse'],
        the normal magnetic field vector ['b_gse'] and a BiVariateSpline interpolation function ['vdf_func'].   
    p: int
        The p-value used for the adding the Gaussian kernel
    q: int
        The q-value used for the adding the Gaussian kernel
    a_scale: float,optional
        Range to vary the input VDF as a fraction of the current value at a given (p,q) (Default = 0.1)
    p_sig: float,optional
        Sigma of the added gaussian in p space in km/s (Default = 10)
    q_sig: float,optional
        Sigma of the added gaussian in q space in km/s (Default = 10)

    Returns:
    ---------
    ran_vdf: dictionary
        Dictionary containing a discrete VDF function ['vdf'], the propogating direction grid ['pgrid'],
        the perpendicular to the propogating direction grid ['qgrid'], velocity vector in gse ['u_gse'],
        the normal magnetic field vector ['b_gse'] and a BiVariateSpline interpolation function ['vdf_func'].   
    """

    #copy previous variables
    pgrid = dis_vdf['pgrid']
    qgrid = dis_vdf['qgrid']
    u_gse = dis_vdf['u_gse']
    mag_par = dis_vdf['b_gse']

     
    #grab the raw vdf
    rawvdf = dis_vdf['vdf']

    
    #calculate amplitude of vdf at p_grab, q_grab
    a = float(dis_vdf['vdf_func'](p,q,grid=False))


    #change p,q variable name
    p_grab, q_grab = p, q
    #add variation guassian to rawvdf
    ranvdf = a_scale*a*np.exp(- ((pgrid-p_grab)/p_sig)**2. - ((qgrid-q_grab)/q_sig)**2.)+rawvdf

    #replace with original values if something is less than 0
    ranvdf[ranvdf < 0] = rawvdf[ranvdf < 0]

    #create an interpolating function for the vdf
    f = RectBivariateSpline(pgrid[:,0],qgrid[0,:],ranvdf)

    #create dictionary
    ran_vdf = {'vdf':ranvdf,'pgrid':pgrid,'qgrid':qgrid,'u_gse':u_gse,'b_gse':mag_par,'vdf_func':f}
    return ran_vdf


def get_variation_grid(fcs,dis_vdf,p_num=10,q_num=10,a_scale=0.1,nproc=10):
    """
    Returns grid of error values when adding a grid of Gaussian kernels

    Parameters:
    -----------
    fcs: dictionary
        Dictionary of multiple faraday cup measurements of the same solar wind
    dis_vdf: dictionary
        Dictionary containing a discrete VDF function ['vdf'], the propogating direction grid ['pgrid'],
        the perpendicular to the propogating direction grid ['qgrid'], velocity vector in gse ['u_gse'],
        the normal magnetic field vector ['b_gse'] and a BiVariateSpline interpolation function ['vdf_func'].   
    p_num: int,optional
        Number of p values to use when computing a grid of values to compute the effect of varying by a Gaussian kernel
    q_num: int,optional
        Number of q values to use when computing a grid of values to compute the effect of varying by a Gaussian kernel
    a_scale: float,optional
        Range to vary the input VDF as a fraction of the current value at a given (p,q) (Default = 0.1)
    n_proc: int, optional
        Number of processors to use when integrating the p,q velocity distributions into measurements
    """

    #copy previous variables
    pgrid = dis_vdf['pgrid']
    qgrid = dis_vdf['qgrid']
    u_gse = dis_vdf['u_gse']
    mag_par = dis_vdf['b_gse']


    #p and q values to consider
    ps = np.linspace(pgrid.min(),pgrid.max(),p_num)
    qs = np.linspace(qgrid.min(),qgrid.max(),q_num)

    #do positive and negative values
    pn = [1,-1]

    #error array
    err_arr = []

    #loop over p and q then compute errors for each p,q pair
    for p in ps:
        for q in qs:
            for s in pn:

                #compute added Gaussian kernel at fixed position
                dis_vdf_bad = make_discrete_vdf_add_fixed_kernel(dis_vdf,p,q,a_scale=a_scale*s)
                
                #variables to pass to parrallel processing
                looper = []

                #loop over all fc in fcs to populate with new VDF guess
                for i,key in enumerate(fcs.keys()):
                    #add variation and store which faraday cup you are working with using key
                    #Updated with varying integration sampling function 2018/10/12 J. Prchlik
                    inpt_x = fcs[key]['x_meas'].copy()
                    g_vdf  = dis_vdf_bad.copy()
                    peak   =  fcs[key]['peak'].copy()
                    looper.append((inpt_x,g_vdf,3.,key))

                #process in parallel
                pool = Pool(processes=nproc)
                dis_cur = pool.map(proc_wrap,looper)
                pool.close()
                pool.join()

                #break into index value in looper and the 1D current distribution
                index   = np.array(zip(*dis_cur)[1])
                dis_cur = np.array(zip(*dis_cur)[0])


                #get sum squared best fit
                tot_err = np.zeros(dis_cur.shape[0])
                #Get error in each faraday cup
                for j,i in enumerate(index):
                     tot_err[j] = np.sum((dis_cur[j,:] - fcs[i]['rea_cur'])**2)
 
                #sum squared error for all FC
                fcs_err = np.sqrt(np.sum(tot_err))
  
                #store values of p,q,+/-, and total error for a given p,q,+/-
                err_arr.append([p,q,s,fcs_err]) 


    #Add Zero order addition to the grid
    for j,i in enumerate(fcs.keys()):
         tot_err[j] = np.sum((dis_cur[j,:] - fcs[i]['rea_cur'])**2)
    #sum squared error for all FC without adding any Gaussians
    zro_err = np.sqrt(np.sum(tot_err))

    #Add zero order error to array
    err_arr.append([0,0,0,zro_err])

    #convert the error list into a numpy array
    err_arr = -np.array(err_arr)+zro_err  

    #if less than 0 then set to 0
    err_arr[err_arr < 0.] = 0.
  
    return err_arr


def create_random_vdf_multi_fc(fcs,nproc,cur_err,dis_vdf_guess,pred_grid,kernel,improved=False,samp=3.,verbose=False,ip=0.,iq=0.,n_p_prob=[0.5,0.5],sc_range=0.1):
    """
 
    Parameters
    -----------
    fcs: dictionary
        Dictionary of multiple faraday cup measurements of the same solar wind
    nproc : int
        Number of processors to run on
    cur_err : float
        Current sum squared errors between the observations and the integrated VDF
    dis_vdf_guess: np.array
        The current best fit 2D VDF guess
    pred_grid: np.array
        A numpy array with the same shape as the p and q grids, which is the probability of selecting a point 
        any where on the grid. The grid should be updated in a way that guesses that improve the fit are favored 
        over guess that do not.
    kernel: float
        The size of the Gaussian kernel to use in both the p and q directions
    improved: boolean, optional
        Whether the previous iteration improved the fit (Default = False)
    samp : int, optional
        The number of samples to use in 3D integration
    verbose: boolean
        Print Chi^2 min values when a solution improves the fit
    ip: float, optional
        Location of the last Gaussian kernel guess in the P coordinate (Default = 0.). If improved is true then
        this coordinate improved the fit and will be used for the next guess.
    iq: float, optional
        Location of the last Gaussian kernel guess in the Q coordinate (Default = 0.). If improved is true then
        this coordinate improved the fit and will be used for the next guess.
    n_p_prob: list or np.array, optional
        The probability of selecting a positive or negative gaussian. The first element is the probability
        of selecting a gaussian that removes from the vdf, while the second element is the probability of 
        selecting a gaussian that adds to the vdf. The total probability must sum to 1 (default = [0.5,0.5]).
    sc_range: float,optional
        Range to vary the input VDF as a fraction (Default = 0.1)

    Returns
    -------
    fcs: dictionary
        Dictionary of FC measurements and parameters
    fcs_err: float
        The total sum squared error for all FC given the current 2D velocity model
    dis_vdf_bad: np.array
        2D numpy array describing the best fit velocity distribution 
    improved: boolean, optional
        Whether the previous iteration improved the fit (Default = False)
    ip: float, optional
        Location of the last Gaussian kernel guess in the P coordinate (Default = 0.). If improved is true then
        this coordinate improved the fit and will be used for the next guess.
    iq: float, optional
        Location of the last Gaussian kernel guess in the Q coordinate (Default = 0.). If improved is true then
        this coordinate improved the fit and will be used for the next guess.
    n_p_prob: list or np.array, optional
        The probability of selecting a positive or negative gaussian. The first element is the probability
        of selecting a gaussian that removes from the vdf, while the second element is the probability of 
        selecting a gaussian that adds to the vdf. The total probability must sum to 1 (default = [0.5,0.5]).

  
    Usage
    ------
    create_random_vdf_multi_fc(proc,cur_err)
   
    """

    #looper for multiprocessing
    looper = []


    #Total intensity measurement
    tot_I = 0
    for key in fcs.keys():
        tot_I += np.sum(fcs[key]['rea_cur'])


    sc_range = 0.1
    #on the first loop do not update
    if cur_err > 1e30:
        dis_vdf_bad = dis_vdf_guess
        n_p = 1.

    else:
        #Add Guassian 2D perturbation
        #create Vper and Vpar VDF with pertabation from first key because VDF is the same for all FC
        #use 3% of total width for Gaussian kernels instead of 10% 2018/09/19 J. Prchlik
        #Switched to fixed kernel size 2018/10/25 J. Prchlik
        kernel_size = kernel # Switched so current error is a percent/tot_I*100. 2018/10/24 J. Prchlik
        
        #Get the new velocity distribution and the location and sign of the added Gaussian
        dis_vdf_bad,ip,iq,n_p = mdv.make_discrete_vdf_random(dis_vdf_guess,pred_grid,improved=improved,sc_range=sc_range,
                                                         p_sig=kernel_size,q_sig=kernel_size,ip=ip,iq=iq,
                                                         n_p_prob=n_p_prob)
                        

    #loop over all fc in fcs to populate with new VDF guess
    for i,key in enumerate(fcs.keys()):
        #add variation and store which faraday cup you are working with using key
        #Updated with varying integration sampling function 2018/10/12 J. Prchlik
        inpt_x = fcs[key]['x_meas'].copy()
        g_vdf  = dis_vdf_bad.copy()
        peak   =  fcs[key]['peak'].copy()
        looper.append((inpt_x,g_vdf,samp,key))

    #process in parallel
    pool = Pool(processes=nproc)
    dis_cur = pool.map(proc_wrap,looper)
    pool.close()
    pool.join()

    #break into index value in looper and the 1D current distribution
    index   = np.array(zip(*dis_cur)[1])
    dis_cur = np.array(zip(*dis_cur)[0])


    #get sum squared best fit
    tot_err = np.zeros(dis_cur.shape[0])
    tot_int = np.zeros(dis_cur.shape[0])
    #Get error in each faraday cup
    for j,i in enumerate(index):
         tot_err[j] = np.sum((dis_cur[j,:] - fcs[i]['rea_cur'])**2)
         tot_int[j] = np.sum(( fcs[i]['rea_cur'])**2)

    #print(tot_err)
    #total error for all fc
    #fcs_err = np.median(tot_err)
    fcs_err = np.sqrt(np.sum(tot_err))/np.sqrt(np.sum(tot_int))
    

    #return best VDF and total error
    if fcs_err < cur_err:
        if verbose:
            print('STATS for this run')
            print('Old Err = {0:4.3e}'.format(cur_err))
            print('New Err = {0:4.3e}'.format(fcs_err))
            print('X2 Err = {0:4.3e}'.format(np.sum(np.sum((dis_cur - rea_cur)**2.,axis=1))))
            print('Mean Err = {0:4.3e}'.format(tot_err.mean()))
            print('Med. Err = {0:4.3e}'.format(np.median(tot_err)))
            print('Max. Err = {0:4.3e}'.format(np.max(tot_err)))
            print('Min. Err = {0:4.3e}'.format(np.min(tot_err)))
            print(tot_err)
            print('##################')
        #updated measured currents
        for j,i in enumerate(index):
            fcs[i]['dis_cur'] = dis_cur[j]

        #create probability array for +/- if previous guess worked
        #keep the same sign for the next guess
        n_p_prob = np.zeros(2)
        if n_p > 0:
            n_p_prob[1] = 1
        else:
            n_p_prob[0] = 1

        #Whether to repeat the try nearby ord not
        #only replace if it reduces the error by greater than 1%
        repeat  = np.abs(fcs_err-cur_err)/cur_err > 0.0001

        #Only keep previous coordinate if imporovment is better than 1 part in 100
        #if np.absfcs_err-cur_err)/cur_err > 0.01:
        return fcs,fcs_err,dis_vdf_bad,repeat,ip,iq,n_p_prob
        #else:
        #    return fcs,fcs_err,dis_vdf_bad,False,ip,iq,n_p_prob
    #If no improvement just return the pervious value
    else:
        #keep 50/50 guess for next guess if no improvement
        n_p_prob = np.zeros(2)+0.5
        return fcs,cur_err,dis_vdf_guess,False,ip,iq,n_p_prob
    
def create_grid_vals_multi_fc(fcs,proc,cur_err,dis_vdf_guess,verbose=False):
    """
    Parameters
    -----------
    fcs: dictionary
        Dictionary of multiple faraday cup measurements of the same solar wind
    cur_err : float
        Current sum squared errors between the observations and the integrated VDF
    dis_vdf_guess: np.array
        The current best fit 2D VDF guess
    samp : int, optional
        The number of samples to use in 3D integration
    verbose: boolean
        Print Chi^2 min values when a solution improves the fit

    Returns
    -------
        looper[best_ind][1],tot_err[best[0]],dis_cur[best[0]]
  
    Usage
    ------
    create_random_vdf_multi_fc(proc,cur_err)
   
    """

    #looper for multiprocessing
    looper = []


    #Total intensity measurement
    tot_I = 0
    for key in fcs.keys():
        tot_I += np.sum(fcs[key]['rea_cur'])


    sc_range = 0.1
    #on the first loop do not update
    if cur_err > 1e30:
        dis_vdf_bad = dis_vdf_guess

    else:
        #Add Guassian 2D perturbation
        #create Vper and Vpar VDF with pertabation from first key because VDF is the same for all FC
        #use 3% of total width for Gaussian kernels instead of 10% 2018/09/19 J. Prchlik
        kernel_size = 5.
        
        dis_vdf_bad = mdv.make_discrete_vdf_random(dis_vdf_guess,sc_range=sc_range,p_sig=kernel_size,q_sig=kernel_size)
                        

    #loop over all fc in fcs to populate with new VDF guess
    for i,key in enumerate(fcs.keys()):
        #add variation and store which faraday cup you are working with using key
        #Updated with varying integration sampling function 2018/10/12 J. Prchlik
        inpt_x = fcs[key]['x_meas'].copy()
        g_vdf  = dis_vdf_bad.copy()
        peak   =  fcs[key]['peak'].copy()
        looper.append((inpt_x,g_vdf,peak,key))

    #process in parallel
    pool = Pool(processes=nproc)
    dis_cur = pool.map(proc_wrap,looper)
    pool.close()
    pool.join()

    #break into index value in looper and the 1D current distribution
    index   = np.array(zip(*dis_cur)[1])
    dis_cur = np.array(zip(*dis_cur)[0])


    #get sum squared best fit
    tot_err = np.zeros(dis_cur.shape[0])
    #Get error in each faraday cup
    for j,i in enumerate(index):
         tot_err[j] = np.sum((dis_cur[j,:] - fcs[i]['rea_cur'])**2)

    #print(tot_err)
    #total error for all fc
    #fcs_err = np.median(tot_err)
    fcs_err = np.sqrt(np.sum(tot_err))
    

    #return best VDF and total error
    if fcs_err < cur_err:
        if verbose:
            print('STATS for this run')
            print('Old Err = {0:4.3e}'.format(cur_err))
            print('New Err = {0:4.3e}'.format(fcs_err))
            print('X2 Err = {0:4.3e}'.format(np.sum(np.sum((dis_cur - rea_cur)**2.,axis=1))))
            print('Mean Err = {0:4.3e}'.format(tot_err.mean()))
            print('Med. Err = {0:4.3e}'.format(np.median(tot_err)))
            print('Max. Err = {0:4.3e}'.format(np.max(tot_err)))
            print('Min. Err = {0:4.3e}'.format(np.min(tot_err)))
            print(tot_err)
            print('##################')
        #updated measured currents
        for j,i in enumerate(index):
            fcs[i]['dis_cur'] = dis_cur[j]
        return fcs,fcs_err,dis_vdf_bad
    #If no improvement just return the pervious value
    else:
        return fcs,cur_err,dis_vdf_guess
    

def create_fc_grid_plot(fcs,waeff=3.8e6,q0=1.6021892e-7):
    """
    Plot multiple FC on one grid of plots

    Parameters
    ----------
    fcs: dictionary
        Dictionary of FC measurements
    waeff: float, optional
        Effective array of FC  (Default = 3.8e6 #cm^3/km, Wind)
    q0: float
        Some constant I can't recally this moment (Default  = 1.6021892e-7 # picocoulombs)

    Return
    -------
    fig, ax: Figure Object, Axis Object
        The created plots figure and axis objects
    """



    #create a square as possible plot
    nrows = int(np.sqrt(len(fcs.keys())))
    ncols = len(fcs.keys())/nrows
    #add an extra column if needed
    if nrows*ncols < len(fcs.keys()): 
        ncols += 1
       
    
    #create figure to plot
    fig, axs = plt.subplots(ncols=ncols,nrows=nrows,sharex=True,sharey=True,figsize=(3*ncols,3*nrows))
    fig.subplots_adjust(wspace=0.01,hspace=0.01)
    counter = 0
    for k,ax in zip(range(len(fcs.keys())),axs.ravel()):
        key = 'fc_{0:1d}'.format(k)

        #first column of x_measure is the velocity grid
        grid_v = fcs[key]['x_meas'][0,:]
        dv    = np.diff(grid_v)
        dv    = np.concatenate([dv,[dv[-1]]])
        cont  = 1.e12/(waeff*q0*dv*grid_v)

        #removed to test improving Gaussian fit
        ax.plot(grid_v,fcs[key]['dis_cur'].ravel()*cont,label='Best MC',color='black',linewidth=3)
        ax.plot(grid_v,fcs[key]['rea_cur'].ravel()*cont,'-.b',label='Input',linewidth=3)
        #ax.plot(grid_v,rea_cur.ravel()*cont,'-.b',label='Input',linewidth=3)
        ax.plot(grid_v,fcs[key]['init_guess'].ravel()*cont,':',color='purple',label='Init. Guess',linewidth=3)
        ax.text(0.05,0.8,'$\Phi$={0:2.1f}$^\circ$\n$\Theta$={1:2.1f}$^\circ$'.format(*np.degrees(fcs[key]['x_meas'][[2,3],0])),transform=ax.transAxes)
        #ax.plot(grid_v, gaus(grid_v,*popt),'--',marker='o',label='Gauss Fit',linewidth=3)
        fancy_plot(ax)
        #ax.set_yscale('log')
        #only plot x-label if in the last row
        if counter >= (nrows-1)*(ncols):
           ax.set_xlabel('Speed [km/s]')
        #set ylabel on the left edge
        if np.isclose(float(counter)/ncols - int(counter/ncols),0):
            ax.set_ylabel('p/cm$^{-3}$/(km/s)')
        #put legend only on the first plot
        if counter == 0:
            ax.legend(loc='best',frameon=False)
        counter += 1

    return fig,axs

def create_multi_fc(dis_vdf,ncup,phi_lim=90.,tht_lim=30.,random_seed=None,v_rng=150.,v_smp=20,i_smp=15):
    """
    Create a random distribution of FC between +/- phi and theta limits,
    where phi is angle between FC normal and positive Xgse and theta is the angle
    between the XYgse plane and the FC normal. Then make the measurements for the
    given FC.

    Parameters
    ----------
    dis_vdf: dictionary
        discrete vdf object created by mdv.make_discrete_vdf()
    ncup: int
        Number of random FC to create measurement arrays for
    phi_lim: float, optional
        The absolute limit of phi angles allowed by the random pull (i.e. +/-phi_lim,
        Default = 90).
    tht_lim: float, optional
        The absolute limit of theta angles allows by the random pull (i.e. +/-tht_lim,
        Default = 30)
    random_seed: np.random.RandomState object, optional
        The random seed to use for selecting a given set of FCs. If None
        then the code will get its own random state (Default = None).
    v_rng: float, optional
        Range of velocities (+/-) in km/s of the "measuring" FC with respect to the peak (Default = 150).
    v_smp:int
        Number of measurment bins on the FC (Default = 20)
    i_smp:int
        Sampling of the integrating function in p',q',r' coordiantes in km/s (Default = 15)

    Returns
    ----------
    fcs: dictionary
        A dictionary of FC measurements which includes one key for each FC in the form fc_{0:1d}.
        For each FC is a dictionary containing four keys, which I will now described. 'x_meas' 
        is a measurement array created by mdv.arb_p_response. 'rea_cur' is what the FC would
        measure given the x_meas array, the input 2D VDF, and some sampling frequency (i_smp).
        'peak' is the absolute value of the z component of the velocity in the FC frame.
        'cont' is the constant to convert the "measured current" into p/cc/(km/s).

    big_arr = np.array
        An array ncup by 8 np.array of Gaussian measurment of the solar wind parameters according 
        a given FC orientation. The first column is the speed, 2nd the Gaussian width, 3rd the
        number density, 4th FC phi angle in radians, 5th FC theta angle in radians, 6th the
        speed uncertainty, 7th the width uncertainty, and 8th the number density uncertainty.
    """


    #get a random state if one is not given
    if random_seed == None:
        random_seed = np.random.RandomState()




    #get random set of phis and theta for a given state
    phis   = random_seed.uniform(-phi_lim,phi_lim,size=ncup)
    thetas = random_seed.uniform(-tht_lim,tht_lim,size=ncup)


    #plasma_velocties from generated observation
    pls_par = dis_vdf['u_gse']


    #array that store all the fit parameters 
    big_arr = []
    #calculate the "real measured reduced response function" for all fc cups
    fcs = {}
    for k,(phi,theta) in enumerate(zip(phis,thetas)):

        #rotate velocity into FC coordinates
        #Just use velocity peak
        pls_fc = mdv.convert_gse_fc(pls_par,phi,theta)

        ####Assume a 45 FoV cut off
        pls_fc[:3] *= np.cos(np.radians(45.))

        #########################################
        #Set up observering condidtions before making any VDFs
        #veloity grid
        #########################################
        ######################################
        #get velocity normal to the FC
        v_mag = np.sqrt(np.sum(pls_fc[:3]**2))
        grid_v = np.arange(v_mag-v_rng,v_mag+v_rng,v_smp)
        #switch to linear sampling to make sure all are the same size
        grid_v = np.linspace(v_mag-v_rng,v_mag+v_rng,v_smp)

        #do not let grid go below 180 km/s
        if grid_v.min() < 180.:
            grid_v += 180-grid_v.min()
        if grid_v.max() > 1300.:
            grid_v -= 1300-grid_v.max()
        #get effective area of wind and other coversion parameters
        waeff = 3.8e6 #cm^3/km
        q0    = 1.6021892e-7 # picocoulombs
        dv    = np.diff(grid_v)
        dv    = np.concatenate([dv,[dv[-1]]])
        cont  = 1.e12/(waeff*q0*dv*grid_v)


        #calculate x_meas array
        x_meas = mdv.make_fc_meas(dis_vdf,fc_spd=grid_v,fc_phi=phi,fc_theta=theta)
        #compute the observed current in the instrument
        #Use dynamic sampling 2018/10/12 J. Prchlik
        #rea_cur = mdv.arb_p_response(x_meas,dis_vdf,samp)
        rad_phi,rad_theta = np.radians((phi,theta))
        #here sampling is in km/s
        rea_cur = mdv.arb_p_response(x_meas,dis_vdf,i_smp)
        #switched back to static sampling but now using p',q',r' for sampling
        #rea_cur = mdv.arb_p_response_dyn_samp(x_meas,dis_vdf,peak)
    
        #create key for input fc
        key = 'fc_{0:1d}'.format(k)
        fcs[key] = {}
    
        #populate key with measurements and parameter 
        fcs[key]['x_meas']  = x_meas
        fcs[key]['rea_cur'] = rea_cur
        fcs[key]['peak']    = v_mag
        fcs[key]['cont']    = cont
    
    
        #calculate the Gaussian fit of the response
        try:
            popt, pcov = curve_fit(gaus,grid_v,rea_cur*cont,p0=[np.nanmax(rea_cur*cont),np.mean(grid_v),np.sqrt(2.)*2*dv[0]],sigma=1./(rea_cur/np.nanmin(rea_cur)),maxfev=5000)
        except RuntimeError:
            #give number that will be thrown out if no fit is found 
            popt = np.zeros(3)-9999.9
            pcov = np.zeros((3,3))-9999.9
    
    
        #Switched to computing the average
        #####get the parameters from the fit
        u = popt[1] #speed in km/s
        w = np.abs(popt[2]*np.sqrt(2.)) #thermal speed in km/s
        n = popt[0]*w*np.sqrt(np.pi) #density in cc
        ####
        #####uncertainty in parameters from fit
        du = np.sqrt(pcov[1,1])
        dw = np.sqrt(pcov[2,2])
        dn = np.sqrt(np.pi*((w**2.*pcov[0,0]) + (dw*n)**2))
    
    
        #Add fit parameters with velocity guesses
        big_arr.append([u,w,n,phi,theta,du,dw,dn])
    
    
    #convert big_arr intop numpy array
    big_arr = np.array(big_arr)


    return fcs,big_arr

def mc_reconstruct(fcs,nproc,dis_vdf,pred_grid,kernel,iters,
                  tot_err=1e31,improved=False,ip=0.,iq=0.,
                  n_p_prob=np.array([0.5,0.5]),sc_range=0.1,samp=15,
                  min_kernel=15.,verbose=False,counter=0,default_grid=None,
                  tol_cnt=100,return_convergence=False):
    """
	mc_reconstruct attempts to reconstruct a 2D velocity distribution function (VDF) from multiple 1D Faraday Cup (FC) measurements. The program 
	does this by iteratively adding a Gaussian kernel to a initial reconstruction of the 2D VDF. For each iteration, the program checks whether the added
	Gaussian kernel reduces the squared residuals of the integrated "pseudo" 2D VDF compared to the 1D FC observations. If True, then the program accepts
	the newly added Gaussian kernel to the 2D reconstruction, stores whether the kernel location, and whether the kernel was postive or negative. It uses
    the kernel location to updated the probability array to allow more guesses in a radius equal to the parallel (p), perpendicular (q) value. It uses
    the kernel sign to decide whether the next guess should be positive or negative. If the function cannot find a improving kernel after tol_cnt iterations,
    it skrinks the kernel size by 10% and resets the probability array to the input value. The kernel size plateus at a min value of min_kernel. 
 
    Parameters
    ----------
    fcs: dictionary
        A dictionary of FC measurements which includes one key for each FC in the form fc_{0:1d}.
        For each FC is a dictionary containing four keys, which I will now described. 'x_meas' 
        is a measurement array created by mdv.arb_p_response. 'rea_cur' is what the FC would
        measure given the x_meas array, the input 2D VDF, and some sampling frequency (i_smp).
        'peak' is the absolute value of the z component of the velocity in the FC frame.
        'cont' is the constant to convert the "measured current" into p/cc/(km/s).
    nproc: int
        Number of processors to use when integrating FCs.
    dis_vdf: dictionary
        discrete vdf object created by mdv.make_discrete_vdf(). Use the dictionary VDF
        recreated from the "measured" plasma parameters.
    pred_grid: np.array
        A probability array to be used when to selecting the location of a new Gaussian
        kernel location (ip,iq). The sum of the array must be 1 and be the same shape
        a dis_vdf['pgrid']/vdf_vdf['qgrid'].
    kernel: float
        The size of the Gaussian kernel to add or subtract at location ip,iq
    iters: int
        The maximum number of iterations before convergence
    tot_err: float, optional
        The total sum squared error for each FC normalized by the total signal squared in the FC summed in quadrature
        (Default 1e31)
    improved: boolean, optional
        Whether the last interation of p,q improved the fit (Default = False). Only really
        useful if running mc_reconstruct multiple times.
    ip: float, optional
        The initial point in the p coordiante. Only readlly useful if itering mc_reconstrcturion (Default = 0.)
    iq: float, optional
        The initial point in the q coordiante. Only readlly useful if itering mc_reconstrcturion (Default = 0.)
    n_p_prob: np.array, optional
        Probability the Gaussian is positive or negative (Default = [0.5,0.5]). The default is 50/50.
    samp: int, optional
        The velocity width of the sample points in p,q space
    min_kernel: float, optional
        The minimum kernel size used for image reconstruction
    counter: int, optional
        Counter value until reset to default_grid for probabilities of p,q points (Default = 0)
    default_grid: np.array, optional
        The default grid to use when the counter equals tol_cnt (Default = None, which is pred_grid)
    tol_cnt: int, optional
        The tolerance value when the prediction grid should reset to the default_grid (Default = 100).
    verbose: boolean, optional
        Print the total error for each iteration
    return_convergence: boolean, optional
        Store the error per iteration number and return when finished (Default = False)

    Returns
    ----------
    fcs,dis_vdf,pred_grid,kernel,improved,ip,iq,n_p_prob,counter, (per_err_list)
        All input parameters updated by this module. If return_convergence is True then return percent error
        per iteration number (per_err_list) in addition to previous parameters
   
    """

    #set default grid if not set
    if not isinstance(default_grid,np.ndarray):
        default_grid = pred_grid.copy()
    
    #init pre_err variable
    per_err = tot_err

    #set up list of percent errors
    if return_convergence:
        per_err_list = []
        #list of kernel sizes for each guess
        ker_sze_list = []


    #removed to test improving fit
    for i in range(iters):
        #error from previous iteration
        pre_err = per_err
        #get a new vdf and return if it is the best fit
        #dis_vdf_bad,tot_error,dis_cur = create_random_vdf(dis_vdf_bad,nproc,n_p_prob)
        #print(ip,iq,n_p_prob)
        fcs,tot_err,dis_vdf,improved,ip,iq,n_p_prob = create_random_vdf_multi_fc(fcs,nproc,tot_err,
                                                                                dis_vdf,pred_grid,
                                                                                kernel,
                                                                                improved=improved,ip=ip,
                                                                                iq=iq,n_p_prob=n_p_prob,
                                                                                sc_range=sc_range,samp=samp)
        
        if improved:
            #scale probability by how large of a jump is made
            if pre_err > 1e30:
                scale = 0.1
            else:
                scale = (pre_err-tot_err)
                if scale < 0.:
                    scale = 0.
            
                #calculate peak at ip,iq value using the percent error change
                a = 100.*(scale)/float(pred_grid.size)
                
                #Also add a band at a radius p,q values
                r_ipq = np.sqrt(ip**2+iq**2)
                #Add probability kernel at that location
                pred_grid += a*np.exp(- ((np.sqrt((dis_vdf['pgrid']**2+dis_vdf['qgrid']**2))-(r_ipq))/kernel)**2.)
                #normalize to 1
                pred_grid /= np.sum(pred_grid)
                
                #Remove 10 guesses for counter for bad guesses
                counter -= 10
                #Do not let counter go below 0
                if counter < 0:
                    counter = 0
        else:
            #increment the bad guess for the current model pdf
            counter += 1
            #reset the grid back to the default if there are too many bad guess with current pdf
            #do not probe velocity structures less than the measured spacing in the FC
            if (counter > tol_cnt):
                #reset the grid
                pred_grid = default_grid.copy()
                counter = 0
                if kernel >= min_kernel: 
                   #decrease the kernel size by 10%
                   kernel *= 0.9
            
        #updated per_err with new tot_err
        per_err = tot_err
    
        if verbose:
            print(counter) 
            print('Total error for iteration {0:1d} is {1:4.3f}%'.format(i,100.*float(tot_err)))

        if return_convergence:
            per_err_list.append(tot_err*100.)
            ker_sze_list.append(kernel)

    #Add per_err_list to output if return_convergence is set
    if return_convergence:
        return fcs,dis_vdf,pred_grid,kernel,improved,ip,iq,n_p_prob,counter,per_err_list,ker_sze_list
    else:
        return fcs,dis_vdf,pred_grid,kernel,improved,ip,iq,n_p_prob,counter
    

def gaus(x,a,x0,sigma):
    """
    Gaussian function 

    Parameters
    ------------
    x: np.array or float
        The independent value for a 1D Gaussian
    a: float
        The amplitude of Gaussian
    x0: float
        The center value of the Gaussian
    sigma: float
        The one sigma width of the Gaussian

    Returns
    -------
    y:float or np.array
        The dependent value for a given independent value (x). 
    """
    return a*np.exp(-(x-x0)**2/(2*sigma**2))


def parameter_reconstruct(z,fcs,cur_vdf):
    """
    Parameters
    ------------
    z: list
        A list of input parameters used in the 2D velocity reconstruction. 
        The list must contain the following variables in the following order:
        vx,vy,vz,wper,wpar,den where vx is the x-component of the velocity in
        GSE coordinates, vy in the y-component of the velocity in GSE coordinates
        vz is the z-component of the velocity in GSE coordinates,
        wper is the thermal speed in the perpendicular direction,
        wpar is the thermal speed in the parallel direction, and
        den is the number density of protons.
    fcs: dictionary
        A dictionary of Faraday cups created by mff.create_multi_fc()
    cur_vdf: dictionary
        Dictionary of the velocity distribution created by make_discrete_vdf()

    Returns
    -------
    powell: scipy.optimize minimum object
    """
        
    powell = minimize(gauss_2d_reconstruct,p_guess, args=(fcs,dis_vdf_bad),method='Powell',options={'maxiter':10,'maxfev': 10})

    return powell



def gauss_2d_reconstruct(z,fcs,cur_vdf):
    """
    Function that reconstructs the 2D velocity distribution in the V parallel
    and V perp reference frame by assuming a 2D Gaussian.
    
    Parameters
    ------------
    z: list
        A list of input parameters used in the 2D velocity reconstruction. 
        The list must contain the following variables in the following order:
        vx,vy,vz,wper,wpar,den where vx is the x-component of the velocity in
        GSE coordinates, vy in the y-component of the velocity in GSE coordinates
        vz is the z-component of the velocity in GSE coordinates,
        wper is the thermal speed in the perpendicular direction,
        wpar is the thermal speed in the parallel direction, and
        den is the number density of protons.
    fcs: dictionary
        A dictionary of Faraday cups created by mff.create_multi_fc()
    cur_vdf: dictionary
        Dictionary of the velocity distribution created by make_discrete_vdf()
        
    Returns
    -------
    y:float or np.array
        The measured current of all Faraday Cups
    """
    vx = cur_vdf['u_gse'][0]
    vy = cur_vdf['u_gse'][1]
    vz = cur_vdf['u_gse'][2]
    
    vx,vy,vz,wper, wpar, den = z
    
    #                   Vx,Vy,Vz,Wper,Wpar, Np
    pls_par = np.array([vx,vy,vz,wper, wpar, den]) 
    
  #Get static variables from cur_vdf to add to creation of new guess VDF
    vel_clip = cur_vdf['pgrid'].max()
    pres     = np.mean(np.diff(cur_vdf['pgrid'][:,0]))
    qres     = np.mean(np.diff(cur_vdf['qgrid'][0,:]))
    
    #Create new VDF guess based on input parameters
    dis_vdf = mdv.make_discrete_vdf(pls_par,cur_vdf['b_gse'],pres=pres,qres=qres,clip=vel_clip) 
    
    looper = []
    #loop over all fc in fcs to populate with new VDF guess
    for i,key in enumerate(fcs.keys()):
        #add variation and store which faraday cup you are working with using key
        #Updated with varying integration sampling function 2018/10/12 J. Prchlik
        inpt_x = fcs[key]['x_meas'].copy()
        g_vdf  = dis_vdf.copy()
        peak   =  fcs[key]['peak'].copy()
        looper.append((inpt_x,g_vdf,samp,key))
        
    #process in parallel
    nproc = 8
    pool = Pool(processes=nproc)
    dis_cur = pool.map(proc_wrap,looper)
    pool.close()
    pool.join()       
    
    
    #break into index value in looper and the 1D current distribution
    index   = np.array(zip(*dis_cur)[1])
    dis_cur = np.array(zip(*dis_cur)[0])


    #get sum squared best fit
    tot_err = np.zeros(dis_cur.shape[0])
    tot_int = np.zeros(dis_cur.shape[0])
    #Get error in each faraday cup
    for j,i in enumerate(index):
        tot_err[j] = np.sum((dis_cur[j,:] - fcs[i]['rea_cur'])**2)
        tot_int[j] = np.sum((fcs[i]['rea_cur'])**2)

    #total error for all fc
    #Remove values with no flux from guess fitting
    tot_err[tot_int < 1e-25] = 0
    fcs_err = np.sum(tot_err**2) /np.sum(tot_int**2)

    return fcs_err


def gauss_2d_reconstruct(z,fcs,cur_vdf,add_ring=False,nproc=8):
    """
    Function that reconstructs the 2D velocity distribution in the V parallel
    and V perp reference frame by assuming a 2D Gaussian.
    
    Parameters
    ------------
    z: list
        A list of input parameters used in the 2D velocity reconstruction. 
        The list must contain the following variables in the following order:
        vx,vy,vz,wper,wpar,den,q_r,p_r,wper_r,wpar_r,peak_r where vx is the x-component of the velocity in
        GSE coordinates, vy in the y-component of the velocity in GSE coordinates
        vz is the z-component of the velocity in GSE coordinates, q_r and p_r are
        the locations of a ring respectively in Vperp and Vpar, wper_r and wpar_r
        are the perpendicular and parallel thermal velocity widths, peak_r is the 
        maximum value of the added ring.
    fcs: dictionary
        A dictionary of Faraday cups created by mff.create_multi_fc()
    cur_vdf: dictionary
        Dictionary of the velocity distribution created by make_discrete_vdf()
    add_ring: boolean, optional
        Add ring to model velocity distribution (Default = False).
    nproc: int, optional
        Number of processors to use when computing theoretical 1D FC measurements
        from the theoretical 2D velocity distribution in Vpar and Vper (Default = 8).
        
    Returns
    -------
    y:float or np.array
        The measured current of all Faraday Cups
    """

    #parameters to adjust in model if there is a ring and a core
    if add_ring:
        vx,vy,vz,wper, wpar, den,q_r,p_r,wper_r,wpar_r,peak_r = z
    #parameters to adjust in model if there is just a core
    else:
        vx,vy,vz,wper, wpar, den = z
    
    #                   Vx,Vy,Vz,Wper,Wpar, Np
    pls_par = np.array([vx,vy,vz,wper, wpar, den]) 
    
    #Get static variables from cur_vdf to add to creation of new guess VDF
    vel_clip = cur_vdf['pgrid'].max()
    pres     = np.mean(np.diff(cur_vdf['pgrid'][:,0]))
    qres     = np.mean(np.diff(cur_vdf['qgrid'][0,:]))
    
    #Create new VDF guess based on input parameters
    dis_vdf = mdv.make_discrete_vdf(pls_par,cur_vdf['b_gse'],pres=pres,qres=qres,clip=vel_clip) 
    
    #Add ring to fit
    if add_ring:
        #Add a positive Gaussian Kernal to "Measured" VDF
        dis_vdf['vdf'] += peak_r*np.exp(- ((dis_vdf['pgrid']-(p_r))/wpar_r)**2. - ((dis_vdf['qgrid']-(q_r))/wper_r)**2.)
        #update the interpolator function
        dis_vdf['vdf_func'] =  RectBivariateSpline(dis_vdf['pgrid'][:,0],dis_vdf['qgrid'][0,:],dis_vdf['vdf'])
    
    looper = []
    #loop over all fc in fcs to populate with new VDF guess
    for i,key in enumerate(fcs.keys()):
        #add variation and store which faraday cup you are working with using key
        #Updated with varying integration sampling function 2018/10/12 J. Prchlik
        inpt_x = fcs[key]['x_meas'].copy()
        g_vdf  = dis_vdf.copy()
        peak   =  fcs[key]['peak'].copy()
        looper.append((inpt_x,g_vdf,samp,key))
        
    #process in parallel
    pool = Pool(processes=nproc)
    dis_cur = pool.map(proc_wrap,looper)
    pool.close()
    pool.join()       
    
    
    #break into index value in looper and the 1D current distribution
    index   = np.array(zip(*dis_cur)[1])
    dis_cur = np.array(zip(*dis_cur)[0])


    #get sum squared best fit
    tot_err = np.zeros(dis_cur.shape[0])
    tot_int = np.zeros(dis_cur.shape[0])
    #Get error in each faraday cup
    for j,i in enumerate(index):
        tot_err[j] = np.sum((dis_cur[j,:] - fcs[i]['rea_cur'])**2)
        tot_int[j] = np.sum((fcs[i]['rea_cur'])**2)

    #print(tot_err)
    #total error for all fc
    #tot_err[tot_int < 1e-25] = 0
    #fcs_err = np.median(tot_err)
    fcs_err = np.sqrt(np.sum(tot_err**2) /np.sum(tot_int**2))
    #Remove really bad values from guess fitting

        
    return fcs_err


def ring_vdf(cur_vdf,vx,vy,vz,wper,wpar,den,q_r,p_r,wper_r,wpar_r,peak_r):
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



