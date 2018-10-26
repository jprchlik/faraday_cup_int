import make_discrete_vdf as mdv
from scipy.interpolate import RectBivariateSpline
import numpy as np
from multiprocessing import Pool
from scipy.optimize import curve_fit
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


def create_random_vdf_multi_fc(fcs,nproc,cur_err,dis_vdf_guess,cont,pred_grid,kernel,improved=False,samp=3.,verbose=False,ip=0.,iq=0.,n_p_prob=[0.5,0.5],sc_range=0.1):
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
    cont: np.array
        Array of floats to convert shape into current
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
    
def create_grid_vals_multi_fc(fcs,proc,cur_err,dis_vdf_guess,cont,verbose=False):
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
    cont: np.array
        Array of floats to convert shape into current
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
