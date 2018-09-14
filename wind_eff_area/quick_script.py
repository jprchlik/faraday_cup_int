import make_discrete_vdf as mdv
import numpy as np
import matplotlib.pyplot as plt
from fancy_plot import fancy_plot
from multiprocessing import Pool
import time
from scipy.optimize import curve_fit



def gaus(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def proc_wrap(arg):
    return [mdv.arb_p_response(*arg[:-1]),arg[-1]]

def create_random_vdf(dis_vdf_guess,nproc,n_p_prob):
    global rea_cur,x_meas
    looper = [(x_meas,dis_vdf_guess,samp,0)]
    for i in range(1,nproc*2+1):
        #add variation
        dis_vdf_bad = mdv.make_discrete_vdf_random(dis_vdf_guess,sc_range=0.1)
        looper.append((x_meas,dis_vdf_bad,samp,i))

    #process in parallel
    pool = Pool(processes=nproc)
    dis_cur = pool.map(proc_wrap,looper)
    pool.close()
    pool.join()

    #break into index value in looper and the 1D current distribution
    index   = np.array(zip(*dis_cur)[1])
    dis_cur = np.array(zip(*dis_cur)[0])

    #get sum squared best fit
    tot_err = np.sqrt(np.sum((dis_cur - rea_cur)**2.,axis=1))
    best_fit = np.partition(tot_err,0)[0]
    best, = np.where(tot_err == np.min(tot_err))

    #get the best fit dictionary from looper list
    best_ind = int(index[best[0]])

    #return best dictionary and total error
    return looper[best_ind][1],tot_err[best[0]],dis_cur[best[0]]

def create_random_vdf_multi_fc(fcs,proc,cur_err,dis_vdf_guess,cont,samp=30,verbose=False):
    """
    Parameters
    -----------
    proc : int
        Number of processors to run on
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


    sc_range = 0.1
    #on the first loop do not update
    if cur_err > 1e30:
        dis_vdf_bad = dis_vdf_guess

    else:
        #Add Guassian 2D perturbation
        #create Vper and Vpar VDF with pertabation from first key because VDF is the same for all FC
        dis_vdf_bad = mdv.make_discrete_vdf_random(dis_vdf_guess,sc_range=sc_range,
                         p_sig=0.1*np.ptp(dis_vdf_guess['pgrid']),q_sig=0.1*np.ptp(dis_vdf_guess['qgrid']))

    #loop over all fc in fcs to populate with new VDF guess
    for i,key in enumerate(fcs.keys()):
        #add variation and store which faraday cup you are working with using key
        inpt_x = fcs[key]['x_meas'].copy()
        g_vdf  = dis_vdf_bad.copy()
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
    


#start time for comp
time_start = time.time()

#set up plasma parameters
#                    Vx  ,  Vy,  Vz ,Wper,Wpar, Np
pls_par = np.array([-380., -30., 30., 20., 40., 5.]) 
pls_par = np.array([-580., 100., -10., 10., 80., 15.]) 
#pls_par = np.array([-480., -100., -80., 20., 50., 25.]) 
#pls_par = np.array([-380., -100., 50., 30., 10., 50.]) 
#pls_par = np.array([-880., 100.,-150., 30., 10., 5.]) 
mag_par = np.array([np.cos(np.radians(75.))*np.cos(.1), np.sin(np.radians(75.))*np.cos(.1), np.sin(.1)]) 
#mag_par = np.array([1., 0., 0.]) 

#number of sample for integration
samp = 3e1
#make a discrete VDF
dis_vdf = mdv.make_discrete_vdf(pls_par,mag_par,pres=1.00,qres=1.00,clip=4.)



#get random angles of faraday cup in phi and theta
#number of fc cups
ncup = 20
#set random seed for FC angles
np.random.seed(1107)

#Get two uniform number between -30 and 30
phis = np.random.uniform(-20.,20.,size=ncup)
thetas = np.random.uniform(-20.,20.,size=ncup)


#spacecraft measurements of phi and theta to show when plotting
phi = float(phis[-1])
theta = float(thetas[-1])


#report some measurements
#veloity grid
grid_v = np.arange(300,600,20)
v_mag = np.sqrt(np.sum(pls_par**2))
grid_v = np.arange(v_mag-150,v_mag+150,20)
#get effective area of wind and other coversion parameters
waeff = 3.8e6 #cm^3/km
q0    = 1.6021892e-7 # picocoulombs
dv    = np.diff(grid_v)
dv    = np.concatenate([dv,[dv[-1]]])
cont  = 1.e12/(waeff*q0*dv*grid_v)
    # (picocoulomb * s-1)*km-2*s2 * km * cm-3 / picocoulomb
    #                              = cm-3 * s * km-1 =
    #                              particles per cm-3 per km/s
#number of processes to use per calculation
nproc = 8

#array that store all the fit parameters 
big_arr = []

#calculate the "real measured reduced response function" for all fc cups
fcs = {}
for k,(phi,theta) in enumerate(zip(phis,thetas)):
    #calculate x_meas array
    x_meas = mdv.make_fc_meas(dis_vdf,fc_spd=grid_v,fc_phi=phi,fc_theta=theta)
    #compute the observed current in the instrument
    rea_cur = mdv.arb_p_response(x_meas,dis_vdf,samp)

    #create key for input fc
    key = 'fc_{0:1d}'.format(k)
    fcs[key] = {}
    
    #populate key with measurements and parameter 
    fcs[key]['x_meas']  = x_meas
    fcs[key]['rea_cur'] = rea_cur


    #calculate the Gaussian fit of the response
    popt, pcov = curve_fit(gaus,grid_v,rea_cur*cont,p0=[np.nanmax(rea_cur*cont),np.mean(grid_v),np.sqrt(2.)*2*dv[0]],sigma=1./(rea_cur/rea_cur.min()))
    
    
    #Switched to computing the average
    #####get the parameters from the fit
    u = popt[1] #speed in km/s
    w = np.abs(popt[2]*np.sqrt(2.)) #thermal speed in km/s
    n = popt[0]*w*np.sqrt(np.pi) #density in cc
    ####
    #####uncertainty in parameters from fit
    ####du = pcov[1,1]
    ####dw = pcov[2,2]
    ####dn = np.sqrt(np.pi*((w*pcov[0,0])**2 + (dw*n)**2))

 

    #####get velocity componets
    #####assuming magnetic field normal with respect to the spacecraft
    ####vx,vy,vz = mdv.convert_fc_gse(u,np.radians(phi),np.radians(theta))
    ####wx,wy,wz = mdv.convert_fc_gse(w,np.radians(phi),np.radians(theta))

    #####calculate V_par and V_per
    ####wpar = np.abs(np.dot(np.array([wx,wy,wz]),mag_par))
    ####wper = np.sqrt(np.sum(np.array([wx,wy,wz])**2.)-wpar**2.)

    #Add fit parameters with velocity guesses
    big_arr.append([u,w,n,phi,theta])

#convert big_arr intop numpy array
big_arr = np.array(big_arr)
#get speed solution per observatory
v_angl = big_arr[:,0]
#thermal speed in GSE
w_angl = big_arr[:,1]
#get the density to compute the magnitude
n_angl = big_arr[:,2]

#density components in  GSE
rot_mat = mdv.euler_angles(np.arctan2(mag_par[1],mag_par[0]),np.arcsin(mag_par[2]),psi_ang=0.)


#get v_gse solution (Produces the same solution as the Wind spacecraft solution)
#in /crater/observatories/wind/code/dvapbimax/sub_bimax_moments.pro
vx,vy,vz =  mdv.compute_gse_from_fit(np.radians(phis),np.radians(thetas),v_angl) #np.dot(np.dot(np.dot(v_svdc.T,wp_svdc),u_svdc.T),v_angl)
wv =  mdv.compute_gse_from_fit(np.radians(phis),np.radians(thetas),w_angl) #np.dot(np.dot(np.dot(v_svdc.T,wp_svdc),u_svdc.T),v_angl)
nv =  np.mean(n_angl)

#compute density
n = np.sqrt(np.sum(nv**2))

#Transform wv gse to magnetic field frame
wmag = np.array(rot_mat.dot(wv))[0]

#compute Wpar and Wper
#Use Theta and phi angles between Vgse and Bnorm to get the new vectors
#we = np.abs(wv.dot(mag_par)) #Wper
#wa = np.abs(wv.dot(mag_par)) #Wpar
#we = np.sqrt(np.sum(wv**2)-wa**2)/2. #Wper (factor of 2 comes from assumption of half in each V_perp comp
wa = np.abs(wmag[0])
we = np.sqrt(np.sum(wmag[1:]**2))



#make a discrete VDF with the incorrect parameters but the same grid
pls_par_bad = np.array([vx, vy, vz,we,wa,n])

#Try and update the input Guess for the Wper Wpar parameters before X^2 min
wa, we =  mdv.find_best_vths(wa,we,pls_par_bad,mag_par,fcs[key]['rea_cur'],fcs[key]['x_meas'])

#update with new Wper and Wpar paraemters
pls_par_bad = np.array([vx, vy, vz,we,wa,n])

dis_vdf_bad = mdv.make_discrete_vdf(pls_par_bad,mag_par,pres=1.00,qres=1.00,clip=5.)
#store the initial bad guess 
dis_vdf_bad_guess = dis_vdf_bad
#calculate x_meas array for the average with multiple observatories
#put in the same rest frame as the modelled input 2018/09/13 J. Prchlik
x_meas_eff = mdv.make_fc_meas(dis_vdf,fc_spd=grid_v,fc_phi=phi,fc_theta=theta)
dis_cur_bad_guess = mdv.arb_p_response(fcs[key]['x_meas'],dis_vdf_bad_guess,samp)




#######Give info on best fit versus real solution######
print(pls_par)
print(pls_par_bad)

#store the guess 2D measurment by the FC in all fcs dictionaries
#for key in fcs.keys():
#    fcs[key]['gue_vdf'] = dis_vdf_bad
#
#Probability of selecting a gaussian that subtracts or adds to the vdf
n_p_prob = np.array([0.5,0.5])

#Inital bad sum squared error value
tot_err = 1e31 #a very large number

start_loop = time.time()
#takes about 1000 iterations to converge (~30 minutes), but converged to the wrong solution, mostly overestimated the peak
#removed to test improving fit
for i in range(10000):
    #get a new vdf and return if it is the best fit
    #dis_vdf_bad,tot_error,dis_cur = create_random_vdf(dis_vdf_bad,nproc,n_p_prob)
    fcs,tot_err,dis_vdf_bad = create_random_vdf_multi_fc(fcs,nproc,tot_err,dis_vdf_bad,cont)

    print('Total error for iteration {0:1d} is {1:4.3e}'.format(i,float(tot_err)))
    #get probability modification for addition and subtraction
    #per_err  = np.sum(-(dis_cur-rea_cur)/rea_cur)/100.
    #n_p_prob = np.array([0.5,0.5])#+per_err*np.array([-1.,1.])
    ##do no allow values greater than 1 or less than 0
    #n_p_prob[n_p_prob < 0 ] = 0
    #n_p_prob[n_p_prob > 1 ] = 1
    #print(n_p_prob,per_err)
    
end_loop = time.time()
print('Loop time {0:1.1f}'.format(end_loop-start_loop))




#for single run tests
#dis_cur = mdv.arb_p_response(x_meas,dis_vdf,samp)
#dis_cur = [dis_cur]
#code run time
time_elapsed = (time.time() - time_start)

#Time elapsed print
print('Time to run for {0:1.0f} samples is {1:5.1f}s'.format(samp,time_elapsed))

col_cur = mdv.p_bimax_response(fcs[key]['x_meas'],np.concatenate([pls_par,mag_par]))
idl_cur = np.array([4.5560070e-11,7.7672213e-11,1.1308081e-10,1.4061885e-10,1.4938468e-10,1.3559443e-10,1.0517354e-10,6.9718318e-11
                   ,3.9500478e-11,1.9129573e-11,7.9192133e-12,2.8025510e-12,8.4788266e-13,2.1930091e-13,4.8492592e-14,9.1673766e-15
                   ,1.4816679e-15,2.0473478e-16,2.4185986e-17,2.4426542e-18])


   
#create figure to plot
fig, ax = plt.subplots()
#ax.plot(grid_v,col_cur*cont,'--r',label='Cold',linewidth=3)

#Get the average current and plot
#removed 2018/09/11 J. Prchlik No longer calculating an average from many integrations
#ave_cur = np.array(dis_cur).mean(axis=0)
#std_cur = np.array(dis_cur[1::5]).std(axis=0)/np.sqrt(2)

#print percent error
print('Cold Case Current to Integrated Current')
#print(abs(col_cur-ave_cur)/ave_cur)

#ax.plot(grid_v,ave_cur,label='Midpoint Intgration',linewidth=3,color='black')
#ax.errorbar(grid_v,ave_cur,yerr=std_cur,label=None,color='black',linewidth=3)

#removed to test improving Gaussian fit
ax.plot(grid_v,fcs[key]['dis_cur'].ravel()*cont,label='Best MC',color='black',linewidth=3)
ax.plot(grid_v,fcs[key]['rea_cur'].ravel()*cont,'-.b',label='Input',linewidth=3)
#ax.plot(grid_v,rea_cur.ravel()*cont,'-.b',label='Input',linewidth=3)
ax.plot(grid_v,dis_cur_bad_guess.ravel()*cont,':',color='purple',label='Init. Guess',linewidth=3)
#ax.plot(grid_v, gaus(grid_v,*popt),'--',marker='o',label='Gauss Fit',linewidth=3)


fancy_plot(ax)
#ax.set_yscale('log')
ax.set_xlabel('Speed [km/s]')
ax.set_ylabel('Current')
ax.legend(loc='best',frameon=False)
#plt.show()

#Best Fit MC VDF
mdv.plot_vdf(dis_vdf_bad)

#"REAL" OBSERVATION
mdv.plot_vdf(dis_vdf)

#mdv.vdf_calc(pls_par[0],pls_par[1],pls_par[2],hold_bfc=dis_vdf['b_gse'],hold_ufc=dis_vdf['u_gse'],hold_qgrid=dis_vdf['qgrid'],hold_pgrid=dis_vdf['pgrid'],hold_vdf=dis_vdf['vdf'],tol=5)