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
    pool = Pool(processes=8)
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


#start time for comp
time_start = time.time()

#set up plasma parameters
pls_par = np.array([-380., -30., 30., 60., 20., 5.]) 
mag_par = np.array([np.cos(np.radians(75.)), np.sin(np.radians(75.)), 0.]) 
#mag_par = np.array([1., 0., 0.]) 

#number of sample for integration
samp = 3e1
#make a discrete VDF
dis_vdf = mdv.make_discrete_vdf(pls_par,mag_par,pres=1.00,qres=1.00,clip=4.)


#spacecraft measurements of phi and theta
phi = -15.
theta = 15.

#report some measurements
grid_v = np.arange(300,600,20)
x_meas = mdv.make_fc_meas(dis_vdf,fc_spd=grid_v,fc_phi=phi,fc_theta=theta)


#calculate the "real measured reduced response function"
rea_cur = mdv.arb_p_response(x_meas,dis_vdf,samp)

#get effeictive area of wind and other coversion parameters
waeff = 3.8e6 #cm^3/km
q0    = 1.6021892e-7 # picocoulombs
dv    = np.diff(grid_v)
dv    = np.concatenate([dv,[dv[-1]]])
cont  = 1.e12/(waeff*q0*dv*grid_v)

#calculate the Gaussian fit of the response
popt, pcov = curve_fit(gaus,grid_v,rea_cur*cont,p0=[1e-9,450.,30.])


#get the parameters from the fit
u = popt[1] #speed in km/s
w = np.abs(popt[2]*np.sqrt(2.)) #thermal speed in km/s
n = popt[0]*w*np.sqrt(np.pi) #density in cc

#uncertainty in parameters from fit
du = pcov[1,1]
dw = pcov[2,2]
dn = np.sqrt(np.pi*((w*pcov[0,0])**2 + (dw*n)**2))

#get velocity componets
vx,vy,vz = mdv.convert_fc_gse(u,np.radians(phi),np.radians(theta))
wx,wy,wz = mdv.convert_fc_gse(w,np.radians(phi),np.radians(theta))


#number of processes to use per calculation
nproc = 8


#make a discrete VDF with the incorrect parameters but the same grid
pls_par_bad = np.array([vx, vy, vz,w,np.sqrt(wy**2+wz**2.),n])
dis_vdf_bad = mdv.make_discrete_vdf(pls_par_bad,mag_par,pres=1.00,qres=1.00,clip=4.)


#Probability of selecting a gaussian that subtracts or adds to the vdf
n_p_prob = np.array([0.5,0.5])


start_loop = time.time()
#takes about 1000 iterations to converge (~30 minutes), but converged to the wrong solution, mostly overestimated the peak
for i in range(10000):
    #get a new vdf and return if it is the best fit
    dis_vdf_bad,tot_error,dis_cur = create_random_vdf(dis_vdf_bad,nproc,n_p_prob)
    print('Total error for iteration {0:1d} is {1:4.3e}'.format(i,float(tot_error)))
    #get probability modification for addition and subtraction
    per_err  = np.sum(-(dis_cur-rea_cur)/rea_cur)/100.
    n_p_prob = np.array([0.5,0.5])#+per_err*np.array([-1.,1.])
    #do no allow values greater than 1 or less than 0
    n_p_prob[n_p_prob < 0 ] = 0
    n_p_prob[n_p_prob > 1 ] = 1
    print(n_p_prob,per_err)
    
end_loop = time.time()
print('Loop time {0:1.1f}'.format(end_loop-start_loop))




#for single run tests
#dis_cur = mdv.arb_p_response(x_meas,dis_vdf,samp)
#dis_cur = [dis_cur]
#code run time
time_elapsed = (time.time() - time_start)

#Time elapsed pring
print('Time to run for {0:1.0f} samples is {1:5.1f}s'.format(samp,time_elapsed))

col_cur = mdv.p_bimax_response(x_meas,np.concatenate([pls_par,mag_par]))
idl_cur = np.array([4.5560070e-11,7.7672213e-11,1.1308081e-10,1.4061885e-10,1.4938468e-10,1.3559443e-10,1.0517354e-10,6.9718318e-11
                   ,3.9500478e-11,1.9129573e-11,7.9192133e-12,2.8025510e-12,8.4788266e-13,2.1930091e-13,4.8492592e-14,9.1673766e-15
                   ,1.4816679e-15,2.0473478e-16,2.4185986e-17,2.4426542e-18])


   
#create figure to plot
fig, ax = plt.subplots()
ax.plot(grid_v,col_cur,'--r',label='Cold',linewidth=3)

#Get the average current and plot
ave_cur = np.array(dis_cur).mean(axis=0)
std_cur = np.array(dis_cur[1::5]).std(axis=0)/np.sqrt(2)

#print percent error
print('Cold Case Current to Integrated Current')
#print(abs(col_cur-ave_cur)/ave_cur)

#ax.plot(grid_v,ave_cur,label='Midpoint Intgration',linewidth=3,color='black')
#ax.errorbar(grid_v,ave_cur,yerr=std_cur,label=None,color='black',linewidth=3)

ax.plot(grid_v,dis_cur.ravel(),label='best fit',color='black',linewidth=3)
ax.plot(grid_v,rea_cur.ravel(),'-.b',label='Input',linewidth=3)

fancy_plot(ax)
#ax.set_yscale('log')
ax.set_xlabel('Speed [km/s]')
ax.set_ylabel('Current')
ax.legend(loc='best',frameon=False)
plt.show()


#mdv.vdf_calc(pls_par[0],pls_par[1],pls_par[2],hold_bfc=dis_vdf['b_gse'],hold_ufc=dis_vdf['u_gse'],hold_qgrid=dis_vdf['qgrid'],hold_pgrid=dis_vdf['pgrid'],hold_vdf=dis_vdf['vdf'],tol=5)