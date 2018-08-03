import make_discrete_vdf as mdv
import numpy as np
import matplotlib.pyplot as plt
from fancy_plot import fancy_plot
from multiprocessing import Pool
import time



def proc_wrap(arg):
    return mdv.arb_p_response(*arg)

#start time for comp
time_start = time.time()

#set up plasma parameters
pls_par = np.array([-380., -30., 30., 60., 20., 5.]) 
mag_par = np.array([np.cos(np.radians(75.)), np.sin(np.radians(75.)), 0.]) 
#mag_par = np.array([1., 0., 0.]) 

#number of sample for integration
samp = 7e1
#make a discrete VDF
dis_vdf = mdv.make_discrete_vdf(pls_par,mag_par,pres=1.00,qres=1.00,clip=4.)

#report some measurements
grid_v = np.arange(300,600,20)
x_meas = mdv.make_fc_meas(dis_vdf,fc_spd=grid_v,fc_phi=-15.,fc_theta=15.)

#interatable to pass to pool

#Get the integral values in parallel
looper = np.array((x_meas,dis_vdf,samp)*8).reshape(8,3)



pool = Pool(processes=8)
dis_cur = pool.map(proc_wrap,looper)
pool.close()
pool.join()

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
print(abs(col_cur-ave_cur)/ave_cur)

ax.plot(grid_v,ave_cur,label='Midpoint Intgration',linewidth=3,color='black')
ax.errorbar(grid_v,ave_cur,yerr=std_cur,label=None,color='black',linewidth=3)

for i in dis_cur:
   ax.plot(grid_v,i)

fancy_plot(ax)
#ax.set_yscale('log')
ax.set_xlabel('Speed [km/s]')
ax.set_ylabel('Current')
ax.legend(loc='best',frameon=False)
plt.show()


#mdv.vdf_calc(pls_par[0],pls_par[1],pls_par[2],hold_bfc=dis_vdf['b_gse'],hold_ufc=dis_vdf['u_gse'],hold_qgrid=dis_vdf['qgrid'],hold_pgrid=dis_vdf['pgrid'],hold_vdf=dis_vdf['vdf'],tol=5)