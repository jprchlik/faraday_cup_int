import monte_carlo_int as mci
import numpy as np
from scipy.integrate import tplquad
import time
import mid_point_loop as mpl


def test_midpoint():
     samp = 200
     args = []
     x_lo = 0
     x_hi = 40.5
     y_lo = lambda x : x*0
     y_hi = lambda x : 2.0*np.pi+x*0
     z_lo =  lambda x,y : x*y*0
     z_hi =  lambda x,y : x*0*y+np.pi

     def dA_Sphere(theta,phi):
       return  np.sin(phi)
     def dV_Sphere( theta,phi,r):
       return r * r * dA_Sphere(phi, theta)

     val, err = tplquad(dV_Sphere, 0.0, 3.5, y_lo,
        y_hi, z_lo, z_hi)

     

     #time_mp_s = time.time()
     #int_runu,arr = mci.mp_trip(dV_Sphere,x_lo,x_hi,y_lo, y_hi, z_lo, z_hi ,args=args,samp=samp)
     #time_mp_e = time.time()
     #print('Time to run MP integration using list comp {0:2.1f}'.format(time_mp_e-time_mp_s))

     #time_mp_s = time.time()
     #int_runv = mci.mp_trip2(dV_Sphere,x_lo,x_hi,y_lo, y_hi, z_lo, z_hi ,args=args,samp=samp)
     #time_mp_e = time.time()
     #print('Time to run MP integration using nested funcitons {0:2.1f}'.format(time_mp_e-time_mp_s))
     #int_runw,x_grid,y_grid,z_grid = mci.mp_trip3(dV_Sphere,x_lo,x_hi,y_lo, y_hi, z_lo, z_hi ,args=args,samp=samp)


     #using cython array creation
     time_mp_s = time.time()
     int_runu = mci.mp_trip_cython(dV_Sphere,x_lo,x_hi,y_lo, y_hi, z_lo, z_hi ,args=args,samp=samp)
     time_mp_e = time.time()
     print('Time to run MP integration using cython {0:2.1f}'.format(time_mp_e-time_mp_s))

     time_tq_s = time.time()
     int_test,error = tplquad(dV_Sphere,x_lo,x_hi,y_lo, y_hi, z_lo, z_hi , epsabs=1.e-4, epsrel=1.e-4,args=args)
     time_tq_e = time.time()
     print('Time to run quad intregration python integration {0:2.1f}'.format(time_tq_e-time_tq_s))

     int_real = 4./3.*np.pi*x_hi**3

    
     print(int_test,int_real,int_runu)
     print(np.abs(int_runu-int_real)/np.abs(int_real))
     print(error)
     #return int_runu,arr
     assert np.abs(int_runu-int_real)/np.abs(int_real) < 1.e-4
     assert np.abs(int_test-int_real)/np.abs(int_real) < 1.e-4


if __name__ == '__main__':
    arr = test_midpoint()