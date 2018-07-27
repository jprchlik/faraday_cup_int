import numpy as np

def mc_trip(int_3d,z_lo,z_hi,x_lo,x_hi,y_lo,y_hi,args=(),n=1000,nproc=1):
    """
    Monte carlo integrator for solar wind on generic Faraday Cup
    """

    #done so multiprocessing with not give the same seed 
    np.random.seed()
    #Get random Velocity values between the high and low points
    vz = np.random.uniform(z_lo,z_hi,n)
    vx = np.random.uniform(x_lo(z_hi),x_hi(z_hi),n)
    vy = np.random.uniform(x_lo(z_hi),x_hi(z_hi),n) #assuming vx,vy are symmetric



    #mean f value inside FC
    f_mean = 0
    #number of points inside area
    n_cnts = 0

    #Find points inside area
    #Vz in FC
    z_lo_g  = vz > z_lo
    z_hi_g  = vz < z_hi

    #Vx in FC
    x_lo_g  = vx > x_lo(vz)
    x_hi_g  = vx < x_hi(vz)

    #Vy in FC
    y_lo_g  = vy > y_lo(vz,vx)
    y_hi_g  = vy < y_hi(vz,vx)

    #print(z_lo_g,z_hi_g)
    #print(x_lo_g,x_hi_g)
    #points inside bound
    inside, = np.where((z_lo_g) & (z_hi_g) & (x_lo_g) & (x_hi_g) & (y_lo_g) & (y_hi_g))

    #get total current inside area
    total_c = np.sum(int_3d(vz[inside],vx[inside],vy[inside],*args))
    
    #get average under the curve
    averg_c = total_c/inside.size

    #get area under the FC cup curve
    area_mc = inside.size/(float(n**3))

    return area_mc*total_c