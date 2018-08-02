import numpy as np

def mc_trip(int_3d,z_lo,z_hi,x_lo,x_hi,y_lo,y_hi,args=(),samp=1e4):
    """
    Monte carlo integrator for solar wind on generic Faraday Cup
    """

    #number of samples
    n = int(samp)
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
    area_mc = inside.size/(float(n))*(z_hi-z_lo)*(x_hi(z_hi)-(x_lo(z_hi)))**2
    #area_mc = 1

    return area_mc*averg_c

def mp_trip(int_3d,z_lo,z_hi,x_lo,x_hi,y_lo,y_hi,args=(),samp=1e4):
    """
    Midpoint integration for generic Faraday Cup
    """

    #midpoint_triple2(int_3d,z_lo,z_hi,x_lo(z_hi),x_hi(z_hi),y)
    #width of mid-point
    h = (z_hi-z_lo)/samp


    area = 0.
    z = z_lo
    #loop over all z values
    while z <= z_hi: 
        #get x min and max
        x_min,x_max = x_lo(z),x_hi(z)
        x = x_min
        #loop over all x values
        while x <= x_max:
             print(x)
             y_min,y_max = y_lo(z,x),y_hi(z,x)
             y,h_y = np.linspace(y_min,y_max,samp,endpoint=True,retstep=True)
             area_cal = int_3d(z,x,y,*args)
             area += np.sum(area_cal*h_y)
             #increment x
             x += h
 
        #increment z
        z += h    

    #return the cumlative area
    return area*h**2

def midpoint(f, a, b, n):
    h = float(b-a)/n
    result = 0
    for i in range(n):
        result += f((a + h/2.0) + i*h)
    result *= h
    return result

def mp_trip2(int_3d,z_lo,z_hi,x_lo,x_hi,y_lo,y_hi,args=(),samp=1e4):
    samp = int(samp)
    print(samp)
    def p(z,x):
        print(z)
        return midpoint(lambda y: int_3d(z,x,y,*args),y_lo(z,x),y_hi(z,x),samp)

    def q(z):
        return midpoint(lambda x: p(z,x),x_lo(z),x_hi(z),samp)

    return midpoint(q,z_lo,z_hi,samp)

def mp_trip3(int_3d,z_lo,z_hi,x_lo,x_hi,y_lo,y_hi,args=(),samp=1e4):
    """
    Midpoint integration for generic Faraday Cup
    """

    #midpoint_triple2(int_3d,z_lo,z_hi,x_lo(z_hi),x_hi(z_hi),y)
    #width of mid-point
    z,h_z = np.linspace(z_lo,z_hi,samp,endpoint=True,retstep=True)

    #get xvalues
    x_min,x_max = x_lo(z),x_hi(z)
    x_diff = x_max-x_min
    x_rang = np.arange(samp)
    x_resl = x_diff/samp 
    h_x = np.outer(x_resl,x_rang)
    x      = np.outer(x_min,np.ones(h_x.shape[1]))+h_x  

    #update z shape
    z       = np.outer(z,np.ones(x.shape[1]))
    h_z     = np.outer(h_z+np.zeros(int(samp)),np.ones(x.shape[1]))

    #get yvalues
    y_min, y_max = y_lo(z,x),y_hi(z,x)
    y_diff       = y_max-y_min
    y_rang       = np.arange(samp)
    y_resl       = y_diff/samp
    h_y          = np.outer(y_resl,y_rang).reshape(z.shape[0],x.shape[1],y_rang.size)
    y            = np.outer(y_min,np.ones(h_y.shape[1])).reshape(z.shape[0],x.shape[1],y_rang.size)+h_y

    #update z shape
    z            = np.outer(z,np.ones(y.shape[2])).reshape(z.shape[0],x.shape[1],y_rang.size)  
    h_z          = np.outer(h_z,np.ones(y.shape[2])).reshape(z.shape[0],x.shape[1],y_rang.size)  
    #update x shape
    x            = np.outer(x,np.ones(y.shape[2])).reshape(z.shape[0],x.shape[1],y_rang.size)  
    h_x          = np.outer(h_x,np.ones(y.shape[2])).reshape(z.shape[0],x.shape[1],y_rang.size)  

    #get total current inside area
    mid_pnt = int_3d(z.ravel()+h_z.ravel()/2.,x.ravel()+h_x.ravel()/2.,y.ravel()+h_y.ravel()/2.,*args)
    mid_area= mid_pnt*h_z.ravel()*h_x.ravel()*h_y.ravel()
    tot_area= np.sum(mid_area)

    return tot_area    


