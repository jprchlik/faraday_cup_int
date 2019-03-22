import numpy as np
import mid_point_loop as mpl
import time

def mc_trip(int_3d,z_lo,z_hi,x_lo,x_hi,y_lo,y_hi,args=(),samp=1e5):
    """
    Monte carlo integrator for solar wind on generic Faraday Cup
    Works but takes 30s per run to achieve a reasonable precision (samp = 1e5)

    Parameters
    ---------- 
    int_3d: function
        A function to integrate over. Can be any 3D function defined in python.
    z_lo: float
        The lower limit to integrate over in the z variable
    z_hi: float
        The upper limit to integrate over in the z variable
    x_lo: function
        The lower bound for the x variable in integration. If you want a constant
        you can define x_lo = z: z*0+C
    x_hi: function
        The upper bound for the x variable in integration. If you want a constant
        you can define x_hi = z: z*0+C
    y_lo: function
        The lower bound for the y variable in integration. If you want a constant
        you can define y_lo = z,x: z*x*0+C
    y_hi: function
        The upper bound for the y variable in integration. If you want a constant
        you can define y_hi = z,x: z*x*0+C
    args: tuple,optional
        Tuple of arguments to pass to a function.
    samp: int or float,optional
        Number of samples per variable (Default = 1e5)
   
    Returns
    -------
    area_mc*averg_c: float
        The integral of the given function over the given limits
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

    ###start counting time
    ##time_start = time.time()
    #get total current inside area
    total_c = np.sum(int_3d(vx[inside],vy[inside],vz[inside],*args))
    ###code run time
    ##time_elapsed = (time.time() - time_start)
    ###Time elapsed pring
    ##print('Time to run Intregration  is {1:5.1f}s'.format(samp,time_elapsed))
    
    #get average under the curve
    averg_c = total_c/inside.size

    #get area under the FC cup curve
    area_mc = inside.size/(float(n))*(z_hi-z_lo)*(x_hi(z_hi)-(x_lo(z_hi)))**2
    #area_mc = 1

    return area_mc*averg_c

def mp_trip(int_3d,z_lo,z_hi,x_lo,x_hi,y_lo,y_hi,args=(),samp=1e4):
    """
    Midpoint integration for generic Faraday Cup
    Works really slow for high precision ~2 minutes.

    Parameters
    ---------- 
    int_3d: function
        A function to integrate over. Can be any 3D function defined in python.
    z_lo: float
        The lower limit to integrate over in the z variable
    z_hi: float
        The upper limit to integrate over in the z variable
    x_lo: function
        The lower bound for the x variable in integration. If you want a constant
        you can define x_lo = z: z*0+C
    x_hi: function
        The upper bound for the x variable in integration. If you want a constant
        you can define x_hi = z: z*0+C
    y_lo: function
        The lower bound for the y variable in integration. If you want a constant
        you can define y_lo = z,x: z*x*0+C
    y_hi: function
        The upper bound for the y variable in integration. If you want a constant
        you can define y_hi = z,x: z*x*0+C
    args: tuple,optional
        Tuple of arguments to pass to a function.
    samp: int or float,optional
        Number of samples per variable (Default = 70)
   
    Returns
    -------
    area_cal: float
        The integral of the given function over the given limits
    """


    #midpoint_triple2(int_3d,z_lo,z_hi,x_lo(z_hi),x_hi(z_hi),y)
    #width of mid-point
    h_z = (z_hi-z_lo)/float(samp)

    samp = int(samp)
    #arr = []
    #dif = []
    #[ (y_lo(z_lo+i*h_z,)+k*,h_y,x_min+j*h_x,h_x,z_lo+i*h_z ,h_z) for i in z_low*np.arange(samp) for j in range(samp) for k in range(samp)]
    time_mp_s = time.time()
    arr = [ (k+(0.5+kk)*kka,kka,j+(jj+0.5)*jja,jja,i+(ii+0.5)*iia,iia) for i,ii,iia in zip([z_lo]*samp,range(samp),[h_z]*samp) for j,jj,jja in zip([x_lo(i+(ii+0.5)*iia)]*samp,range(samp),[(x_hi(i+(ii+0.5)*iia)-x_lo(i+(ii+0.5)*iia))/float(samp)]*samp) for k,kk,kka in zip([y_lo(i+(ii+0.5)*iia,j+(jj+0.5)*jja)]*samp,range(samp),[(y_hi(i+(ii+0.5)*iia,j+(jj+0.5)*jja)-y_lo(i+(ii+0.5)*iia,j+(jj+0.5)*jja))/float(samp)]*samp)]
    time_mp_e = time.time()
    print('Time to Create list {0:2.1f}'.format(time_mp_e-time_mp_s))

    area = 0.
    #z = z_lo
    ##loop over all z values
    #while z <= z_hi: 
    #    #get x min and max
    #    x_min,x_max = x_lo(z),x_hi(z)
    #    x = x_min
    #    h_x = (x_max-x_min)/float(samp)
    #    #loop over all x values
    #    while x <= x_max:
    #         y_min,y_max = y_lo(z,x),y_hi(z,x)
    #         y = y_min
    #         h_y = (y_max-y_min)/float(samp)
    #         while y <= y_max:
    #           #  area_cal = int_3d(z,x,y,*args)
    #           #  area += np.sum(area_cal*h_y)
    #             arr.append([y,x,z])
    #             dif.append([h_y,h_x,h_z])
    #             y += h_y
    #         #increment x
    #         x += h_x
    #    #increment z
    #    z += h_z    

    arr =  np.array(zip(*arr))

    time_mp_s = time.time()
    area_cal = np.sum(int_3d(arr[0,:],arr[2,:],arr[4,:],*args)*np.prod(arr[1::2,:],axis=0))
    time_mp_e = time.time()
    print('Time to run Int. Calc. {0:2.1f}'.format(time_mp_e-time_mp_s))

    #return the cumlative area
    return area_cal

def midpoint(f, a, b, n,vector=False):
    n = int(n)
    h = (b-a)/float(n)
    #trigger for the final integration
    if (vector):
        result = f((a + h/2.0) + np.arange(n)*h)
        result *= h
        return np.sum(result)
    #high levels should use the loop
    else:
        result = 0
        for i in range(n):
            result += f((a + h/2.0) + i*h)
        result *= h
        return result

def mp_trip2(int_3d,z_lo,z_hi,x_lo,x_hi,y_lo,y_hi,args=(),samp=70):
    """
    Midpoint integration for generic Faraday Cup. Uses nested functions 
    to speed up computations but ends up being slow for moderate values of
    samp (> 20), thus does not quickly return good precision.

    Parameters
    ---------- 
    int_3d: function
        A function to integrate over. Can be any 3D function defined in python.
    z_lo: float
        The lower limit to integrate over in the z variable
    z_hi: float
        The upper limit to integrate over in the z variable
    x_lo: function
        The lower bound for the x variable in integration. If you want a constant
        you can define x_lo = z: z*0+C
    x_hi: function
        The upper bound for the x variable in integration. If you want a constant
        you can define x_hi = z: z*0+C
    y_lo: function
        The lower bound for the y variable in integration. If you want a constant
        you can define y_lo = z,x: z*x*0+C
    y_hi: function
        The upper bound for the y variable in integration. If you want a constant
        you can define y_hi = z,x: z*x*0+C
    args: tuple,optional
        Tuple of arguments to pass to a function.
    samp: int or float,optional
        Number of samples per variable (Default = 70)
   
    Returns
    -------
    area_cal: float
        The integral of the given function over the given limits
    """
    def p(z,y):
        return midpoint(lambda x: int_3d(x,y,z,*args),y_lo(z,y),y_hi(z,y),samp,vector=True)

    def q(z):
        return midpoint(lambda y: p(z,y),x_lo(z),x_hi(z),samp)

    return midpoint(q,z_lo,z_hi,samp)

def mp_trip3(int_3d,z_lo,z_hi,x_lo,x_hi,y_lo,y_hi,args=(),samp=1e4):
    """
    Midpoint integration for generic Faraday Cup
    DOES NOT WORK 
    """

    #midpoint_triple2(int_3d,z_lo,z_hi,x_lo(z_hi),x_hi(z_hi),y)
    #width of mid-point
    z,h_z = np.linspace(z_lo,z_hi,samp,endpoint=True,retstep=True)

    samp = int(samp)
    #get xvalues
    x_min,x_max = x_lo(z),x_hi(z)
    x_diff = x_max-x_min
    x_rang = np.arange(samp)
    x_resl = x_diff/float(samp) 
    h_x = np.outer(x_resl,x_rang)
    x      = (np.outer(x_min,np.ones(h_x.shape[1]))+h_x).ravel()  
    h_x    = h_x.ravel()
    print(x_min,x_max)

    #update z shape
    z       = np.outer(z,np.ones(samp)).ravel()
    h_z     = np.outer(h_z+np.zeros(int(samp)),np.ones(int(samp))).ravel()

    #get yvalues
    y_min, y_max = y_lo(z,x),y_hi(z,x)
    y_diff       = y_max-y_min
    y_rang       = np.arange(samp)
    y_resl       = y_diff/float(samp)
    h_y          = np.outer(y_resl,y_rang)
    y            = (np.outer(y_min,np.ones(h_y.shape[1]))+h_y).ravel()
    h_y          = h_y.ravel()

    #update z shape
    z            = np.outer(z,np.ones(samp)).ravel()
    h_z          = np.outer(h_z,np.ones(samp)).ravel() 
    #update x shape
    x            = np.outer(x,np.ones(samp)).ravel()
    h_x          = np.outer(h_x,np.ones(samp)).ravel()

    #get total current inside area
    mid_pnt = int_3d(x+h_x/2.,y+h_y/2.,z+h_z/2.,*args)
    mid_area= mid_pnt*h_z*h_x*h_y
    tot_area= np.sum(mid_area)

    return tot_area,z,x,y


def mp_trip_cython(int_3d,z_lo,z_hi,x_lo,x_hi,y_lo,y_hi,args=(),samp=70):
    """
    Midpoint integral using Cython to build an array of x,dx,y,dy,z,dz arrays. It was built
    for intregration to compute a RDF for the solar wind but can be used for any integral.
    Note the Z,X,Y convention is left over from the IDL convention. The integration works
    by integrating over z then x then y.
   
    Parameters
    ---------- 
    int_3d: function
        A function to integrate over. Can be any 3D function defined in python.
    z_lo: float
        The lower limit to integrate over in the z variable
    z_hi: float
        The upper limit to integrate over in the z variable
    x_lo: function
        The lower bound for the x variable in integration. If you want a constant
        you can define x_lo = z: z*0+C
    x_hi: function
        The upper bound for the x variable in integration. If you want a constant
        you can define x_hi = z: z*0+C
    y_lo: function
        The lower bound for the y variable in integration. If you want a constant
        you can define y_lo = z,x: z*x*0+C
    y_hi: function
        The upper bound for the y variable in integration. If you want a constant
        you can define y_hi = z,x: z*x*0+C
    args: tuple,optional
        Tuple of arguments to pass to a function.
    samp: int or float,optional
        Number of samples per variable (Default = 70)
   
    Returns
    -------
    area_cal: float
        The integral of the given function over the given limits
    """


    #use Cython function to create array for loop
    arr = mpl.create_mid_point_arr(z_lo,z_hi,x_lo, x_hi, y_lo, y_hi,samp)
    #time_mp_s = time.time()
    #Outputs in the following coordinates
    #Z,dZ,X,dX,Y,dY
    arr = np.asarray(arr)
    #compute the integral using midpoint integration
    #Use the same Y,X,Z as used by scipy tplquad
    area_cal = np.sum(int_3d(arr[:,4],arr[:,2],arr[:,0],*args)*np.prod(arr[:,1::2],axis=1))
    #time_mp_e = time.time()
    #print('Time to run Int. Calc. {0:2.1f}'.format(time_mp_e-time_mp_s))

    return area_cal
 


def mp_trip_cython_pqr(int_3d,z_lo,z_hi,x_lo,x_hi,y_lo,y_hi,z_samp,x_samp,y_samp,args=()):
    """
    Midpoint integral using Cython to build an array of x,dx,y,dy,z,dz arrays. It was built
    for intregration to compute a RDF for the solar wind but can be used for any integral.
    Note the Z,X,Y convention is left over from the IDL convention. The integration works
    by integrating over z then x then y.
   
    Parameters
    ---------- 
    int_3d: function
        A function to integrate over. Can be any 3D function defined in python.
    z_lo: float
        The lower limit to integrate over in the z variable
    z_hi: float
        The upper limit to integrate over in the z variable
    x_lo: function
        The lower bound for the x variable in integration. If you want a constant
        you can define x_lo = z: z*0+C
    x_hi: function
        The upper bound for the x variable in integration. If you want a constant
        you can define x_hi = z: z*0+C
    y_lo: function
        The lower bound for the y variable in integration. If you want a constant
        you can define y_lo = z,x: z*x*0+C
    y_hi: function
        The upper bound for the y variable in integration. If you want a constant
        you can define y_hi = z,x: z*x*0+C
    z_samp: int
        Number of samples in the z direction.
    x_samp: int
        Number of samples in the x direction.
    y_samp: int
        Number of samples in the y direction.
    args: tuple,optional
        Tuple of arguments to pass to a function.
   
    Returns
    -------
    area_cal: float
        The integral of the given function over the given limits
    """

    #Total number of the samples
    samp = z_samp*x_samp*y_samp
    #use Cython function to create array for loop
    arr = mpl.create_mid_point_arr_pqr_samp(z_lo,z_hi,x_lo, x_hi, y_lo, y_hi,samp,z_samp,x_samp,y_samp)

    #time_mp_s = time.time()
    #Outputs in the following coordinates
    #Z,dZ,X,dX,Y,dY
    arr = np.asarray(arr)
    #compute the integral using midpoint integration
    #Use the same Y,X,Z as used by scipy tplquad
    area_cal = np.sum(int_3d(arr[:,4],arr[:,2],arr[:,0],*args)*np.prod(arr[:,1::2],axis=1))
    #time_mp_e = time.time()
    #print('Time to run Int. Calc. {0:2.1f}'.format(time_mp_e-time_mp_s))

    return area_cal
 


