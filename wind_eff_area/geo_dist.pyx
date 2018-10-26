import numpy as np
cimport numpy as np
DTYPE = np.float64
#robost length
def robost_length(float x0,float x1):
    """
    Get the sum squared distance between two points
 
    Input
    -----
    x0: float
        Distance in x direction
    x1: float
        Distance in y direction
   
    Returns
    -------
    dist: float
        The distance given the two differences in the components

    """
    cdef double dist

    if x0 > x1:
        dist = x0*np.sqrt(1.+(x1/x0)**2)
    else:
        dist = x1*np.sqrt(1.+(x0/x1)**2)
    return dist

#compute root 
def get_root(float r0,float z0,float z1,float g):
    """
    Finds Quadratic root fast

    Parameters
    ----------
    r0: float
    z0: float
    g: float
        Current value of the equation for the ellipse is the solver

    Return
    -----
    s: float
        The solution of root equation
    """

    cdef int   max_iterations = 100
    cdef int i=0
    cdef float n0,s0,s1,s 

    n0 = r0*z0
    s0 = z1-1
    s1 = robost_length(n0,z1)
    s = 0.

    while i < max_iterations:
        s = (s0+s1)/2.
        if ((s == s0) | (s == s1)):
            return s
        ratio0, ratio1 = n0/(s+r0), z1/(s+1)
        g = (ratio0)**2+(ratio1)**2-1
        if (g > 0.):
            s0 = s
        elif (g < 0.): 
            s1 = s
        else:
            return s
        i += 1
        
    return 1.e30
        

#compute distance from ellipse 
#https://www.geometrictools.com/Documentation/DistancePointEllipseEllipsoid.pdf Section 2.7
def distance_point_ellipse(float a,float b,float x,float y):
    """
    
    Parameters
    ----------
    a: float
        Semimajor axis
    b: float
        Semiminor axis
    x: float
        x-coordinate of a point to find a distance to with respect to the defined ellipse
    y: float
        y-coordinate of a point to find a distance to with respect to the defined ellipse
 
    Returns
    -------
    dist: float
        Distance between given point and a ellipse
    """


    cdef float dist,g,z0,z1,sbar
    cdef float x0,y0
    cdef float numer0,denom0,xde0

    if y > 0:
        if x > 0:
            #initial guess for root finding
            z0 = x/a
            z1 = y/b
            g = (z0)**2+(z1)**2-1
            if (g != 0):
                r0 = np.sqrt(a/b)
                sbar = get_root(r0,z0,z1,g)
                x0 = r0*x/(sbar+r0)
                y0 = y/(sbar+1)
                #print(sbar)
                dist = np.sqrt((x-x0)**2+(y-y0)**2)
            else:
                x0,y0,dist = x,y,0
        else: 
            x0 = 0
            y0 = b
            dist = np.abs(y-y0)
    else:
        numer0 = a*x
        denom0 = np.sqrt(a)-np.sqrt(b)
        if numer0 < denom0:
            xde0 = numer0/denom0
            x0 = a*xde0
            y0 = b*np.sqrt(1-(xde0)**2)
            dist = np.sqrt((x0-x)**2+y0**2)
        else:
            x0 = a
            y0 = 0
            dist = np.abs(x-x0)
    return dist


def ellipse_distance_pq(float a,float b,double[:] xs,double[:] ys):
    """
    Computes distance between ellipse and a point for an array of points   
 
    Parameters
    ----------
    a: float
        Semimajor axis
    b: float
        Semiminor axis
    xs: float
        x-coordinate of points to find a distance to with respect to the defined ellipse. X and Y must have the same number of coordiantes and be in 1D
    ys: float                                                                                                                             
        y-coordinate of points to find a distance to with respect to the defined ellipse. X and Y must have the same number of coordiantes and be in 1D

    Returns
    -------
    dist: float
        Distance between given point and a ellipse
    """

    cdef int sizer = len(xs)
    cdef int i = 0
    cdef double[:] ds = np.zeros(sizer)
       

    #compute the distances to each point from an ellipse
    for i in range(sizer):
        ds[i] = distance_point_ellipse(a,b,xs[i],ys[i])
    
    return ds