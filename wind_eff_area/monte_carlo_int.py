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

    midpoint_triple2(int_3d,z_lo,z_hi,x_lo(z_hi),x_hi(z_hi),y)

    return area

def test_3d_int():

    mc_trip(int_3d,z_lo,z_hi,x_lo,x_hi,y_lo,y_hi,args=(),samp=1e4)

    return


def midpoint(f, a, b, n):
    h = float(b-a)/n
    result = 0
    for i in range(n):
        result += f((a + h/2.0) + i*h)
    result *= h
    return result

def midpoint_triple2(g, a, b, c, d, e, f, nx, ny, nz):
    def p(x, y):
        return midpoint(lambda z: g(x, y, z), e, f, nz)

    def q(x):
        return midpoint(lambda y: p(x, y), c, d, ny)

    return midpoint(q, a, b, nx)

def test_midpoint_triple():
    """Test that a linear function is integrated exactly."""
    def g(x, y, z):
        return 2*x + y - 4*z

    a = 0;  b = 2;  c = 2;  d = 3;  e = -1;  f = 2
    import sympy
    x, y, z = sympy.symbols('x y z')
    I_expected = sympy.integrate(
        g(x, y, z), (x, a, b), (y, c, d), (z, e, f))
    for nx, ny, nz in (3, 5, 2), (4, 4, 4), (5, 3, 6):
        I_computed1 = midpoint_triple1(
            g, a, b, c, d, e, f, nx, ny, nz)
        I_computed2 = midpoint_triple2(
            g, a, b, c, d, e, f, nx, ny, nz)
        tol = 1E-14
        print I_expected, I_computed1, I_computed2
        assert abs(I_computed1 - I_expected) < tol
        assert abs(I_computed2 - I_expected) < tol

if __name__ == '__main__':
    test_midpoint_triple()
