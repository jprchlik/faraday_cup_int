import numpy as np
import make_discrete_vdf as mdv
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from fancy_plot import fancy_plot


def s1(alpha):
    """
    Euler rotation matrix with only a rotation in alpha (phi)
    """
    s1_mat = np.array([[np.cos(alpha),np.sin(alpha),0.],
                       [-np.sin(alpha),np.cos(alpha),0.],
                       [0.,0.,1.]]) 
    return s1_mat

def s2(beta):
    """
    Euler rotation matrix with only a rotation in beta (theta)
    """
    s2_mat = np.array([[np.cos(beta),0.,-np.sin(beta)],
                       [0.,1.,0.], 
                       [np.sin(beta),0.,np.cos(beta)]])
    return s2_mat
   
def s3(gamma):
    """
    Euler rotation matrix with only a rotation in gamma (psi)
    """
    s3_mat = np.array([[np.cos(gamma),np.sin(gamma),0.],
                       [-np.sin(gamma),np.cos(gamma),0.],
                       [0.,0.,1.]]) 
    return s3_mat
   
def gol_d(phi):
    s1_mat = np.array([[np.cos(phi),np.sin(phi),0.],
                       [-np.sin(phi),np.cos(phi),0.],
                       [0.,0.,1.]]) 
    return s1_mat
def gol_c(theta):
    """
    Euler rotation matrix with only a rotation in theta (theta)
    xyz convention
    """
    s2_mat = np.array([[np.cos(theta),0.,-np.sin(theta)],
                       [0.,1.,0.], 
                       [np.sin(theta),0.,np.cos(theta)]])
    return s2_mat
    
   
def gaus(x,a,x0,sigma):
    """
    Simple Gaussian function

    """
    return a*np.exp(-(x-x0)**2/(2*sigma**2))



def get_phi_theta(fc_vec):
    """
    Get Phi and Theta from definition of normal vector with respect to GSE coordinates

    Parameters
    -------
    fc_vec: np.array
        Three components of direction of FC normal with respect to GSE coordinates
    """
    #some phi and theta ancels between the GSE coordinates and FC observations
    theta = np.arctan2(fc_vec[2],np.sqrt(np.sum(fc_vec[:2]**2)))
    phi   = np.arctan2(fc_vec[1],fc_vec[0]) 
    return phi, theta


def get_coor_1():
    """
    Sends a FC coordinate system to a function with equal values of x,y,z GSE pointing for FC
    """

    #a normal vector describing the pointing of the FC in GSE coordinates
    fc_vec = np.array([1.,1.,1.])/np.sqrt(3.)
    #some phi and theta ancels between the GSE coordinates and FC observations
    phi, theta = get_phi_theta(fc_vec)
    return fc_vec,theta, phi

def get_coor_2():
    """
    Sends a FC coordinate system to a function with FC pointing straight down Xgse
    """

    #a normal vector describing the pointing of the FC in GSE coordinates
    fc_vec = np.array([1.,0.,0.])
    #some phi and theta ancels between the GSE coordinates and FC observations
    phi, theta = get_phi_theta(fc_vec)
    return fc_vec,theta, phi



def get_vec_1():
    """
    Get a vector of coordinates
    """
    #some X,Y,Z values in coordiante system
    vec = np.array([500.,350.,-900.])
    return vec

def get_vec_2():
    """
    Get a vector of coordinates with something shooting straight down Xgse
    """
    #some X,Y,Z values in coordiante system
    vec = np.array([-1500.,15.,-10.])
    return vec

def get_vec_3():
    """
    A vector of coordinates to prove I am not crazy
    """

    #a normal vector describing the pointing of the FC in GSE coordinates
    vec = np.array([1.,5.,10.])
    #some phi and theta ancels between the GSE coordinates and FC observations
    return vec



def test_gse_fc_gse_transform():
    """
    Test the faraday cup coordiante system transformation in the code from GSE to FC to GSE
    """

    #some X,Y,Z values in coordiante system
    vec = get_vec_1()


    #get a set of coordinates and phi angles between faraday cup and GSE coordiantes
    fc_vec,theta,phi = get_coor_1()
    

    #Convert the GSE coordinates into FC coordinates given theta and phi
    fc_vec = mdv.convert_fc_gse(vec,phi,theta)

    #convert coordinates back to GSE coordinates
    gse_vec = mdv.convert_gse_fc(fc_vec,phi,theta) 


    #Check that I recover the original vector after coordinate transformation
    assert np.allclose(gse_vec,vec)
    
def test_fc_gse_fc_transform():
    """
    Test the faraday cup coordiante system transformation in the code from GSE to FC to GSE
    """

    #some X,Y,Z values in coordiante system
    vec = get_vec_1()


    #get a set of coordinates and phi angles between faraday cup and GSE coordiantes
    fc_vec,theta,phi = get_coor_1()
    

    #Convert the GSE coordinates into FC coordinates given theta and phi
    gse_vec = mdv.convert_gse_fc(vec,phi,theta)
    #creates 1x3 array from 3x1 array
    gse_vec = gse_vec.ravel()


    #convert coordinates back to GSE coordinates
    fc_vec = mdv.convert_fc_gse(gse_vec,phi,theta) 

    #Check that I recover the original vector after coordinate transformation
    assert np.allclose(fc_vec.ravel(),vec)

def test_gse_fc_transform():
    """
    Test the faraday cup coordiante system transformation in the code from GSE to FC 
    """

    #some X,Y,Z values in coordiante system
    #np.array([-1500.,15.,-10.])
    vec = get_vec_2()


    #get a set of coordinates and phi angles between faraday cup and GSE coordiantes
    fc_unt,theta,phi = get_coor_2()
    

    #Convert the GSE coordinates into FC coordinates given theta and phi
    fc_vec = mdv.convert_gse_fc(vec,phi,theta)
   

    #These coordinates are shooting something straight down Xgse with -10 in Xgse and 10 in Ygse
    #based on the coordiante transformation with Zfc down the cup normal and Xfc pointing to the right
    #value should be [np.array([-10.,-15.,-1500.])
    assert np.allclose(fc_vec,np.array([15.,-10.,-1500.]))

def test_fc_gse_transform_2():
    """
    Test the faraday cup coordiante system transformation in the code from GSE to FC 
    """


    #get a set of coordinates and phi angles between faraday cup and GSE coordiantes
    #a normal vector in the FC coordinate system
    #rotate at many angles to make sure transformation work
    vec,theta,phi = get_coor_2()
    

    #Convert the GSE coordinates into FC coordinates given theta and phi
    fc_vec_00   = mdv.convert_gse_fc(vec,0.,0.)
    fc_vec_pi20 = mdv.convert_gse_fc(vec,np.pi/2.,0.)
    fc_vec_0pi2 = mdv.convert_gse_fc(vec,0.,np.pi/2.)
   

    assert np.allclose(fc_vec_00,np.array([0.,0.,1.]))
    assert np.allclose(fc_vec_00,np.array([-1.,0.,0.]))
    assert np.allclose(fc_vec_00,np.array([0.,-1.,0.]))

def test_gse_fc_transform_2():
    """
    Test the faraday cup coordiante system transformation in the code from GSE to FC 
    """


    #get a set of coordinates and phi angles between faraday cup and GSE coordiantes
    #a normal vector in the FC coordinate system
    #rotate at many angles to make sure transformation work
    vec = get_vec_3()
    

    #Convert the GSE coordinates into FC coordinates given theta and phi
    fc_vec_00   = mdv.convert_gse_fc(vec,0.,0.)
    fc_vec_pi20 = mdv.convert_gse_fc(vec,np.pi/2.,0.)
    fc_vec_0pi2 = mdv.convert_gse_fc(vec,0.,np.pi/2.)
   

    assert np.allclose(fc_vec_00,np.array([5.,10.,1.]))
    assert np.allclose(fc_vec_pi20,np.array([-1.,10.,5.]))
    assert np.allclose(fc_vec_0pi2,np.array([5.,-1.,10.]))

def pls_par_1():
    """
    One set of plasma parameters for test in FC measurements 
    """

    #                    Vx  ,  Vy,  Vz ,Wper,Wpar, Np
    pls_par = np.array([-380., -130., 30., 10., 40., 5.])
    return pls_par

def pls_par_2():
    """
    One set of plasma parameters for test in FC measurements 
    """

    #                    Vx  ,  Vy,  Vz ,Wper,Wpar, Np
    pls_par = np.array([-550., -30., 10., 20., 80., 50.])
    return pls_par


def calc_cont(grid_v):
    """
    Calculate constant to change from measured current to particles per velocity bin
   
    Parameters
    ------
    grid_v: np.array 
        Grid of velocities for the measured FC

    Returns
    -------
    cont: np.array
        Constants which will turn current in a velocity grid into particles in a velocity
        grid.
    """

    waeff = 3.8e6 #cm^3/km
    q0    = 1.6021892e-7 # picocoulombs
    dv    = np.diff(grid_v)
    dv    = np.concatenate([dv,[dv[-1]]])
    cont  = 1.e12/(waeff*q0*dv*grid_v)

    return cont


def test_pars_one_fc_1(plot=False):
    """
    Test returns plasma parameters when you have the solar wind coming right down the barrel of the FC
    """

    #get a set of plasma parameters
    pls_par = pls_par_1()

    #Use a special case when magnetic field vector follows plasma vector
    mag_par = pls_par[:3]/np.linalg.norm(pls_par[:3])
    #have FC also aligned with magnetic field velocity field        
    fc_vec = mag_par

    #get phi and theta for particular Parameters values
    phi, theta = get_phi_theta(-fc_vec)

    phi,theta = np.degrees([phi,theta])

    #number of sample for integration aming for 2 part in 1,000
    samp = 6.9e1
    #switched to km/s in p',q',r' space 2018/10/19 J. Prchlik
    samp = 1.5e1
    #make a discrete VDF
    #Switched to clipping in km/s 2018/10/19 J. Prchlk
    dis_vdf = mdv.make_discrete_vdf(pls_par,mag_par,pres=1.00,qres=1.00,clip=200.)

    #measurement velocity grid
    dv = 15
    grid_v = np.arange(270,600,dv)

    #Constant to convert current to particles
    cont = calc_cont(grid_v)

    #calculate x_meas array
    #phi and theta should be Parameters in degrees
    x_meas = mdv.make_fc_meas(dis_vdf,fc_spd=grid_v,fc_phi=phi,fc_theta=theta)

    #compute the observed current in the instrument
    rea_cur = mdv.arb_p_response(x_meas,dis_vdf,samp)


    #compute Using a cold current assumption which should be good considering the FC cup is looking down the propaging vector
    col_cur = mdv.p_bimax_response(x_meas,np.concatenate([pls_par,mag_par]))


    #get fit parameters from measurement
    #calculate the Gaussian fit of the response
    popt, pcov = curve_fit(gaus,grid_v,rea_cur*cont,p0=[np.nanmax(rea_cur*cont),np.mean(grid_v),np.sqrt(2.)*2*dv],sigma=1./(rea_cur/rea_cur.min()))

    #Switched to computing the average
    #####get the parameters from the fit
    u = popt[1] #speed in km/s
    w = np.abs(popt[2]*np.sqrt(2.)) #thermal speed in km/s
    n = popt[0]*w*np.sqrt(np.pi) #density in cc
    

    #print(np.linalg.norm(pls_par[:3]),np.linalg.norm(pls_par[3:5]),pls_par[5])
    ##Guess
    #print(u,w,n)

    fig, ax = plt.subplots()
    ax.plot(grid_v,rea_cur.ravel()*cont,'-.b',label='Parameters',linewidth=3)
    ax.plot(grid_v,col_cur*cont,'--r',label='Cold',linewidth=3)
    #ax.plot(grid_v, gaus(grid_v,*popt),'--',marker='o',label='Gauss Fit',linewidth=3)
    #ax.plot(grid_v,init_guess.ravel()*cont,':',color='purple',label='Init. Guess',linewidth=3)


    ax.set_xlabel('Speed [km/s]')
    ax.set_ylabel('p/cm$^{-3}$/(km/s)')



    err_val = np.sqrt(np.sum((rea_cur-col_cur)**2))/np.sum(col_cur)

    #plot if keyword set
    if plot:
        mdv.plot_vdf(dis_vdf)
        plt.show()
    
    #Test error value is within tolerance
    print(err_val)
    assert err_val < 2e-3


def test_pars_one_fc_2(plot=False):
    """
    Test returns plasma parameters when you have the solar wind coming right down the barrel of the FC
    """

    #get a set of plasma parameters
    pls_par = pls_par_2()

    #Use a random magnetic field vector follows plasma vector
    mag_par =  [np.cos(np.degrees(75.)), np.sin(np.degrees(75.)), 0.]# chosen arbitrarily

    #get phi and theta for particular Parameters values
    phi,theta = -10,4

    #number of sample for integration aming for 2 part in 1,000
    samp = 4.5e1
    #switched to km/s in p',q',r' space 2018/10/19 J. Prchlik
    samp = 1.0e1
    #make a discrete VDF
    dis_vdf = mdv.make_discrete_vdf(pls_par,mag_par,pres=1.00,qres=1.00,clip=200.)

    #measurement velocity grid
    dv = 15
    grid_v = np.arange(470,800,dv)

    #Constant to convert current to particles
    cont = calc_cont(grid_v)

    #calculate x_meas array
    #phi and theta should be Parameters in degrees
    x_meas = mdv.make_fc_meas(dis_vdf,fc_spd=grid_v,fc_phi=phi,fc_theta=theta)

    #compute the observed current in the instrument
    rea_cur = mdv.arb_p_response(x_meas,dis_vdf,samp)


    #compute Using a cold current assumption which should be good considering the FC cup is looking down the propaging vector
    col_cur = mdv.p_bimax_response(x_meas,np.concatenate([pls_par,mag_par]))

    fig, ax = plt.subplots()
    ax.plot(grid_v,rea_cur.ravel()*cont,'-.b',label='Parameters',linewidth=3)
    ax.plot(grid_v,col_cur*cont,'--r',label='Cold',linewidth=3)
    #ax.plot(grid_v, gaus(grid_v,*popt),'--',marker='o',label='Gauss Fit',linewidth=3)
    #ax.plot(grid_v,init_guess.ravel()*cont,':',color='purple',label='Init. Guess',linewidth=3)


    ax.set_xlabel('Speed [km/s]')
    ax.set_ylabel('p/cm$^{-3}$/(km/s)')

    if plot:
        mdv.plot_vdf(dis_vdf)
        plt.show()
      
    err_val = np.sqrt(np.sum((rea_cur-col_cur)**2))/np.sum(col_cur)
    print(err_val)
    #Test error value is within tolerance
    assert err_val < 2e-3


def test_rotation_matrix():
    """
    Compare rotation matrix to one derived from the dot of 3 different rotations
    """
   
    phi,theta,psi = np.zeros(3)+np.pi/2.


    rot_mat_1 = mdv.rotation_matrix(phi,theta,psi_ang=psi)
    rot_mat_2 = (s1(phi).dot(s2(theta))).dot(s3(psi))

    assert np.allclose(rot_mat_1,rot_mat_2)

if __name__ == '__main__':
    """
    Run a series of test that make sure all coordinate transformation work
    """

    #Test that GSE to FC to GSE transformation returns the same values
    test_gse_fc_gse_transform()
    ####Test that FC to GSE to FC transformation returns the same values
    test_fc_gse_fc_transform()

    ###run a couple more tests
    test_gse_fc_transform_2()
    ###run a couple more tests
    #test_fc_gse_transform_2()

    #Test that GSE to FC transformation works
    test_gse_fc_transform()


    #test rotation matrix is the same from function and 3 R3 rotations
    test_rotation_matrix()

    #test you get reasonable answers when you shove everything right down on FC
    test_pars_one_fc_1(plot=False)
    test_pars_one_fc_2(plot=False)
