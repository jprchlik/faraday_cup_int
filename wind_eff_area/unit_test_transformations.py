import numpy as np
import make_discrete_vdf as mdv


def test_fc_transform():
    """
    Test the faraday cup coordiante system transformation in the code

    """
    print('Nothing Yet')

    assert True


def get_coor_1():
    """
    Sends a FC coordinate system to a function with equal values of x,y,z GSE pointing for FC
    """

    #a normal vector describing the pointing of the FC in GSE coordinates
    fc_vec = np.array([1.,1.,1.])/np.sqrt(3.)
    #some phi and theta ancels between the GSE coordinates and FC observations
    theta = np.arctan2(fc_vec[2],np.sqrt(np.sum(fc_vec[:2]**2)))
    phi   = np.arctan2(fc_vec[1],fc_vec[0]) 
    return fc_vec,theta, phi

def get_coor_2():
    """
    Sends a FC coordinate system to a function with FC pointing straight down Xgse
    """

    #a normal vector describing the pointing of the FC in GSE coordinates
    fc_vec = np.array([1.,0.,0.])
    #some phi and theta ancels between the GSE coordinates and FC observations
    theta = np.arctan2(fc_vec[2],np.sqrt(np.sum(fc_vec[:2]**2)))
    phi   = np.arctan2(fc_vec[1],fc_vec[0]) 
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
    vec = get_vec_2()


    #get a set of coordinates and phi angles between faraday cup and GSE coordiantes
    fc_unt,theta,phi = get_coor_2()
    

    #Convert the GSE coordinates into FC coordinates given theta and phi
    fc_vec = mdv.convert_gse_fc(vec,phi,theta)
   

    #These coordinates are shooting something straight down Xgse with -10 in Xgse and 10 in Ygse
    #based on the coordiante transformation with Zfc down the cup normal and Xfc point up the returned
    #value should be [np.array([-10.,-15.,-1500.])
    assert np.allclose(fc_vec,np.array([-10.,-15.,-1500.]))

            



if __name__ == '__main__':
    """
    Run a series of test that make sure all coordinate transformation work
    """

    #Test that GSE to FC to GSE transformation returns the same values
    test_gse_fc_gse_transform()
    ####Test that FC to GSE to FC transformation returns the same values
    test_fc_gse_fc_transform()

    #Test that GSE to FC transformation works
    test_gse_fc_transform()
