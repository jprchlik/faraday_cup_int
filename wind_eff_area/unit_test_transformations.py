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
    Sends a FC coordinate system to a funciton
    """

    #a normal vector describing the pointing of the FC in GSE coordinates
    fc_vec = np.array([1.,1.,1.])/np.sqrt(3.)
    #some phi and theta ancels between the GSE coordinates and FC observations
    theta = np.arctan2(fc_vec[2],np.sqrt(np.sum(fc_vec[:2]**2)))
    phi   = np.arctan2(fc_vec[1],fc_vec[0]) 
    return fc_vec,theta, phi

def get_vec_1():
    """
    Get a vector of coordinates
    """
    #some X,Y,Z values in coordiante system
    vec = [500.,350.,-900.]
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
    assert np.allclose(gse_vec.ravel(),vec)
    
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
    gse_vec = gse_vec.ravel()


    #convert coordinates back to GSE coordinates
    fc_vec = mdv.convert_fc_gse(gse_vec,phi,theta) 

    #Check that I recover the original vector after coordinate transformation
    assert np.allclose(fc_vec.ravel(),vec)


if __name__ == '__main__':
    """
    Run a series of test that make sure all coordinate transformation work
    """

    #Test that GSE to FC to GSE transformation returns the same values
    test_gse_fc_gse_transform()
    #Test that FC to GSE to FC transformation returns the same values
    test_fc_gse_fc_transform()
