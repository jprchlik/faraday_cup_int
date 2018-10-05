import numpy as np
def create_mid_point_arr(float z_lo,float z_hi,object x_lo,object x_hi,object y_lo,object y_hi, int samp):

    
    cdef int N = samp**3
    cdef double[:,:] arr = np.zeros((N,6))
    cdef int i, j, k, l = 0
    cdef double r = 0
    cdef double z,h_z,x,h_x,y,h_y
    cdef double x_min,x_max,y_min,y_max,z_min,z_max


    #differnce in z parameter
    h_z = (z_hi-z_lo)/float(samp)
    samp = int(samp)
    z = z_lo+h_z/2.

    #loop over all z values
    for i in range(samp): 
        #get x min and max
        x_min,x_max = x_lo(z),x_hi(z)
        h_x = (x_max-x_min)/float(samp)
        x = x_min+h_x/2.
        #loop over all x values
        for j in range(samp):
             y_min,y_max = y_lo(z,x),y_hi(z,x)
             h_y = (y_max-y_min)/float(samp)
             y = y_min+h_y/2.
             for k in range(samp):
                 #store values in array
                 #arr[l,:] = np.array([y,h_y,x,h_x,z,h_z])
                 arr[l,0] = z
                 arr[l,1] = h_z
                 arr[l,2] = x
                 arr[l,3] = h_x
                 arr[l,4] = y
                 arr[l,5] = h_y
                 l += 1 #increment array index                 
                 y += h_y
             #increment x
             x += h_x
        #increment z
        z += h_z    

    return arr
