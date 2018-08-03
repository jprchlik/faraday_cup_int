# RDF Integrations

This is a suite of code which will calculate a Reduced Distribution Function (RDF) by integrating the Velocity Distribution Function (VDF) over a Faraday Cup (FC).


make_discrete_vdf.py
-------
A set of function for creating a discrete vdf and integrating over a discrete vdf.
For more information on the functions enclosed use the python help function.    

python\>import make_discrete_vdf as mdv    
python\>help(mdv)     



monte_carlo_int.py
-------
The now poorly named 3D integration module. It contains various methods for 
performing 3D integrations. The fastest and most accurate module is mp_trip_cython,
which is why mp_trip_cython is used in the integration in make_discrete_vdf.py.



mid_point_loop.pyx
-------
A cython module used to create an array of x,y,z and their bin widths.
It is used by the mp_trip_cython module in monte_car_int.
In order for this to run in python first run the following command,
which will compile the code:


tcsh\>python setup.py build_ext --inplace



setup.py
----
Used to compile the mid_point_loop.pyx module.




quick_script.py
-----
A example script on how to use make_discrete_vdf to calculate a RDF.





unit_test_int.py
---------
A module to unit test whether monte_carlo_int returns a value within a given relative tolerance (1e-4 currently) for integrating a sphere.