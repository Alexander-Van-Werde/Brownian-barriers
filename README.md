This repository accompanies the paper "Recovering semipermeable barriers from reflected Brownian motion" by Alexander Van Werde and Jaron Sanders (2024). That paper studied _reflected Brownian motion with semipermeable barriers_ in a planar domain, particularly the recovery of barrier locations based on an observed trajectory.

The module _brownian\_barriers_ provides recovery and simulation algorithms for reflected Brownian motion with semipermeable barriers in a planar domain. Examples for the usage are given in the Jupyter notebook. 

Main methods 
-------------
* recover\_barriers\_wasserstein: 
    Recovers the barriers based on a trajectory by looking for discontinuities in the empirical transition kernel along a grid of squares. Continuity is here measured using the Wasserstein distance. 
* get\_sample\_path: 
    Simulation scheme which generates a trajectory of a reflected Brownian motion with semipermeable barriers.


Dependencies
------------
We rely on the Python Optimal Transport library (POT) for computing Wasserstein distances (https://pythonot.github.io/). The following libraries are also used: numpy, scipy, xml, svg, and typing.  
