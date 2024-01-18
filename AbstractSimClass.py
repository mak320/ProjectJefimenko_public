import numpy as np
from HelperFunctions import *


class AbstractClass:

    def __init__(self, mass, c:float, forcefields, **kwargs):
        """
        Here we include all fundamental paramters needed in a single time step
        """

        """
        :param c (float): speed of information propagation ('speed of light')
        :param mass  (array, shape = (N, 1)): Mass array
        :param forcefields: dictionary. Stores forcefield names and equation(s) in the form of (u)functions, which must at least have 
        inputs of x, y, z, vx, vy, vz, xr, yr, zr, vx_r, xy_r, vz_r, ax_r, ay_r, az_r and might call kwargs. Any 'interaction' strength should either be 
        hardcoded or passed as an optional argument
        :param kwargs: stores 'mass-like' data, i.e. charge, magnetic charge, some other form of inertia etc. 
        """
        self.c = c
        self.mass = mass

    def append_forcefield(self,name:str):

        return None

    def remove_forcefield(self,name:str):
    
        return None

    
    def construct_forcefield(self):

        return None

        

        
        