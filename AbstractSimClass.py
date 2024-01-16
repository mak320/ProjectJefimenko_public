import numpy as np


class AbstractClass:


    def __init__(self, mass, c, **kwargs):
        """
        Here we include all fundametal paramters needed in a single time step
        """

        """
        :param c (float): speed of information propagation ('speed of light')
        :param mass  (array, shape = (N, 1)): Mass array
        """
        self.c = c
        self.mass = mass

        

        
        