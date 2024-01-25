import numpy as np
from HelperFunctions import *

@nb.njit()
def CoulumbField(pos, mom, retpos, retveloc, retacc, charge, k):

    x, y, z = split_cartesian_comps(pos)
    rx, ry, rz = split_cartesian_comps(retpos)

    dx, dy, dz = x-rx.T, y-ry.T, z-rz.T

    r3 = ((dx)**2 + (dy)**2 + (dz)**2)**(3/2)

    invr3 = 1/r3

    for val in np.nditer(invr3):
        if np.isfinite(val) == False:
            val = 0.0

    Fx = (k * dx * invr3 @ charge) * charge
    Fy = (k * dy * invr3 @ charge) * charge
    Fz = (k * dz * invr3 @ charge) * charge

    forces = np.vstack((Fx,Fy,Fz)).T

    return forces


class Field:

    def __init__(self, mass, c:float):
        """
        Here we include all fundamental paramters needed in a single time step
        """

        """
        :param c (float): speed of information propagation ('speed of light')
        :param mass  (array, shape = (N, 1)): Mass array
        :param forcefields: dictionary. Stores forcefield names and equation(s) in the form of (u)functions, which must at least have 
        inputs of pos, mom, retpos, retmom, filter and might call kwargs. 
        
        """
        self.c = c
        self.mass = mass
        self.forcefields = {}
        self.interaction_constants = []

    def append_forcefield(self,ufunc, args_declaration,name:str):

        def lfunc(pos,mom,retpos,retveloc,retacc):

            return ufunc(pos,mom,retpos,retveloc,retacc,*args_declaration)
        
        self.forcefields[name] = lfunc
        


    def remove_forcefield(self,name:str):
    
        del self.forcefields[name]

    
    def construct_forcefield(self):

        def total_function(pos,mom,retpos,retveloc,retacc):
            outp = np.zeros((mass.shape[0],3))
            
            for i in self.forcefields:
                
                outp += self.forcefields[i](pos,mom,retpos,retveloc,retacc)
            return outp
        
        self.callable_forcefield = total_function


mass = np.array([[1.0,1.0,1.0,1.0]]).T     
charge = np.array([1.0,2.0,3.0,4.0])
test = Field(mass, 1.0)
test.append_forcefield(CoulumbField,[charge,2.0],'Coulomb')

pos = np.random.randn(4,3)
rpos = np.random.randn(4,3)

print(test.forcefields['Coulomb'](pos,pos,rpos,rpos,rpos))
test.construct_forcefield()
test.callable_forcefield(pos,pos,rpos,rpos,rpos)
        
        