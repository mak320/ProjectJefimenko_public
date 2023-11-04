##ProjectJefimenko_public
About
Welcome to ProjectJefimenko_public, a Python-based fully relativistic electromagnetic N-body simulator.

#Project Overview
This repository contains a stable release of an academic project focused on the development of a relativistic N-body simulator (Relativistic Molecular Dynamics Code). The project is a collaborative effort by Tamas A. Vaszary (tamas.vaszary20@imperial.ac.uk), Mate A. Koszta (mate.koszta20@imperial.ac.uk), and Bendeguz Szabo (bendeguz.szabo20@imperial.ac.uk).

#Detailed Explanation
For a comprehensive understanding of the algorithm and its properties, a draft paper is available upon request. Please feel free to contact the authors via email.

#Application
The implemented algorithm, offering precise calculations up to $\mathcal{O}(\Delta t^2)$, is broadly applicable in classical field theories. It addresses the need to incorporate finite signal propagation speed.

This implementation is specifically tailored for classical electrodynamics, utilizing the Liénard–Wiechert (LW) fields. The process involves determining retarded quantities, calculating LW fields, and utilizing a second-order Runge-Kutta time integrator to advance particles. A collision mechanism are also implemented between charges.
