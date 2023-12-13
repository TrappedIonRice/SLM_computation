Main operational files:
IFTA.py is the primary file to be run for this module, as it contains all the IFTA phase profile optimization algorithms

SLM.py contains a wrapper class for SLM and planar light field basic functions and helper functions

profile.py produces some useful planar light fields

ArrayModulator.py implements elementary SLM functions to deflect/scatter beams, transform to TEM01, etc.

InversePhase backpropagates a target light field and simply applies the resulting phase to the SLM

image_modifier.py modifies existing phase profiles by simply adding other phase masks


Unfinished/experimental files:
Main.py utilizes the qpSLM package functionality

Arrizon.py implements an analytical algorithm of the Arrizon paper

ConjugateGradient.py implements the conjugate gradient algorithm