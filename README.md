# Gen alpha
This repository contains the implementation of the Generalized alpha Lie group integrator in the index-3-formulation
and its application to the heavy top benchmarking problem and a version of it extended by dissipative forces modeled
via the attachment of a torsion spring along the axis of the top. 

The basics about the generalied alpha integrator can be found in ```ma_thesis.pdf``` along with the modeling process
of the torsion top model.

Within the directory `./src` one will find the following Python code 

- `integrator.py`: Contains classes representing the Generalized alpha integrator in an index-3 Formulation in a basic and scaled version


- `lie.py`: Contains utility classes for calculus in the Lie groups SO3, SE3 and the cartesian product S03xR3


- `model.py`: Contains classes defining the Heavy-top-model in the standard (unconstrained) version and one under the influence of dissipative forces


- `test_visualize_heavytop.py`: Test script visualizing the 3D movement of the regular heavy top


- `test_visualize_heavytorsiontop.py`: Test script visualizing the 3D movement of the heavy top model with torsion spring attached


- `test_visualize_heavytorsiontop_energy.py`: Test script visualizing the energy level over time of the heavy top system in the dissipative case
