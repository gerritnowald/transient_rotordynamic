# transient_rotordynamic
Simulation of an elastic (Jeffcott/Laval) rotor in fixed and journal bearings (analytical short bearing solution).

Details of the dynamic behavior can be found in this blog post: https://gerritnowald.wordpress.com/2022/02/05/simulating-vibration-of-rotors-with-python/

Some comments on optimization of runtime, including numba: https://gerritnowald.wordpress.com/2022/02/08/speeding-up-simulations-in-python/

ODEs of the rotors are taken from my Ph.D. dissertation:
Nowald, Gerrit Edgar (2018). Numerical Investigation of Rotors in Floating Ring Bearings using Co-Simulation. Technische Universität Darmstadt. https://tuprints.ulb.tu-darmstadt.de/8186

Short bearing theory is taken from
Vrande, van de, B. L. (2001). Nonlinear dynamics of elementary rotor systems with compliant plain journal bearings. Technische Universiteit Eindhoven. https://doi.org/10.6100/IR550147
