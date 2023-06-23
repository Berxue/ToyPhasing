# ToyPhasing
## General
2D iterative phasing toy example.
The code uses the Hybrid Input-Output method as well as Shrinkwrap.
Some relevant papers are
* Fienup, J. R. (1978), ‘Reconstruction of an object from the modulus of its fourier transform’, Opt. Lett. 3(1), 27–29.
* Marchesini et al. (2003), ‘X-ray image reconstruction from a diffraction pattern alone’, Phys. Rev. B 68(14).

Given an input picture and a support mask the algorithm will try to recover the input picture from only the amplitude of the pictures Fourier transform.

## Dependencies & Startup
The code depends on numpy and opencv-python.

Once the dependencies are installed the pasing example can be started by calling:

    python toy_phasing.py

## Results
The phasing routine might take some time to complete.
