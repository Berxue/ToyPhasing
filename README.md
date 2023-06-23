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
The phasing routine might take some time to complete. Once it completes the following subfolders and files wil be generated.
* results
  * _picture_name_
    * _date_-_time_
      * fft_images
        * fft_intensity.png
        * fft_phase.png
        * fft.png
        * phase_inverse.png
        * sqrt_of_intensity_inverse.png
        * autocorrelation.png
        * full_inverse.png
      * initial_density.png
      * initial_mask.png
      * reconstructed_density.png
      * reconstructed_mask.png
      * reconstruction_video_density_colored.avi
        
The folder fft_images contain images depicting the Fourier transform of the starting image, its [phase](https://en.wikipedia.org/wiki/Argument_(complex_analysis)) and intensity(modulus squared) as well as pictures depicting the result of using an inverse Fourier transform on the aforementioned. The main take away here is, that simply applying an inverse Fourier transform to the phases or intensity patterns alone does not yield the original image. The picture listed as autocorrelation is simply the the inverse Fourier transform directly applied to the Intensity pattern, see [autocorrelation](https://en.wikipedia.org/wiki/Autocorrelation).

Note that in the fft images each single pixel represents a complex number.
The pixels color stores information about the phase while its 'brightness' represents the modulus squared.
To be precise the [hsl color scheme](https://en.wikipedia.org/wiki/HSL_and_HSV) is used where the phase of a complex number is used to define the the 'Hue'(H), the 'Saturation' (S) is set to 1 and the modulus squared defines the 'Lightness'(L).

Finally the main folder contains the reconstructed image (called reconstructed_density) as well as the computed support mask (i.e. non black regions of the initial image) as well as a false color movie depicting the complete phasing process.
The colors in the movie dont represent complex numbers since each reconstruction guess is just a real image, they are just used because to better see brightness differences in the images.

## Options
In the

    if __name__=='main':

part of the toy_phasing.py script the phasing routine options can be found.
There one is also able to specify other input images.
Keep in mind that always a pair of an input image and a fitting support mask is needed to start the phasing routine.
