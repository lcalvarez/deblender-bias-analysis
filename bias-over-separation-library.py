# -----------------------------------------------------------------------------
# Filename: BiasOverSeparationLibrary.py
# Author: Luis Alvarez
# Set of functions for running the deblender, simultaneous fitter
# and fits to true objects to compare results and perform
# bias analysis over a specific set of separations. 

# Import statements -----------------------------------------------------------
from __future__ import division
import galsim
import numpy as np
import lmfit
import deblend
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import triangle
import seaborn as sb
import csv
import shutil

# Function Definitions --------------------------------------------------------

# Create a galaxy with a sersic profile and optional psf to the image. 
def create_galaxy(flux, hlr, e1, e2, x0, y0, galtype_gal=galsim.Sersic, sersic_index=0.5,
                  psf_flag=False, psf_type=galsim.Moffat, beta=3, size_psf=1, flux_psf=1,
                  x_len=100, y_len=100, scale=0.2, method='fft', seed=None,
                  verbose=False, max_fft_size=10000, return_obj=False):
                      
    """ Helper function to create galsim objects
    
    Keyword Arguments:
    
    flux -- flux count for object
    hlr -- half-light radius in arcseconds for object
    e1 -- real component of the shear for object
    e2 -- imaginary component of the shear for object
    x0 -- x component of the centroid for object
    y0 -- y component of the centroid for object
    galtype_gal -- function definition of object
    sersic_index -- sersic index for object (Default 0.5) (Gaussian)
    psf_flag -- boolean indicating whether profiles are psf convolved (Default False)
    psf_type -- Type of psf to use (Recommended Moffat as parameters default for Moffat)
    beta -- parameter for Moffat profile (Default 3)
    size_psf -- size of the psf (Default 1)
    flux_psf -- flux of the psf profile (Default 1) (Should stay 1) 
    x_len -- horizontal dimension of images (Default 100)
    y_len -- vertical dimension of images (Default 100)
    scale -- scale of images in arcsec/pixel (Default 0.2)
    method -- method of creating objects (Default 'fft')
    seed -- galsim seed object (Default None)
    verbose -- boolean indicating printing status to terminal (Default False)
    max_fft_size -- integer for updating gsparams (Default 100000)
    return_obj -- boolean indicating whether galsim model is returned

    Returns:
    
    Galsim image or model 

    """
    
                  
    big_fft_params = galsim.GSParams(maximum_fft_size=max_fft_size)
    
    if verbose:
        print "\nPostage Stamp is", x_len, "by", y_len, "with\na scale of", scale,"\"/Pixel"    
        
    if galtype_gal is galsim.Sersic:
        assert sersic_index != 0
        if verbose:        
            if sersic_index == 0.5:
                print "\nThe object drawn is a gaussian with n = 0.5" 
        # Apply shearing and shifting of image
        gal = galtype_gal(n=sersic_index, half_light_radius=hlr, flux=flux, gsparams=big_fft_params)
        gal = gal.shear(g1=e1, g2=e2)
        gal = gal.shift(x0,y0)
        if return_obj == True:
            return gal
        image = galsim.ImageD(x_len, y_len, scale=scale)
        # Convolve with PSF
        if psf_flag:
            psf_gal = convolve_with_psf(gal, beta=beta, size_psf=size_psf, psf_type=psf_type, flux_psf=flux_psf,
                                        verbose=verbose, max_fft_size=max_fft_size)
            if method == 'fft':
                image = psf_gal.drawImage(image=image, method=method)
            else:
                image = psf_gal.drawImage(image=image, method=method,rng=seed)
            return image
        else:
            if method == 'fft':
                image = gal.drawImage(image=image, method=method)
            else:
                image = gal.drawImage(image=image, method=method,rng=seed)
            return image    
        
    else:
        raise ValueError("Not using a sersic profile for the object.")

def convolve_with_psf(gal, beta, size_psf, psf_type=galsim.Moffat, flux_psf=1, 
                      verbose=False, max_fft_size=100000):
    """ Helper function to convolve objects with Moffat PSF"""
    big_fft_params = galsim.GSParams(maximum_fft_size=max_fft_size)
    if verbose:
        print "Using a psf with beta =", beta,"and size = ", size_psf," \"" 
    psf = psf_type(beta=beta, fwhm=size_psf, flux=flux_psf, gsparams=big_fft_params)
    psf_gal = galsim.Convolve([gal,psf])
    return psf_gal

def add_noise(image, noise_type=galsim.PoissonNoise, seed=None, sky_level=0):
    """ Helper function to add Poisson and Sky noise to an image. """
    if noise_type is galsim.PoissonNoise:
        image.addNoise(noise_type(sky_level=sky_level,rng=seed))
        return image
    else:
        raise ValueError("Not using poisson noise in your image.")
        
def residual_1_obj(param, data_image, sky_level, x_len, y_len, pixel_scale, 
                   galtype,n,
                   psf_flag,beta,fwhm_psf):
                       
    """ Residual function for use in lmfit for one object fitting.
    
    Keyword Arguments:
    
    param -- lmfit parameters 
    data_image -- target image array
    sky_level -- sky
    x_len -- horizontal dimension of images
    y_len -- vertical dimension of images
    pixel_scale -- scale of images in arcsec/pixel
    galtype -- function definition of object a
    n -- sersic index of object
    psf_flag -- boolean indicating whether profiles are psf convolved
    beta -- parameter for Moffat profile
    fwhm_psf -- FWHM of PSF
    
    Returns:

    Least squares residual of data and model.
    
    """
                       
    assert galtype != None
    
    # Lmfit parameters
    flux = param['flux'].value
    hlr = param['hlr'].value
    e1 = param['e1'].value
    e2 = param['e2'].value
    x0 = param['x0'].value
    y0 = param['y0'].value
    
    # Create model image
    image = create_galaxy(flux,hlr,e1,e2,x0,y0,galtype_gal=galtype,sersic_index=n,
                          x_len=x_len,y_len=y_len,scale=pixel_scale,
                          psf_flag=psf_flag,beta=beta,size_psf=fwhm_psf)
    
    # Return residuals
    if sky_level > 10:        
        return (data_image-image.array).ravel()/np.sqrt(sky_level + image.array).ravel()
    else:
        return (data_image-image.array).ravel()

def residual_func_simple(param, data_image, sky_level, x_len, y_len, pixel_scale, 
                         galtype_a,n_a,galtype_b,n_b,
                         psf_flag,beta,fwhm_psf):
                             
    """ Residual function for use in lmfit for simultaneous fitting.
    
    Keyword Arguments:
    
    param -- lmfit parameters 
    data_image -- target galsim image (image.array contains array of pixels)
    sky_level -- mean count of electrons from sky for full-stack image
    x_len -- horizontal dimension of images
    y_len -- vertical dimension of images
    pixel_scale -- scale of images in arcsec/pixel
    galtype_a -- function definition of object a
    n_a -- sersic index of object a
    galtype_b -- function definition of object b
    n_b -- sersic index of object b
    psf_flag -- boolean indicating whether profiles are psf convolved
    beta -- parameter for Moffat profile
    fwhm_psf -- FWHM of PSF 

    Returns:

    Least squares residual of data and model.   
    
    """
        
    assert galtype_a != None
    assert galtype_b != None
    
    # Parameters for lmfit
    flux_a = param['flux_a'].value
    hlr_a = param['hlr_a'].value
    e1_a = param['e1_a'].value
    e2_a = param['e2_a'].value
    x0_a = param['x0_a'].value
    y0_a = param['y0_a'].value

    flux_b = param['flux_b'].value
    hlr_b = param['hlr_b'].value
    e1_b = param['e1_b'].value
    e2_b = param['e2_b'].value
    x0_b = param['x0_b'].value
    y0_b = param['y0_b'].value
    
    # Create model 
    image_a = create_galaxy(flux_a,hlr_a,e1_a,e2_a,x0_a,y0_a,galtype_gal=galtype_a,sersic_index=n_a,
                            x_len=x_len,y_len=y_len,scale=pixel_scale,
                            psf_flag=psf_flag,beta=beta,size_psf=fwhm_psf)
                            
    image_b = create_galaxy(flux_b,hlr_b,e1_b,e2_b,x0_b,y0_b,galtype_gal=galtype_b,sersic_index=n_b,
                            x_len=x_len,y_len=y_len,scale=pixel_scale,
                            psf_flag=psf_flag,beta=beta,size_psf=fwhm_psf)
    
    # Sum models
    image = image_a + image_b
    
    # Return residual
    if sky_level > 10:        
        return (data_image-image).array.ravel()/np.sqrt(sky_level + image.array).ravel()
    else:
        return (data_image-image).array.ravel()
        
def run_2_galaxy_full_params_simple(flux_a,hlr_a,e1_a,e2_a,x0_a,y0_a,n_a,
                                    flux_b,hlr_b,e1_b,e2_b,x0_b,y0_b,n_b,
                                    psf_flag,beta,fwhm_psf,
                                    x_len,y_len,pixel_scale,galtype_a,galtype_b,seed_a,seed_b,seed_p,
                                    add_noise_flag,sky_level,
                                    method,factor_init):
                                        
    """ Function that deblends target image and fits to child objects
    as well as true objects, returning estimates for both methods.
    
    Keyword Arguments:
    
    flux_a -- flux count for object a
    hlr_a -- half-light radius in arcseconds for object a
    e1_a -- real component of the shear for object a
    e2_a -- imaginary component of the shear for object a
    x0_a -- x component of the centroid for object a
    y0_a -- y component of the centroid for object a
    n_a -- sersic index for object a
    flux_b -- flux count for object b
    hlr_b -- half-light radius in arcseconds for object b
    e1_b -- real component of the shear for object b
    e2_b -- imaginary component of the shear for object b
    x0_b -- x component of the centroid for object b
    y0_b -- y component of the centroid for object b
    n_b -- sersic index for object b
    psf_flag -- boolean indicating whether profiles are psf convolved
    beta -- parameter for Moffat profile
    fwhm_psf -- FWHM of PSF
    x_len -- horizontal dimension of images
    y_len -- vertical dimension of images
    pixel_scale -- scale of images in arcsec/pixel
    galtype_a -- function definition of object a
    galtype_b -- function definition of object b
    seed (a-p) -- galsim seed objects
    add_noise_flag -- boolean indicating noisy images
    sky_level -- counts from sky (texp*sbar)
    method -- method in which to generate objects
    factor_init -- scalar to multiply initial guess

    Returns:
        
    image_no_noise -- target image without noise
    image_noise -- image with noise (both equivalent if no noise added)
    result -- lmfit minimizer object of image parameter estimates    
    
    """

    # Create target images and sum
    image_a = create_galaxy(flux_a,hlr_a,e1_a,e2_a,x0_a,y0_a,galtype_gal=galtype_a,sersic_index=n_a,
                            x_len=x_len,y_len=y_len,scale=pixel_scale,
                            psf_flag=psf_flag, beta=beta, size_psf=fwhm_psf,
                            method=method,seed=seed_a)
                                
    image_b = create_galaxy(flux_b,hlr_b,e1_b,e2_b,x0_b,y0_b,galtype_gal=galtype_b,sersic_index=n_b,
                            x_len=x_len,y_len=y_len,scale=pixel_scale,
                            psf_flag=psf_flag,beta=beta,size_psf=fwhm_psf,
                            method=method,seed=seed_b)
    
    # Add noise to image
    image_no_noise = image_a + image_b                        
    image = image_a + image_b
    if add_noise_flag:
        image_noise = add_noise(image,seed=seed_p,sky_level=sky_level)
        image = image_noise
    else:
        image_noise = image
    
    # -----------------------------------------------------------------------    
    # Estimate the parameters of the image.
    
    # Define some initial guess and insert into lmfit object for galaxy one and two
    # Keep ellipticity search within a bounding box to conserve magnitude
    # less than or equivalent to unity     
    lim = 1/np.sqrt(2)
    p0 = factor_init*np.array([flux_a,hlr_a,e1_a,e2_a,x0_a,y0_a,
                               flux_b,hlr_b,e1_b,e2_b,x0_b,y0_b])
    parameters = lmfit.Parameters()
    parameters.add('flux_a', value=p0[0])
    parameters.add('hlr_a', value=p0[1])
    parameters.add('e1_a',value=p0[2],min=-lim,max=lim)
    parameters.add('e2_a',value=p0[3],min=-lim,max=lim)    
    parameters.add('x0_a',value=p0[4])
    parameters.add('y0_a',value=p0[5])
    
    parameters.add('flux_b', value=p0[6])
    parameters.add('hlr_b', value=p0[7])
    parameters.add('e1_b',value=p0[8],min=-lim,max=lim)
    parameters.add('e2_b',value=p0[9],min=-lim,max=lim)
    parameters.add('x0_b',value=p0[10])
    parameters.add('y0_b',value=p0[11])
    
    
    # Extract params that minimize the difference of the data from the model.
    result = lmfit.minimize(residual_func_simple, parameters, args=(image, sky_level, x_len, y_len, pixel_scale, galtype_a, n_a, galtype_b, n_b,
                                                                    psf_flag, beta, fwhm_psf))                                   
                                                                      
    return image_no_noise, image_noise, result

# Create a blend, deblend, then estimate ellipticity of deblended objects and true objects.
def deblend_estimate(flux_a,hlr_a,e1_a,e2_a,x0_a,y0_a,n_a,
                     flux_b,hlr_b,e1_b,e2_b,x0_b,y0_b,n_b,
                     truth,
                     func, seed_1, seed_2, seed_3,
                     pixel_scale, x_len, y_len,
                     add_noise_flag, sky_level,
                     psf_flag, beta, fwhm_psf,
                     method,
                     factor_init,
                     path,
                     run,
                     plot=False):
                         
    """ Function that deblends target image and fits to child objects
    as well as true objects, returning estimates for both methods.
    
    Keyword Arguments:
    
    flux_a -- flux count for object a
    hlr_a -- half-light radius in arcseconds for object a
    e1_a -- real component of the shear for object a
    e2_a -- imaginary component of the shear for object a
    x0_a -- x component of the centroid for object a
    y0_a -- y component of the centroid for object a
    n_a -- sersic index for object a
    flux_b -- flux count for object b
    hlr_b -- half-light radius in arcseconds for object b
    e1_b -- real component of the shear for object b
    e2_b -- imaginary component of the shear for object b
    x0_b -- x component of the centroid for object b
    y0_b -- y component of the centroid for object b
    n_b -- sersic index for object b
    truth -- truth value array
    func -- galsim function definitions for objects
    seed (1-3) -- galsim seed objects
    pixel_scale -- scale of images in arcsec/pixel
    x_len -- horizontal dimension of images
    y_len -- vertical dimension of images
    add_noise_flag -- boolean indicating noisy images
    sky_level -- mean count of electrons from sky for full-stack image
    psf_flag -- boolean indicating whether profiles are psf convolved
    beta -- parameter for Moffat profile
    fwhm_psf -- FWHM of PSF
    method -- method in which to generate objects
    path -- directory in which to store lmfit report
    run -- current run number
    factor_init -- scalar to multiply initial guesspath    
    plot -- boolean indicating whether to show plots actively (Default False)

    Returns:

    results_deblend -- pandas Series of raw estimates from deblender
    results_true -- pandas Series of raw estimates from true fitter
    children -- children objects returned by deblender
    
    """
             
    # Rename parameters for use in previously defined functions
    sersic_func = func

    # Failures of fit
    failures = {'deblended_a':[],'deblended_b':[],'unblended_a':[],'unblended_b':[]}

    # Create the targets objects and sum them
    image_a = create_galaxy(flux_a,hlr_a,e1_a,e2_a,x0_a,y0_a,galtype_gal=func,sersic_index=n_a,
                            x_len=x_len,y_len=y_len,scale=pixel_scale,
                            psf_flag=psf_flag,beta=beta,size_psf=fwhm_psf,
                            method=method, seed=seed_1)

    image_b = create_galaxy(flux_b,hlr_b,e1_b,e2_b,x0_b,y0_b,galtype_gal=func,sersic_index=n_b,
                            x_len=x_len,y_len=y_len,scale=pixel_scale,
                            psf_flag=psf_flag,beta=beta,size_psf=fwhm_psf,
                            method=method, seed=seed_2)


    tot_image = image_a + image_b
    
    # Add noise 
    if add_noise_flag:
        image_noise = add_noise(tot_image,seed=seed_3,sky_level=sky_level)
    else:
        image_noise = np.copy(tot_image.array)
        

    # Deblend the resulting blend
    peak1 = (x0_a,y0_a)
    peak2 = (x0_b,y0_b)
    peaks_pix = [[p1/pixel_scale for p1 in peak1],
                 [p2/pixel_scale for p2 in peak2]]

    templates, template_fractions, children = deblend.deblend(image_noise.array, peaks_pix)

    if np.isnan(np.any(children[0])) or np.isnan(np.any(children[1])):
        raise ValueError("NaN's in deblended images.")

    # Now we can run the fitter to estimate the parameters of the children
    # -----------------------------------------------------------------------
    # Estimate the parameters of the image.

    # Define some initial guess and insert into lmfit object for galaxy one and two
    # Keep ellipticity search within a bounding box to conserve magnitude
    # less than or equivalent to unity     
    lim = 1/np.sqrt(2)
    p0_a = factor_init*np.array([flux_a,hlr_a,e1_a,e2_a,x0_a,y0_a])
    p0_b = factor_init*np.array([flux_b,hlr_b,e1_b,e2_b,x0_b,y0_b])

    parameters_a = lmfit.Parameters()
    parameters_a.add('flux', value=p0_a[0])
    parameters_a.add('hlr', value=p0_a[1])
    parameters_a.add('e1',value=p0_a[2],min=-lim,max=lim)
    parameters_a.add('e2',value=p0_a[3],min=-lim,max=lim)
    parameters_a.add('x0',value=p0_a[4])
    parameters_a.add('y0',value=p0_a[5])

    parameters_b = lmfit.Parameters()
    parameters_b.add('flux', value=p0_b[0])
    parameters_b.add('hlr', value=p0_b[1])
    parameters_b.add('e1',value=p0_b[2],min=-lim,max=lim)
    parameters_b.add('e2',value=p0_b[3],min=-lim,max=lim)
    parameters_b.add('x0',value=p0_b[4])
    parameters_b.add('y0',value=p0_b[5])


    # Extract params that minimize the difference of the data from the model.

    result_a = lmfit.minimize(residual_1_obj, parameters_a, args=(children[0], sky_level, x_len, y_len, pixel_scale, sersic_func, n_a,
                                                                  psf_flag, beta, fwhm_psf))

    with open(path + '/lmfit_result_deblended_object_a.txt','a+') as a:
        a.write('Trial ' + str(run) + '\n\n')
        a.write(lmfit.fit_report(result_a))
    if not result_a.success:
        failures['deblended_a'].append(run)

    result_b = lmfit.minimize(residual_1_obj, parameters_b, args=(children[1], sky_level, x_len, y_len, pixel_scale, sersic_func, n_a,
                                                                  psf_flag, beta, fwhm_psf))

    with open(path + '/lmfit_result_deblended_object_b.txt','a+') as b:
        b.write('Trial ' + str(run) + '\n\n')
        b.write(lmfit.fit_report(result_b))
    if not result_b.success:
        failures['deblended_b'].append(run)

    # Plot the data if necessary
    if plot != False:
        gs = gridspec.GridSpec(2,8)
        fig = plt.figure(figsize=(15,11))
        sh = 0.8
        plt.suptitle('True Objects vs Deblended Objects')
        ax1 = fig.add_subplot(gs[0,0:2])
        a = ax1.imshow(image_a.array,interpolation='none',origin='lower'); plt.title('Object A'); plt.colorbar(a,shrink=sh)
        ax2 = fig.add_subplot(gs[1,0:2])
        b = ax2.imshow(image_b.array,interpolation='none',origin='lower'); plt.title('Object B'); plt.colorbar(b,shrink=sh)
        ax3 = fig.add_subplot(gs[0,2:4])
        c = ax3.imshow(children[0],interpolation='none',origin='lower'); plt.title('Child A'); plt.colorbar(c,shrink=sh)
        ax4 = fig.add_subplot(gs[1,2:4])
        d = ax4.imshow(children[1],interpolation='none',origin='lower'); plt.title('Child B'); plt.colorbar(d,shrink=sh)
        ax5 = fig.add_subplot(gs[:,4:])
        e = ax5.imshow(tot_image.array,interpolation='none',origin='lower'); plt.title('Original Blend'); plt.colorbar(e,shrink=sh)
        plt.show()

    # Store the results in an array
    results_deblend = pd.Series(np.array([result_a.params['flux'].value,
                                          result_a.params['hlr'].value,
                                          result_a.params['e1'].value,
                                          result_a.params['e2'].value,
                                          result_a.params['x0'].value,
                                          result_a.params['y0'].value,
                                          result_b.params['flux'].value,
                                          result_b.params['hlr'].value,
                                          result_b.params['e1'].value,
                                          result_b.params['e2'].value,
                                          result_b.params['x0'].value,
                                          result_b.params['y0'].value]),
                                          index=['flux_a','hlr_a','e1_a','e2_a','x0_a','y0_a',
                                                 'flux_b','hlr_b','e1_b','e2_b','x0_b','y0_b'])

    # Now estimate the parameters for the true objects
    p0_a_t = (truth['flux_a'],truth['hlr_a'],truth['e1_a'],truth['e2_a'],truth['x0_a'],truth['y0_a'])
    p0_b_t = (truth['flux_b'],truth['hlr_b'],truth['e1_b'],truth['e2_b'],truth['x0_b'],truth['y0_b'])

    # Create the target images and sum them
    image_a_t = create_galaxy(p0_a_t[0],p0_a_t[1],p0_a_t[2],p0_a_t[3],p0_a_t[4],p0_a_t[5],galtype_gal=func,sersic_index=n_a,
                              x_len=x_len,y_len=y_len,scale=pixel_scale,
                              psf_flag=psf_flag, beta=beta, size_psf=fwhm_psf,
                              method=method, seed=seed_1)

    image_b_t = create_galaxy(p0_b_t[0],p0_b_t[1],p0_b_t[2],p0_b_t[3],p0_b_t[4],p0_b_t[5],galtype_gal=func,sersic_index=n_b,
                              x_len=x_len,y_len=y_len,scale=pixel_scale,
                              psf_flag=psf_flag, beta=beta, size_psf=fwhm_psf,
                              method=method, seed=seed_2)

    # Add noise
    if add_noise_flag:
        image_a_t = add_noise(image_a_t,seed=seed_3,sky_level=sky_level)
        image_b_t = add_noise(image_b_t,seed=seed_3,sky_level=sky_level)

    parameters_a = lmfit.Parameters()
    parameters_a.add('flux', value=p0_a_t[0])
    parameters_a.add('hlr', value=p0_a_t[1])
    parameters_a.add('e1',value=p0_a_t[2],min=-lim,max=lim)
    parameters_a.add('e2',value=p0_a_t[3],min=-lim,max=lim)
    parameters_a.add('x0',value=p0_a_t[4])
    parameters_a.add('y0',value=p0_a_t[5])

    parameters_b = lmfit.Parameters()
    parameters_b.add('flux', value=p0_b_t[0])
    parameters_b.add('hlr', value=p0_b_t[1])
    parameters_b.add('e1',value=p0_b_t[2],min=-lim,max=lim)
    parameters_b.add('e2',value=p0_b_t[3],min=-lim,max=lim)
    parameters_b.add('x0',value=p0_b_t[4])
    parameters_b.add('y0',value=p0_b_t[5])

    # Now estimate the shape for each true object
    result_a_true = lmfit.minimize(residual_1_obj, parameters_a, args=(image_a_t.array, sky_level, x_len, y_len, pixel_scale, sersic_func, n_a,
                                                                       psf_flag, beta, fwhm_psf))

    with open(path + '/lmfit_result_unblended_object_a.txt','a+') as c:
        c.write('Trial ' + str(run) + '\n\n')
        c.write(lmfit.fit_report(result_a_true))
    if not result_a_true.success:
        failures['unblended_a'].append(run)

    result_b_true = lmfit.minimize(residual_1_obj, parameters_b, args=(image_b_t.array, sky_level, x_len, y_len, pixel_scale, sersic_func, n_b,
                                                                       psf_flag, beta, fwhm_psf))

    with open(path + '/lmfit_result_unblended_object_b.txt','a+') as d:
        d.write('Trial ' + str(run) + '\n\n')
        d.write(lmfit.fit_report(result_b_true))
    if not result_b_true.success:
        failures['unblended_b'].append(run)
                                                                       
    # Store the results
    results_true = pd.Series(np.array([result_a_true.params['flux'].value,
                                       result_a_true.params['hlr'].value,
                                       result_a_true.params['e1'].value,
                                       result_a_true.params['e2'].value,
                                       result_a_true.params['x0'].value,
                                       result_a_true.params['y0'].value,
                                       result_b_true.params['flux'].value,
                                       result_b_true.params['hlr'].value,
                                       result_b_true.params['e1'].value,
                                       result_b_true.params['e2'].value,
                                       result_b_true.params['x0'].value,
                                       result_b_true.params['y0'].value]),
                                       index=['flux_a','hlr_a','e1_a','e2_a','x0_a','y0_a',
                                              'flux_b','hlr_b','e1_b','e2_b','x0_b','y0_b'])

    return results_deblend, results_true, children, failures


def rearrange_lmfit_2obj(result):
    """ Helper function to rearrange lmfit dictionary of estimates. """
    arr = np.array([result.params['flux_a'].value,result.params['hlr_a'].value,result.params['e1_a'].value,result.params['e2_a'].value,result.params['x0_a'].value,result.params['y0_a'].value,
                    result.params['flux_b'].value,result.params['hlr_b'].value,result.params['e1_b'].value,result.params['e2_b'].value,result.params['x0_b'].value,result.params['y0_b'].value])
    arr = pd.Series(arr,index=['flux_a','hlr_a','e1_a','e2_a','x0_a','y0_a',
                               'flux_b','hlr_b','e1_b','e2_b','x0_b','y0_b'])
    return arr
        
def obtain_stats(results,runs):
    """ Helper function to obtains significant statistics. """
    data = pd.DataFrame(np.array([np.mean(results),np.std(results),np.std(results)/np.sqrt(runs)]),columns=results.columns)
    data.index = [r'$\bar\mu$',r'$\sigma$', r'$\sigma_{\mu}$']
    return data    

def save_data(path,results_deblend,results_true,results_sim,identifier,stats):
    """ Helper function to save data to directory. """

    if not stats:
        if identifier == 'raw':
            dble = path + '/results_deblend.csv'
            with open(dble,'a') as f:
                results_deblend.to_csv(f)
            tru = path + '/results_true.csv'
            with open(tru,'a') as f:
                results_true.to_csv(f)
            sim = path + '/results_sim.csv'
            with open(sim,'a') as f:
                results_sim.to_csv(f)
        elif identifier == 'resid':
            dble = path + '/residual_deblend.csv'
            with open(dble,'a') as f:
                results_deblend.to_csv(f)
            tru = path + '/residual_true.csv'
            with open(tru,'a') as f:
                results_true.to_csv(f)
            sim = path + '/residual_sim.csv'
            with open(sim,'a') as f:
                results_sim.to_csv(f)
    else:
        if identifier == 'raw':
            dble = path + '/results_deblend_stats.csv'
            with open(dble,'a') as f:
                results_deblend.to_csv(f)
            tru = path + '/results_true_stats.csv'
            with open(tru,'a') as f:
                results_true.to_csv(f)
            sim = path + '/results_sim_stats.csv'
            with open(sim,'a') as f:
                results_sim.to_csv(f)
        elif identifier == 'resid':
            dble = path + '/residual_deblend_stats.csv'
            with open(dble,'a') as f:
                results_deblend.to_csv(f)
            tru = path + '/residual_true_stats.csv'
            with open(tru,'a') as f:
                results_true.to_csv(f)
            sim = path + '/residual_sim_stats.csv'
            with open(sim,'a') as f:
                results_sim.to_csv(f)
            
        
def run_batch(num_trials,
              func,
              seed_1,seed_2,seed_3,
              seed_4,seed_5,seed_6,
              image_params,
              obj_a,obj_b,method,
              sky_info,
              psf_info,
              mod_val,est_centroid,randomize,
              path):
                  
    """ Function that runs num_trials over the deblender,
    the simultaneous fitter, and the true fitter to return
    information on runs.
    
    Keyword Arguments:
    
    num_trial_arr -- the number of trials to run
    func -- galsim function definition of objects
    seed (1-6) -- galsim seed objects
    image_params -- list of image parameters for image [pixel_scale,x_len,y_len] 
    obj_a -- list of object a true parameters [flux_a,hlr_a,e1_a,e2_a,x0_a,y0_a,n_a]
    obj_b -- list of object b true parameters [flux_b,hlr_b,e1_b,e2_b,x0_b,y0_b,n_b]
    method -- method for generating objects in galsim
    sky_info -- list of parameters for sky noise [add_noise_flag,texp,sbar,sky_level]
    psf_info -- list of parameters for Moffat PSF [psf_flag,beta,fwhm_psf]
    mod_val -- value in which to mod number of trials for output of progress to terminal
    est_centroid -- boolean for using simultaneous fitting ouput to deblender
    randomize -- boolean for randomizing x,y coordinates by 1 pixel
    path -- directory in which to store lmfit reports
    
    Returns:
    
    results_deblend -- pandas DataFrame of raw estimates of parameters for deblender
    results_true -- pandas DataFrame of raw estimates of parameters for true fitter 
    results_sim -- pandas DataFrame of raw estimates of parameters for simultaneous fitter
    truth -- truth values of parameters
    x_y_coord -- randomized x,y coordinates of objects a and b  
    images -- array of deblender child objects
    
    """
    
    assert func == galsim.Sersic, "Not using a sersic profile"
    
    # Galsim function definitions
    sersic_func = func
    
    # Image properties
    pixel_scale, x_len, y_len = image_params
    
    # Parameters for object a
    flux_a, hlr_a, e1_a, e2_a, x0_a, y0_a, n_a = obj_a 
    
    # Parameters for object b
    flux_b, hlr_b, e1_b, e2_b, x0_b, y0_b, n_b = obj_b
    
    # Truth array
    truth = pd.Series(np.array([flux_a,hlr_a,e1_a,e2_a,x0_a,y0_a,
                                flux_b,hlr_b,e1_b,e2_b,x0_b,y0_b]),
                                index=['flux_a','hlr_a','e1_a','e2_a','x0_a','y0_a',
                                       'flux_b','hlr_b','e1_b','e2_b','x0_b','y0_b'])
    
    # Use LSST defined sky noise for r-band
    add_noise_flag, texp, sbar, sky_level = sky_info
    
    # psf properties
    psf_flag, beta, fwhm_psf = psf_info

    # Store the results of the fits to the deblended children
    results_deblend = []
    results_true = []
    images = []
    factor_init = 1
    results_sim = []
    x_y_coord = []
    failures_simult = []
    
    # Run through each trial
    for i in xrange(0,num_trials):
         
        # If randomizing, add +/- 1/2 pixel for 1 full pixel coverage
        if randomize == True:
            x0_a_r = x0_a + np.random.uniform(-pixel_scale/2,pixel_scale/2)
            x0_b_r = x0_b + np.random.uniform(-pixel_scale/2,pixel_scale/2)
            y0_a_r = y0_a + np.random.uniform(-pixel_scale/2,pixel_scale/2)
            y0_b_r = y0_b + np.random.uniform(-pixel_scale/2,pixel_scale/2)
            truth['x0_a'] = x0_a_r
            truth['y0_a'] = y0_a_r
            truth['x0_b'] = x0_b_r
            truth['y0_b'] = y0_b_r
        else:
            x0_a_r = x0_a
            x0_b_r = x0_b
            y0_a_r = y0_a
            y0_b_r = y0_b
            
        # Store the randomized information
        x_y_coord.append(pd.Series([x0_a_r,y0_a_r,x0_b_r,y0_b_r],index=['x0_a_r',
                                                                        'y0_a_r',
                                                                        'x0_b_r',
                                                                        'y0_b_r']))

        # First run the simultaneous fitter
        im_no_noise, im_noise, lm_results = run_2_galaxy_full_params_simple(flux_a,hlr_a,e1_a,e2_a,x0_a_r,y0_a_r,n_a,
                                                                            flux_b,hlr_b,e1_b,e2_b,x0_b_r,y0_b_r,n_b,
                                                                            psf_flag,beta,fwhm_psf,
                                                                            x_len,y_len,pixel_scale,sersic_func,sersic_func,seed_1,seed_2,seed_3,
                                                                            add_noise_flag,sky_level,
                                                                            method,factor_init)
        with open(path + '/simult_fit_report.txt','a+') as f:
            f.write('Trial ' + str(i) + '\n\n')
            f.write(lmfit.fit_report(lm_results))
        if not lm_results.success:
            failures_simult.append(i)
        
        # Store the results                                                                                              
        results_sim.append(rearrange_lmfit_2obj(lm_results))

        # Use the true centroids or those estimated from the simultaneous
        # fitter for the deblender
        if est_centroid == True:
            x0_a_est = np.copy(lm_results.values['x0_a'])
            x0_b_est = np.copy(lm_results.values['x0_b'])
            y0_a_est = np.copy(lm_results.values['y0_a'])
            y0_b_est = np.copy(lm_results.values['y0_b'])            
        else:
            x0_a_est = x0_a_r
            x0_b_est = x0_b_r
            y0_a_est = y0_a_r
            y0_b_est = y0_b_r
            
            
        # Run the deblender and fits to the true objects
        results_deb, results_tr, children, failures_dbl_tru = deblend_estimate(flux_a,hlr_a,e1_a,e2_a,x0_a_est,y0_a_est,n_a,
                                                                               flux_b,hlr_b,e1_b,e2_b,x0_b_est,y0_b_est,n_a,
                                                                               truth,
                                                                               sersic_func, seed_4, seed_5, seed_6,
                                                                               pixel_scale, x_len, y_len, 
                                                                               add_noise_flag, sky_level, 
                                                                               psf_flag, beta, fwhm_psf,
                                                                               method,
                                                                               factor_init,
                                                                               path,
                                                                               i)
                                                             
        # Store the results from the deblender and true fitter
        results_deblend.append(results_deb)
        results_true.append(results_tr)
        
        # Print to terminal for user information and store images
        if i%mod_val == 0:
            print i
            images.append([children,i])
    
    # Convert information to pandas DataFrames
    results_deblend = pd.DataFrame(results_deblend)
    results_true = pd.DataFrame(results_true)
    results_sim = pd.DataFrame(results_sim)
    x_y_coord = pd.DataFrame(x_y_coord)
    
    truth['x0_a'] = x0_a
    truth['y0_a'] = y0_a
    truth['x0_b'] = x0_b
    truth['y0_b'] = y0_b
    
    return results_deblend, results_true, results_sim, truth, x_y_coord, images, failures_dbl_tru, failures_simult
    
def run_over_separation(separation,
                        num_trial_arr,
                        func,
                        seed_arr,
                        image_params,
                        obj_a,obj_b,method,
                        sky_info,
                        psf_info,
                        mod_val,est_centroid,randomize,
                        directory,
                        create_tri_plots,
                        x_sep=True,y_sep=False,
                        right_diag=False,left_diag=False):
                            
    """ Loop through different separations, running many trials
    estimating the parameters of a blended image using simultaneous
    fitting, deblending, and the true objects. 
    
    Keyword Arguments:
    
    separation -- the list of separation values
    num_trial_arr -- the list of number of trials for each separation
    func -- galsim function definition of objects
    seed_arr -- list of galsim seed objects
    image_params -- list of image parameters for image [pixel_scale,x_len,y_len] 
    obj_a -- list of object a true parameters [flux_a,hlr_a,e1_a,e2_a,x0_a,y0_a,n_a]
    obj_b -- list of object b true parameters [flux_b,hlr_b,e1_b,e2_b,x0_b,y0_b,n_b]
    method -- method for generating objects in galsim
    sky_info -- list of parameters for sky noise [add_noise_flag,texp,sbar,sky_level]
    psf_info -- list of parameters for Moffat PSF [psf_flag,beta,fwhm_psf]
    mod_val -- value in which to mod number of trials for output of progress to terminal
    est_centroid -- boolean for using simultaneous fitting ouput to deblender
    randomize -- boolean for randomizing x,y coordinates by 1 pixel
    number_run -- path name in which to store run information
    create_tri_plots -- boolean for creating and storing triangle plots
    x_sep -- boolean for moving objects across horizontal axis (default True)
    y_sep -- boolean for moving objects across vertical axis (default False)
    right_diag -- boolean for moving objects across right diagonal (default False)
    left_diag -- boolean for moving objects across left diagonal (default False)
    
    Returns:
    
    Dictionary of pandas Series structures containing run information
    for deblender, simultaneous fitter, and true fitter over separation.
    
    means_e --  mean values for e1_a, e2_a, e1_b, and e2_b
    s_means_e -- error on mean values for e1_a, e2_a, e1_b, and e2_b
    means_flux_hlr -- mean values for flux_a, hlr_a, flux_b, and hlr_b
    s_means_flux_hlr -- error on mean values for flux_a, hlr_a, flux_b, and hlr_b
    means_x0_y0 -- mean values for x0_a, y0_a, x0_b, and y0_b
    s_means_x0_y0 -- error on mean values for x0_a, y0_a, x0_b, and y0_b
    
    """
    
    def create_resid_pd(results_deblend,
                        results_true,
                        results_sim,
                        truth,
                        x_y_coord,
                        randomize):
                        
        """ Helper function for creating residual data. """
                            
        if not randomize:
            resid_deblend = results_deblend - truth
            resid_true = results_true - truth
            resid_sim = results_sim - truth
        else:
            alt_truth = truth.copy()
            alt_truth['x0_a'] = 0; alt_truth['y0_a'] = 0;
            alt_truth['x0_b'] = 0; alt_truth['y0_b'] = 0;
            
            resid_deblend = results_deblend.copy()
            resid_deblend['x0_a'] = resid_deblend['x0_a'] - x_y_coord['x0_a_r']
            resid_deblend['y0_a'] = resid_deblend['y0_a'] - x_y_coord['y0_a_r']
            resid_deblend['x0_b'] = resid_deblend['x0_b'] - x_y_coord['x0_b_r']
            resid_deblend['y0_b'] = resid_deblend['y0_b'] - x_y_coord['y0_b_r']
            resid_deblend = resid_deblend - alt_truth
            
            resid_true = results_true.copy()
            resid_true['x0_a'] = resid_true['x0_a'] - x_y_coord['x0_a_r']
            resid_true['y0_a'] = resid_true['y0_a'] - x_y_coord['y0_a_r']
            resid_true['x0_b'] = resid_true['x0_b'] - x_y_coord['x0_b_r']
            resid_true['y0_b'] = resid_true['y0_b'] - x_y_coord['y0_b_r']
            resid_true = resid_true - alt_truth
            
            resid_sim = results_sim.copy()
            resid_sim['x0_a'] = resid_sim['x0_a'] - x_y_coord['x0_a_r']
            resid_sim['y0_a'] = resid_sim['y0_a'] - x_y_coord['y0_a_r']
            resid_sim['x0_b'] = resid_sim['x0_b'] - x_y_coord['x0_b_r']
            resid_sim['y0_b'] = resid_sim['y0_b'] - x_y_coord['y0_b_r']
            resid_sim = resid_sim - alt_truth
            
        return resid_deblend, resid_true, resid_sim
    
    def insert_data(dict1,dict2,dict3,dict4,sep,
                    data_dbl,data_true,data_simult,
                    identifier,index):
                    
        """ Helper function for inserting data from runs. """

        name_arr = ['Unblended', 'Simultaneously Fitted', 'Deblended']
        if identifier == 'e1,e2':
            dict1[str(sep)] = pd.Series(np.array([data_true['e1_a'][index],data_simult['e1_a'][index],data_dbl['e1_a'][index]]),index=name_arr)
            dict2[str(sep)] = pd.Series(np.array([data_true['e2_a'][index],data_simult['e2_a'][index],data_dbl['e2_a'][index]]),index=name_arr)
            dict3[str(sep)] = pd.Series(np.array([data_true['e1_b'][index],data_simult['e1_b'][index],data_dbl['e1_b'][index]]),index=name_arr)
            dict4[str(sep)] = pd.Series(np.array([data_true['e2_b'][index],data_simult['e2_b'][index],data_dbl['e2_b'][index]]),index=name_arr)

        elif identifier == 'flux,hlr':
            dict1[str(sep)] = pd.Series(np.array([data_true['flux_a'][index],data_simult['flux_a'][index],data_dbl['flux_a'][index]]),index=name_arr)
            dict2[str(sep)] = pd.Series(np.array([data_true['hlr_a'][index],data_simult['hlr_a'][index],data_dbl['hlr_a'][index]]),index=name_arr)
            dict3[str(sep)] = pd.Series(np.array([data_true['flux_b'][index],data_simult['flux_b'][index],data_dbl['flux_b'][index]]),index=name_arr)
            dict4[str(sep)] = pd.Series(np.array([data_true['hlr_b'][index],data_simult['hlr_b'][index],data_dbl['hlr_b'][index]]),index=name_arr)
            #dict1[str(sep)] = pd.Series(np.array([data_dbl['flux_a'][index],data_true['flux_a'][index],data_simult['flux_a'][index]]),index=name_arr)
            #dict2[str(sep)] = pd.Series(np.array([data_dbl['hlr_a'][index],data_true['hlr_a'][index],data_simult['hlr_a'][index]]),index=name_arr)
            #dict3[str(sep)] = pd.Series(np.array([data_dbl['flux_b'][index],data_true['flux_b'][index],data_simult['flux_b'][index]]),index=name_arr)
            #dict4[str(sep)] = pd.Series(np.array([data_dbl['hlr_b'][index],data_true['hlr_b'][index],data_simult['hlr_b'][index]]),index=name_arr)
            
        elif identifier == 'x0,y0':
            #dict1[str(sep)] = pd.Series(np.array([data_dbl['x0_a'][index],data_true['x0_a'][index],data_simult['x0_a'][index]]),index=name_arr)
            #dict2[str(sep)] = pd.Series(np.array([data_dbl['y0_a'][index],data_true['y0_a'][index],data_simult['y0_a'][index]]),index=name_arr)
            #dict3[str(sep)] = pd.Series(np.array([data_dbl['x0_b'][index],data_true['x0_b'][index],data_simult['x0_b'][index]]),index=name_arr)
            #dict4[str(sep)] = pd.Series(np.array([data_dbl['y0_b'][index],data_true['y0_b'][index],data_simult['y0_b'][index]]),index=name_arr)
            dict1[str(sep)] = pd.Series(np.array([data_true['x0_a'][index],data_simult['x0_a'][index],data_dbl['x0_a'][index]]),index=name_arr)
            dict2[str(sep)] = pd.Series(np.array([data_true['y0_a'][index],data_simult['y0_a'][index],data_dbl['y0_a'][index]]),index=name_arr)
            dict3[str(sep)] = pd.Series(np.array([data_true['x0_b'][index],data_simult['x0_b'][index],data_dbl['x0_b'][index]]),index=name_arr)
            dict4[str(sep)] = pd.Series(np.array([data_true['y0_b'][index],data_simult['y0_b'][index],data_dbl['y0_b'][index]]),index=name_arr)
                   
    def create_dict():
        """ Helper function for creating four dictionaries. """
        a = {}; b = {}; c = {}; d = {};
        return a,b,c,d
                   
    means_e1_a, means_e2_a, means_e1_b, means_e2_b = create_dict()
    
    s_means_e1_a, s_means_e2_a, s_means_e1_b, s_means_e2_b = create_dict()
    
    means_flux_a, means_hlr_a, means_flux_b, means_hlr_b = create_dict()
    
    s_means_flux_a, s_means_hlr_a, s_means_flux_b, s_means_hlr_b = create_dict()
    
    means_x0_a, means_y0_a, means_x0_b, means_y0_b = create_dict()
    
    s_means_x0_a, s_means_y0_a, s_means_x0_b, s_means_y0_b = create_dict()
    
    # Access each separation value from the array
    for i,sep in enumerate(separation):
        # Print to the terminal which separation the program is currently on
        print 'sep = ' + str(sep) + '\"'
        # Access the number of trials in the corresponding array
        num_trials = num_trial_arr[i]
        
        # Modify obj arrays corresponding to axis
        if x_sep and not y_sep:
            obj_a[4] = -sep/2
            obj_b[4] = sep/2
        elif not x_sep and y_sep:
            obj_a[5] = -sep/2
            obj_b[5] = sep/2
        elif x_sep and y_sep and right_diag:
            obj_a[4] = -np.cos(np.pi/4)*sep/2
            obj_a[5] = -np.sin(np.pi/4)*sep/2
            obj_b[4] = np.cos(np.pi/4)*sep/2
            obj_b[5] = np.sin(np.pi/4)*sep/2
        elif x_sep and y_sep and left_diag:
            obj_a[4] = -np.cos(np.pi/4)*sep/2
            obj_a[5] = np.sin(np.pi/4)*sep/2
            obj_b[4] = np.cos(np.pi/4)*sep/2
            obj_b[5] = -np.sin(np.pi/4)*sep/2

        # Create sub directory to save data from this separation            
        sub_sub_dir = '/sep:' + str(sep) + ';' + 'num_trials:' + str(num_trials)
        path = directory + sub_sub_dir
        os.mkdir(path)
        
        # Run the simultaneous fitter and the deblender and output the results
        results_deblend, results_true, results_sim, truth, x_y_coord, dbl_im, fail_dbl_tru, fail_simult = run_batch(num_trials,
                                                                                                                    func,
                                                                                                                    seed_arr[0],seed_arr[1],seed_arr[2],
                                                                                                                    seed_arr[3],seed_arr[4],seed_arr[5],
                                                                                                                    image_params,
                                                                                                                    obj_a,obj_b,method,
                                                                                                                    sky_info,
                                                                                                                    psf_info,
                                                                                                                    mod_val,est_centroid,randomize,
                                                                                                                    path)


        # Obtain the residuals using the truth values and the x,y coordinates 
        # if randomization was used.
        resid_deblend, resid_true, resid_sim = create_resid_pd(results_deblend,
                                                               results_true,
                                                               results_sim,
                                                               truth,
                                                               x_y_coord,
                                                               randomize)
        
        # Save raw data
        save_data(path,results_deblend,results_true,results_sim,'raw',False)
        save_data(path,resid_deblend,resid_true,resid_sim,'resid',False)
        with open(path + '/x_y_coord.csv','a') as f:
            x_y_coord.to_csv(f)
        
        # Obtain relevant stats
        data_dbl_raw = obtain_stats(results_deblend,num_trials)
        data_true_raw = obtain_stats(results_true,num_trials)
        data_simult_raw = obtain_stats(results_sim,num_trials)
        
        # Save statistics on raw data
        save_data(path,data_dbl_raw,data_true_raw,data_simult_raw,'raw',True)
        
        data_dbl_resid = obtain_stats(resid_deblend,num_trials)
        data_true_resid = obtain_stats(resid_true,num_trials)
        data_simult_resid = obtain_stats(resid_sim,num_trials)
        
        # Save statistics on residual data
        save_data(path,data_dbl_resid,data_true_resid,data_simult_resid,'resid',True)

        # Save failure data
        writer = csv.writer(open(path + '/failures_dbl_true.txt','wb'))
        for key, value in fail_dbl_tru.items():
            writer.writerow([key,value])

        with open(path + '/failures_simult_fit.txt','w+') as f:
            for item in fail_simult:
                f.write("%s\n" % item)

        
         # Save triangle plots
        if create_tri_plots:
            create_triangle_plots(path,sep,num_trials,
                                  results_deblend,data_dbl_raw,
                                  results_true,data_true_raw,
                                  results_sim,data_simult_raw,
                                  truth,
                                  x_y_coord,randomize,'raw')

            alt_truth = truth.copy()                     
            create_triangle_plots(path,sep,num_trials,
                                  resid_deblend,data_dbl_resid,
                                  resid_true,data_true_resid,
                                  resid_sim,data_simult_resid,
                                  alt_truth,
                                  x_y_coord,randomize,'resid')
                             
        # Save a random image from the set of deblended images
        save_image(path,results_deblend,dbl_im,np.copy(image_params),truth,sep,
                   psf_info)
        
        # Obtain the mean values with error on mean values
        # index variable refers to which row of information
        # index of 0 corresponds to mean values
        index = 0
        insert_data(means_e1_a,means_e2_a,means_e1_b,means_e2_b,sep,
                    data_dbl_resid,data_true_resid,data_simult_resid,
                    'e1,e2',index)
                    
        insert_data(means_flux_a,means_hlr_a,means_flux_b,means_hlr_b,sep,
                    data_dbl_resid,data_true_resid,data_simult_resid,
                    'flux,hlr',index)
                    
        insert_data(means_x0_a,means_y0_a,means_x0_b,means_y0_b,sep,
                    data_dbl_resid,data_true_resid,data_simult_resid,
                    'x0,y0',index)
                    
        # index of 2 corresponds to error on mean values
        index = 2
        insert_data(s_means_e1_a,s_means_e2_a,s_means_e1_b,s_means_e2_b,sep,
                    data_dbl_resid,data_true_resid,data_simult_resid,
                    'e1,e2',index)
                    
        insert_data(s_means_flux_a,s_means_hlr_a,s_means_flux_b,s_means_hlr_b,sep,
                    data_dbl_resid,data_true_resid,data_simult_resid,
                    'flux,hlr',index)
                    
        insert_data(s_means_x0_a,s_means_y0_a,s_means_x0_b,s_means_y0_b,sep,
                    data_dbl_resid,data_true_resid,data_simult_resid,
                    'x0,y0',index)             
    
    # Store the information in dictionaries
    means_e = {'means_e1_a':means_e1_a,'means_e2_a':means_e2_a,
             'means_e1_b':means_e1_b,'means_e2_b':means_e2_b}
             
    means_flux_hlr = {'means_flux_a':means_flux_a,'means_hlr_a':means_hlr_a,
                      'means_flux_b':means_flux_b,'means_hlr_b':means_hlr_b}
                      
    means_x0_y0 = {'means_x0_a':means_x0_a,'means_y0_a':means_y0_a,
                   'means_x0_b':means_x0_b,'means_y0_b':means_y0_b}
             
    s_means_e = {'s_means_e1_a':s_means_e1_a,'s_means_e2_a':s_means_e2_a,
                 's_means_e1_b':s_means_e1_b,'s_means_e2_b':s_means_e2_b}
                 
    s_means_flux_hlr = {'s_means_flux_a':s_means_flux_a,'s_means_hlr_a':s_means_hlr_a,
                        's_means_flux_b':s_means_flux_b,'s_means_hlr_b':s_means_hlr_b}
                        
    s_means_x0_y0 = {'s_means_x0_a':s_means_x0_a,'s_means_y0_a':s_means_y0_a,
                     's_means_x0_b':s_means_x0_b,'s_means_y0_b':s_means_y0_b}
                        
    return means_e, s_means_e, means_flux_hlr, s_means_flux_hlr, means_x0_y0, s_means_x0_y0
    
def join_info(directory,
              separation,
              num_trial_arr,
              func,
              seed_arr,
              image_params,
              obj_a,obj_b,method,
              sky_info,
              psf_info,
              mod_val,use_est_centroid,randomize,
              x_sep,y_sep,
              left_diag,right_diag,
              create_tri_plots):
                  
    """ Create a string of run information to write to Run_Information.txt.
          
    Keyword Arguments:
    
    directory -- the directory (string) in which to write Run_Information.txt
    separation -- the list of separation values
    num_trial_arr -- the list of number of trials for each separation
    func -- galsim function definition of objects
    seed_arr -- list of galsim seed objects
    image_params -- list of image parameters for image [pixel_scale,x_len,y_len] 
    obj_a -- list of object a true parameters [flux_a,hlr_a,e1_a,e2_a,x0_a,y0_a,n_a]
    obj_b -- list of object b true parameters [flux_b,hlr_b,e1_b,e2_b,x0_b,y0_b,n_b]
    method -- method for generating objects in galsim
    sky_info -- list of parameters for sky noise [add_noise_flag,texp,sbar,sky_level]
    psf_info -- list of parameters for Moffat PSF [psf_flag,beta,fwhm_psf]
    mod_val -- value in which to mod number of trials for output of progress to terminal
    est_centroid -- boolean for using simultaneous fitting ouput to deblender
    randomize -- boolean for randomizing x,y coordinates by 1 pixel
    x_sep -- boolean for moving objects across horizontal axis (default True)
    y_sep -- boolean for moving objects across vertical axis (default False)
    right_diag -- boolean for moving objects across right diagonal (default False)
    left_diag -- boolean for moving objects across left diagonal (default False)
    create_tri_plots -- boolean for creating and storing triangle plots
    method -- galsim method for creating objects

    Returns:
    
    String of run information.
    
    """
                
    if x_sep and y_sep and not right_diag and not left_diag: assert False, "Choose a diagonal."                                    
    if x_sep and y_sep: assert right_diag != left_diag, "Can't run through both diagonals of the image."                  
                  
    print "\nSaving data and images to the following directory: \"" + directory + '\"\n'                   
                  
    if x_sep and not y_sep:
        direction_str = 'x axis'
        print "Moving objects along " + direction_str + '\n'
    elif not x_sep and y_sep:
        direction_str = 'y axis'
        print "Moving objects along " + direction_str + '\n'
    elif x_sep and y_sep and right_diag:
        direction_str = 'right diagonal'
        print "Moving objects along " + direction_str + '\n'
    elif x_sep and y_sep and left_diag:
        direction_str = 'left diagonal'
        print "Moving objects along " + direction_str + '\n'
        
    if psf_info[0]: 
        print "Convolving objects with PSF\n"
    else:
        print "Not convolving objects with PSF\n"
    if use_est_centroid:
        print "Using simultaneous fitting output estimates of (x,y) for deblender\n"
    else:
        print "Using true (x,y) for deblender\n"
    if randomize:
        print "Randomizing centroids a pixel about median separation\n"
    else:
        "Not Randomizing centroids a pixel about median separation\n"
    if create_tri_plots:
        print "Creating triangle plots\n"
    else:
        print "Not creating triangle plots\n"

    print "Using " + method + " for sampling\n"
        
        
    sub_dir = ('x_y_prior = ' + str(not use_est_centroid) + '\n' +
              'randomized_x_y = ' + str(randomize) + '\n' +
              'sep = ' + str(separation) + ' (arcsec)' + '\n' +
              'num_trial_arr = ' + str(num_trial_arr) + ' Number of trials for each separation' + '\n' + 
              'seed_arr = ' + str(seed_arr) + ' Insert into galsim.BaseDeviate' + '\n' +
              'image_params = ' + str(image_params) + ' (pixel_scale,x_len,y_len) of image' + '\n' + 
              'obj_a_info = ' + str(obj_a) + ' (flux,hlr,e1,e2,x0,y0)' + '\n' +
              'obj_b_info = ' + str(obj_b) + ' (flux,hlr,e1,e2,x0,y0)' + '\n' +
              'sky_info = ' + str(sky_info) + ' (flag,texp,sbar,sky_level)' + '\n'
              'psf_info = ' + str(psf_info) + ' (flag,beta,fwhm)' + '\n' +
              'separation_axis = ' + direction_str + '\n' +
              'create_triangle_plots = ' + str(create_tri_plots) + '\n' +
              'method = ' + str(method) + '\n')
            
    return sub_dir
    
def create_read_me(info_str,dir_str):
    """ Create Run_Information.txt file storing Information of Run """
    try:
        os.mkdir(dir_str)
    except:
        shutil.rmtree(dir_str)
        os.mkdir(dir_str)
    file = open(dir_str + '/Run_Information.txt','w+')
    file.write(info_str)
    file.close()
    
def create_triangle_plots(path,sep,num_trials,
                          results_deblend,data_dbl,
                          results_true,data_true,
                          results_sim,data_sim,
                          truth,
                          x_y_coord,randomize,identifier):
                              
    """ Create and store triangle plots.
    
    Keyword Arguments:
    
    path -- directory to store information
    sep -- current separation
    num_trials -- number of trials for the run
    results_deblend -- raw estimates of parameters for deblender
    data_dbl -- significant statistics on results_deblend
    results_true -- raw estimates of parameters for true fits
    data_true -- significant statistics on results_true
    results_sim -- raw estimates of parameters for simultaneous fitter
    data_sim -- significant statistics on results_sim
    truth -- truth values of parameters
    x_y_coord -- randomized x,y coordinates of objects a and b
    randomize -- boolean indicating centroid randomization
    identifier -- string indicating raw or residual data.
    
    """
    
    # Use the same scaling for all three methods
    max_sigma = pd.Series(np.maximum(np.copy(data_dbl.values[1,:]),
                                     np.copy(data_sim.values[1,:])),
                          index=['flux_a','hlr_a','e1_a','e2_a','x0_a','y0_a',
                                 'flux_b','hlr_b','e1_b','e2_b','x0_b','y0_b']) 
                                                 
    # Create the extent arrays to use on triangle plots
    extents_true = create_extents(3,max_sigma,data_true.iloc[0,:],randomize)
    extents_sim = create_extents(3,max_sigma,data_sim.iloc[0,:],randomize)
    extents_dbl = create_extents(3,max_sigma,data_dbl.iloc[0,:],randomize)
    
    # If randomizing, concatenate true x,y data to estimates. 
    if randomize == True:
        true_tri = pd.concat([results_true,x_y_coord],axis=1)
        sim_tri = pd.concat([results_sim,x_y_coord],axis=1)
        dbl_tri = pd.concat([results_deblend,x_y_coord],axis=1)
        
        # Update the extents arrays accordingly
        extents_true = extents_true + [(x_y_coord['x0_a_r'].min(),x_y_coord['x0_a_r'].max()),
                                       (x_y_coord['y0_a_r'].min(),x_y_coord['y0_a_r'].max()),
                                       (x_y_coord['x0_b_r'].min(),x_y_coord['x0_b_r'].max()),
                                       (x_y_coord['y0_b_r'].min(),x_y_coord['y0_b_r'].max())]
        extents_sim = extents_sim + [(x_y_coord['x0_a_r'].min(),x_y_coord['x0_a_r'].max()),
                                       (x_y_coord['y0_a_r'].min(),x_y_coord['y0_a_r'].max()),
                                       (x_y_coord['x0_b_r'].min(),x_y_coord['x0_b_r'].max()),
                                       (x_y_coord['y0_b_r'].min(),x_y_coord['y0_b_r'].max())]
        extents_dbl = extents_dbl + [(x_y_coord['x0_a_r'].min(),x_y_coord['x0_a_r'].max()),
                                       (x_y_coord['y0_a_r'].min(),x_y_coord['y0_a_r'].max()),
                                       (x_y_coord['x0_b_r'].min(),x_y_coord['x0_b_r'].max()),
                                       (x_y_coord['y0_b_r'].min(),x_y_coord['y0_b_r'].max())]     
                                       
        rand_xy = pd.Series([truth['x0_a'],truth['y0_a'],
                             truth['x0_b'],truth['y0_b']],
                             index=['x0_a_r','y0_a_r',
                                    'x0_b_r','y0_b_r'])
        truth = truth.append(rand_xy)
    else: 
        true_tri = results_true
        sim_tri = results_sim
        dbl_tri = results_deblend
        
    # If plotting residual data, 0 is reference point
    if identifier == 'resid':
        truth[0:12] = 0        
    
    # Use seaborn formatting for correlation plots
    sb.set(style='darkgrid')
    cmap = sb.diverging_palette(220,10,as_cmap=True)
    
    if identifier == 'raw':
        print "Saving triangle and correlation plots for raw data.\n"
    else:
        print "Saving triangle and correlation plots for residual data.\n"

    
    # Produce and save the triangle plots. 
    triangle.corner(true_tri,labels=true_tri.columns,truths=truth.values,
                    show_titles=True,title_args={'fontsize':20},extents=extents_true)
    plt.suptitle('Triangle Plot for Fits to the True Objects\n for a Separation of ' + str(sep) + '\" and ' + str(num_trials) + ' Trials',fontsize=42)
    if identifier == 'raw':    
        plt.savefig(path + '/raw_true_fit_triangle_plot.png')
    else:
        plt.savefig(path + '/residual_true_fit_triangle_plot.png')
    plt.clf()
    plt.close()

    fig = plt.figure(figsize=(20,20))
    plt.suptitle('Correlation Plot for Fits to the True Objects\n for a Separation of ' + str(sep) + '\" and ' + str(num_trials) + ' Trials',fontsize=22)
    ax = fig.add_subplot()
    sb.corrplot(true_tri,cbar=True,cmap=cmap,ax=ax,sig_stars=False,diag_names=False)
    if identifier == 'raw':        
        plt.savefig(path + '/raw_true_fit_correlation_plot.png')
    else:
        plt.savefig(path + '/residual_true_fit_correlation_plot.png')
        
    plt.clf()
    plt.close()
    
    triangle.corner(sim_tri,labels=sim_tri.columns,truths=truth.values,
                    show_titles=True,title_args={'fontsize':20},extents=extents_sim)
    plt.suptitle('Triangle Plot for Simultaneous Fitting to the Blended Objects\n for a Separation of ' + str(sep) + '\" and ' + str(num_trials) + ' Trials',fontsize=42)
    if identifier == 'raw':    
        plt.savefig(path + '/raw_simult_fit_triangle_plot.png')
    else:
        plt.savefig(path + '/residual_simult_fit_triangle_plot.png')    
 
    plt.clf()
    plt.close()

    fig = plt.figure(figsize=(20,20))
    plt.suptitle('Correlation Plot for Simultaneous Fitting to the Blended Objects\n for a Separation of ' + str(sep) + '\" and ' + str(num_trials) + ' Trials',fontsize=22)
    ax = fig.add_subplot()
    sb.corrplot(sim_tri,cbar=True,cmap=cmap,ax=ax,sig_stars=False,diag_names=False)
    if identifier == 'raw':        
        plt.savefig(path + '/raw_simult_fit_correlation_plot.png')
    else:
        plt.savefig(path + '/residual_simult_fit_correlation_plot.png')    
    plt.clf()
    plt.close()
    
    triangle.corner(dbl_tri,labels=dbl_tri.columns,truths=truth.values,
                    show_titles=True,title_args={'fontsize':20},extents=extents_dbl)
    plt.suptitle('Triangle Plot for Fits to the Deblended Objects\n for a Separation of ' + str(sep) + '\" and ' + str(num_trials) + ' Trials',fontsize=42)

    if identifier == 'raw':    
        plt.savefig(path + '/raw_deblended_object_fit_triangle_plot.png')
    else:
        plt.savefig(path + '/residual_deblended_object_fit_triangle_plot.png')
    
    plt.clf()
    plt.close()

    fig = plt.figure(figsize=(20,20))
    plt.suptitle('Correlation Plot for Fits to the Deblended Objects\n for a Separation of ' + str(sep) + '\" and ' + str(num_trials) + ' Trials',fontsize=22)
    ax = fig.add_subplot()
    sb.corrplot(dbl_tri,cbar=True,cmap=cmap,ax=ax,sig_stars=False,diag_names=False)
    if identifier == 'raw':        
        plt.savefig(path + '/raw_deblended_object_fit_correlation_plot.png')
    else:
        plt.savefig(path + '/residual_deblended_object_fit_correlation_plot.png')    
    
    plt.clf()
    plt.close()
    
    if identifier == 'raw':
        print "Finished saving triangle and correlation plots for raw data.\n"
    else:
        print "Finished saving triangle and correlation plots for residual data.\n"
        
def create_extents(factor,max_sigma,truth,randomize):
    
    """ Helper function to create extents array for triangle plots. """

    flux_interval_a = max_sigma['flux_a']*factor
    hlr_interval_a = max_sigma['hlr_a']*factor
    e1_interval_a = max_sigma['e1_a']*factor
    e2_interval_a = max_sigma['e2_a']*factor
    x0_interval_a = max_sigma['x0_a']*factor
    y0_interval_a = max_sigma['y0_a']*factor
    
    flux_interval_b = max_sigma['flux_b']*factor
    hlr_interval_b = max_sigma['hlr_b']*factor
    e1_interval_b = max_sigma['e1_b']*factor
    e2_interval_b = max_sigma['e2_b']*factor
    x0_interval_b = max_sigma['x0_b']*factor
    y0_interval_b = max_sigma['y0_b']*factor
    
    extents = [(truth['flux_a']-flux_interval_a,truth['flux_a']+flux_interval_a),
               (truth['hlr_a']-hlr_interval_a,truth['hlr_a']+hlr_interval_a),
               (truth['e1_a']-e1_interval_a,truth['e1_a']+e1_interval_a),
               (truth['e2_a']-e2_interval_a,truth['e2_a']+e2_interval_a),
               (truth['x0_a']-x0_interval_a,truth['x0_a']+x0_interval_a),
               (truth['y0_a']-y0_interval_a,truth['y0_a']+y0_interval_a),
               (truth['flux_b']-flux_interval_b,truth['flux_b']+flux_interval_b),
               (truth['hlr_a']-hlr_interval_b,truth['hlr_b']+hlr_interval_b),
               (truth['e1_b']-e1_interval_b,truth['e1_b']+e1_interval_b),
               (truth['e2_b']-e2_interval_b,truth['e2_b']+e2_interval_b),
               (truth['x0_b']-x0_interval_b,truth['x0_b']+x0_interval_b),
               (truth['y0_b']-y0_interval_b,truth['y0_b']+y0_interval_b)]
               
    return extents

def create_bias_plot(path,separation,means,s_means,pixel_scale,
                     fs,leg_fs,min_offset,max_offset,psf_flag,identifier,
                     zoom):
                         
    """ Function to create bias plots of pairs of variables 
    for objects a and b
    
    Keyword Arguments:
    
    path -- directory to store information
    separation -- array of separation values
    means -- dictionary of pandas Series containing mean value information
    s_means -- dictionary of pandas Series containing error on mean value information 
    pixel_scale -- arcsecs per pixel parameter used in images
    fs -- fontsize for plots
    leg_fs -- fontsize for legends
    min_offset -- scalar to multiply minimum y values of bias
    max_offset -- scalar to multiply maximum y values of bias
    psf_flag -- boolean of psf usage
    identifier -- string of which pairs of variables to plot
    zoom -- boolean for scaling each plot independently thereby providing zoom effect
    """        

    assert len(separation) >= 2, "Separation array must be larger than 1"
         
    def obtain_min_max_df(df_1,df_2,df_3,df_4):
        """ Helper function for obtaining max and min values for scaling. """
        max_val = np.max(np.max(pd.concat([np.max(df_1.T),np.max(df_2.T),np.max(df_3.T),np.max(df_4.T)],axis=1)))
        min_val = np.min(np.min(pd.concat([np.min(df_1.T),np.min(df_2.T),np.min(df_3.T),np.min(df_4.T)],axis=1)))    
        return max_val, min_val
    
        
    def format_df(df,value_1,value_2,index):
        """ Helper function to format dataframes for plotting. """
        result = pd.concat([pd.DataFrame(np.array([np.NAN,np.NAN,np.NAN]),columns=[str(value_1)],index=index),
                            df,
                            pd.DataFrame(np.array([np.NAN,np.NAN,np.NAN]),columns=[str(value_2)],index=index)],
                           axis=1)
        return result
        
    def insert_subplot(ax,df,s_df,min_sep):
        """ Helper function to plot information on subplot. """
        alpha_arr = [0.99,0.99,0.99]
        linestyle_arr = ['','','']
        c_arr = ['b','g','r']
        z_order = [5,3,4]
        ms_arr = [2.3,3,2.3]
        for i,col_key in enumerate(df.T):
            domain = np.array(map(float,df.T[col_key].index.values)) + i*0.02
            ax.errorbar(domain,df.T[col_key].values,yerr=s_df.T[col_key].values,
                        alpha=alpha_arr[i],linestyle=linestyle_arr[i],
                        label=col_key,c=c_arr[i],marker='o',markersize=ms_arr[i],zorder=z_order[i])
                         
        ax.axhline(y=0,ls='--',c='k',zorder=2,lw=0.5)
        ax.axvline(x=min_sep,ls='--',c='c',zorder=1,lw=0.5)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5,-0.11),
                  prop={'size':leg_fs}, shadow=True, ncol=3, fancybox=True)
    

    # Print to terminal
    if not zoom:
        print '\nSaving Bias vs separation plot for ' + identifier +  '.\n'
    else:
        print '\nSaving Zoomed Bias vs separation plot for ' + identifier +  '.\n'
    
    # Access appropriate information according to identifier
    if identifier == 'e1,e2':
        df1_a = pd.DataFrame(means['means_e1_a'])
        df2_a = pd.DataFrame(means['means_e2_a'])
        df1_b = pd.DataFrame(means['means_e1_b'])
        df2_b = pd.DataFrame(means['means_e2_b'])
        
        s_df1_a = pd.DataFrame(s_means['s_means_e1_a'])
        s_df2_a = pd.DataFrame(s_means['s_means_e2_a'])
        s_df1_b = pd.DataFrame(s_means['s_means_e1_b'])
        s_df2_b = pd.DataFrame(s_means['s_means_e2_b'])
        
        title_arr = ['e1 On Object a', 'e2 On Object a', 'e1 On Object b', 'e2 On Object b']
        if psf_flag: 
            suptitle = 'Ellipticity Bias for Objects a and b\n vs Separation for PSF Convolved Profiles'
        else:
            suptitle = 'Ellipticity Bias for Objects a and b\n vs Separation for Profiles with Only Poisson Noise'
        
    if identifier == 'flux,hlr':
        df1_a = pd.DataFrame(means['means_flux_a'])
        df2_a = pd.DataFrame(means['means_hlr_a'])
        df1_b = pd.DataFrame(means['means_flux_b'])
        df2_b = pd.DataFrame(means['means_hlr_b'])
        
        s_df1_a = pd.DataFrame(s_means['s_means_flux_a'])
        s_df2_a = pd.DataFrame(s_means['s_means_hlr_a'])
        s_df1_b = pd.DataFrame(s_means['s_means_flux_b'])
        s_df2_b = pd.DataFrame(s_means['s_means_hlr_b'])
        
        title_arr = ['Flux On Object a', 'Hlr On Object a', 'Flux On Object b', 'Hlr On Object b']
        if psf_flag: 
            suptitle = 'Flux and Half-light Radius Bias for Objects a and b\n vs Separation for PSF Convolved Profiles'
        else:
            suptitle = 'Flux and Half-light Radius Bias for Objects a and b\n vs Separation for Profiles with Only Poisson Noise'

        
    if identifier == 'x0,y0':
        df1_a = pd.DataFrame(means['means_x0_a'])
        df2_a = pd.DataFrame(means['means_y0_a'])
        df1_b = pd.DataFrame(means['means_x0_b'])
        df2_b = pd.DataFrame(means['means_y0_b'])
        
        s_df1_a = pd.DataFrame(s_means['s_means_x0_a'])
        s_df2_a = pd.DataFrame(s_means['s_means_y0_a'])
        s_df1_b = pd.DataFrame(s_means['s_means_x0_b'])
        s_df2_b = pd.DataFrame(s_means['s_means_y0_b'])
        
        title_arr = ['x0 On Object a', 'y0 On Object a', 'x0 On Object b', 'y0 On Object b']

        if psf_flag: 
            suptitle = 'Centroid Bias for Objects a and b\n vs Separation for PSF Convolved Profiles'
        else:
            suptitle = 'Centroid Bias for Objects a and b\n vs Separation for Profiles with Only Poisson Noise'

    suptitle = suptitle + '\n (Vertical Cyan Line Denotes Loss of Saddle Region in Image)'

    if psf_flag:
        min_sep = 1.6
    else:
        min_sep = 1.4    
    
    # Create the x-limits of plots
    x_min = np.min(separation) - pixel_scale
    x_max = np.max(separation) + pixel_scale
    
    # Obtain the max, min, and associated error bars for scaling
    max_mean, min_mean = obtain_min_max_df(df1_a,df2_a,df1_b,df2_b)
    max_s_mean, min_s_mean = obtain_min_max_df(s_df1_a,s_df2_a,s_df1_b,s_df2_b)

    if identifier == 'flux,hlr':
        max_mean_flux, min_mean_flux = obtain_min_max_df(df1_a,df1_b,df1_a,df1_b)
        max_s_mean_flux, min_s_mean_flux = obtain_min_max_df(s_df1_a,s_df1_b,s_df1_a,s_df1_b)
        max_mean_hlr, min_mean_hlr = obtain_min_max_df(df2_a,df2_b,df2_a,df2_b)
        max_s_mean_hlr, min_s_mean_hlr = obtain_min_max_df(s_df2_a,s_df2_b,s_df2_a,s_df2_b)

    
    # Initiate plot instance
    gs = gridspec.GridSpec(20,2)
    fig = plt.figure(figsize=(18,16))

    plt.suptitle(suptitle,fontsize=fs+6)

    # Input information on each subplot
    ax1 = fig.add_subplot(gs[0:8,0])
    title = title_arr[0]
    plt.title('Bias vs Separation For ' + title,fontsize=fs)
    plt.xlim([x_min,x_max])
    if not zoom:
        if identifier == 'flux,hlr':
            plt.ylim([(min_mean_flux - 2.2*min_s_mean_flux)*min_offset,(max_mean_flux + max_s_mean_flux)*max_offset])
        else:
            plt.ylim([(min_mean - 2.2*min_s_mean)*min_offset,(max_mean + max_s_mean)*max_offset])
    plt.xlabel('Separation (arcsec)',fontsize=fs)
    plt.ylabel('Residual (Fit - True)',fontsize=fs)    
    
    f_df1_a = format_df(df1_a,x_min,x_max,df1_a.index)
    f_s_df1_a = format_df(s_df1_a,x_min,x_max,s_df1_a.index)
    
    insert_subplot(ax1,f_df1_a,f_s_df1_a,min_sep)
        
    ax2 = fig.add_subplot(gs[11:19,0])
    title = title_arr[2]
    plt.title('Bias vs Separation For ' + title,fontsize=fs)
    plt.xlim([x_min,x_max])
    if not zoom:
        if identifier == 'flux,hlr':
            plt.ylim([(min_mean_flux - 2.2*min_s_mean_flux)*min_offset,(max_mean_flux + max_s_mean_flux)*max_offset])
        else:
            plt.ylim([(min_mean - 2.2*min_s_mean)*min_offset,(max_mean + max_s_mean)*max_offset])
    plt.xlabel('Separation (arcsec)',fontsize=fs)
    plt.ylabel('Residual (Fit - True)',fontsize=fs)
    f_df1_b = format_df(df1_b,x_min,x_max,df1_b.index)
    f_s_df1_b = format_df(s_df1_b,x_min,x_max,s_df1_b.index)

    insert_subplot(ax2,f_df1_b,f_s_df1_b,min_sep)
            
    ax3 = fig.add_subplot(gs[0:8,1])
    title = title_arr[1]
    plt.title('Bias vs Separation For ' + title,fontsize=fs)
    plt.xlim([x_min,x_max])
    if not zoom:
        if identifier == 'flux,hlr':
            plt.ylim([(min_mean_hlr - 2.2*min_s_mean_hlr)*min_offset,(max_mean_hlr + max_s_mean_hlr)*max_offset])
        else:
            plt.ylim([(min_mean - 2.2*min_s_mean)*min_offset,(max_mean + max_s_mean)*max_offset])
    plt.xlabel('Separation (arcsec)',fontsize=fs)
    plt.ylabel('Residual (Fit - True)',fontsize=fs)
    f_df2_a = format_df(df2_a,x_min,x_max,df2_a.index)
    f_s_df2_a = format_df(s_df2_a,x_min,x_max,s_df2_a.index)
    
    insert_subplot(ax3,f_df2_a,f_s_df2_a,min_sep)
    
    ax4 = fig.add_subplot(gs[11:19,1])
    title = title_arr[3]
    plt.title('Bias vs Separation For ' + title,fontsize=fs)
    plt.xlim([x_min,x_max])
    if not zoom:
        if identifier == 'flux,hlr':
            plt.ylim([(min_mean_hlr - 2.2*min_s_mean_hlr)*min_offset,(max_mean_hlr + max_s_mean_hlr)*max_offset])
        else:
            plt.ylim([(min_mean - 2.2*min_s_mean)*min_offset,(max_mean + max_s_mean)*max_offset])
    plt.xlabel('Separation (arcsec)',fontsize=fs)
    plt.ylabel('Residual (Fit - True)',fontsize=fs)
    f_df2_b = format_df(df2_b,x_min,x_max,df2_b.index)
    f_s_df2_b = format_df(s_df2_b,x_min,x_max,s_df2_b.index)
    
    insert_subplot(ax4,f_df2_b,f_s_df2_b,min_sep)
    
    # Output filename according to zoomed or not
    if not zoom:
        print '\nBias vs separation plot for ' + identifier +  ' finished.\n'
        if identifier == 'e1,e2':        
            plt.savefig(path + '/bias_vs_separation_ellipticity.png')
        elif identifier == 'flux,hlr':
            plt.savefig(path + '/bias_vs_separation_flux_hlr.png')
        elif identifier == 'x0,y0':
            plt.savefig(path + '/bias_vs_separation_x0_y0.png')
    else:
        print '\nZoomed Bias vs separation plot for ' + identifier +  ' finished.\n'
        if identifier == 'e1,e2':        
            plt.savefig(path + '/bias_vs_separation_ellipticity_zoomed.png')
        elif identifier == 'flux,hlr':
            plt.savefig(path + '/bias_vs_separation_flux_hlr_zoomed.png')
        elif identifier == 'x0,y0':
            plt.savefig(path + '/bias_vs_separation_x0_y0_zoomed.png')
    plt.clf()
    plt.close()
    
def save_image(path,results_deblend,dbl_im,image_params,truth,sep,
               psf_info):
                   
    """ Save a set of images from estimated parameters for visual checking. """ 
                   
    psf_flag , beta, fwhm_psf = psf_info
    
    image_params = pd.Series(image_params,index=['pixel_scale','x_len','y_len'])
 
    # Access a random set of data for the best fit and compare, access a random index and plot the deblended with fits
    index = int(len(dbl_im)*np.random.random(1))
    dbl_obj_im_a = dbl_im[index][0][0]
    dbl_obj_im_b = dbl_im[index][0][1]
    index_val = dbl_im[index][1]
    fit_to_dbl_obj = results_deblend.ix[index_val]
    
    # Create the true images of the unblended profiles
    true_im_a = create_galaxy(truth['flux_a'],truth['hlr_a'],truth['e1_a'],truth['e2_a'],truth['x0_a'],truth['y0_a'],
                              x_len=image_params['x_len'],y_len=image_params['y_len'],scale=image_params['pixel_scale'],
                              psf_flag=psf_flag,beta=beta,size_psf=fwhm_psf)
    true_im_b = create_galaxy(truth['flux_b'],truth['hlr_b'],truth['e1_b'],truth['e2_b'],truth['x0_b'],truth['y0_b'],
                              x_len=image_params['x_len'],y_len=image_params['y_len'],scale=image_params['pixel_scale'],
                              psf_flag=psf_flag,beta=beta,size_psf=fwhm_psf)
    # Create the images of the fits to the deblended objects.x_len=image_params['x_len'],y_len=image_params['y_len'],scale=image_params['pixel_scale'])
    fit_dbl_a = create_galaxy(fit_to_dbl_obj['flux_a'],fit_to_dbl_obj['hlr_a'],fit_to_dbl_obj['e1_a'],fit_to_dbl_obj['e2_a'],fit_to_dbl_obj['x0_a'],fit_to_dbl_obj['y0_a'],
                              x_len=image_params['x_len'],y_len=image_params['y_len'],scale=image_params['pixel_scale'],
                              psf_flag=psf_flag,beta=beta,size_psf=fwhm_psf)
    fit_dbl_b = create_galaxy(fit_to_dbl_obj['flux_b'],fit_to_dbl_obj['hlr_b'],fit_to_dbl_obj['e1_b'],fit_to_dbl_obj['e2_b'],fit_to_dbl_obj['x0_b'],fit_to_dbl_obj['y0_b'],
                              x_len=image_params['x_len'],y_len=image_params['y_len'],scale=image_params['pixel_scale'],
                              psf_flag=psf_flag,beta=beta,size_psf=fwhm_psf)

    # Plot the fits to the deblended profiles vs the deblended proflies vs the true profiles
    gs = gridspec.GridSpec(7,9)                                   
    fig = plt.figure(figsize=(15,25))
    sh = 0.8
    plt.suptitle('  True Objects, Deblended Objects & Fits to Deblended Objects\n For Separation: '+ str(sep) + '\"',fontsize=30)
    
    # Plot the true blend
    ax = fig.add_subplot(gs[0,1:7])
    z = ax.imshow(true_im_a.array + true_im_b.array,interpolation='none',origin='lower',cmap='jet'); plt.title('True Blend'); plt.colorbar(z,shrink=sh)
    
    # Plot the true objects
    ax1 = fig.add_subplot(gs[1,0:4])
    a = ax1.imshow(true_im_a.array,interpolation='none',origin='lower',cmap='jet'); plt.title('True Object A'); plt.colorbar(a,shrink=sh)
    ax2 = fig.add_subplot(gs[1,6:10])
    b = ax2.imshow(true_im_b.array,interpolation='none',origin='lower',cmap='jet'); plt.title('True Object B'); plt.colorbar(b,shrink=sh)
    
    # Plot the deblended objects
    ax3 = fig.add_subplot(gs[2,0:4])
    c = ax3.imshow(dbl_obj_im_a,interpolation='none',origin='lower',cmap='jet'); plt.title('Deblended Object A'); plt.colorbar(c,shrink=sh)
    ax4 = fig.add_subplot(gs[2,6:10])
    d = ax4.imshow(dbl_obj_im_b,interpolation='none',origin='lower',cmap='jet'); plt.title('Deblended Object B'); plt.colorbar(d,shrink=sh)
    
    # Plot the residual of the deblended and the true individual objects
    ax5 = fig.add_subplot(gs[3,0:4])
    e = ax5.imshow(dbl_obj_im_a-true_im_a.array,interpolation='none',origin='lower',cmap='jet'); plt.title('Residual of Deblended A and True A'); plt.colorbar(e,shrink=sh)
    ax6 = fig.add_subplot(gs[3,6:10])
    f = ax6.imshow(dbl_obj_im_b-true_im_b.array,interpolation='none',origin='lower',cmap='jet'); plt.title('Residual of Deblended B and True B'); plt.colorbar(f,shrink=sh)
    
    # Plot the fits to the deblended objects
    ax7 = fig.add_subplot(gs[4,0:4])
    g = ax7.imshow(fit_dbl_a.array,interpolation='none',origin='lower',cmap='jet'); plt.title('Fit To Deblended Object A'); plt.colorbar(g,shrink=sh)
    ax8 = fig.add_subplot(gs[4,6:10])
    h = ax8.imshow(fit_dbl_b.array,interpolation='none',origin='lower',cmap='jet'); plt.title('Fit To Deblended Object B'); plt.colorbar(h,shrink=sh)
    
    # Plot the residual of the fit to the deblended to the deblended objects
    ax9 = fig.add_subplot(gs[5,0:4])
    i = ax9.imshow(fit_dbl_a.array-dbl_obj_im_a,interpolation='none',origin='lower',cmap='jet'); plt.title('Residual Of Fit To Deblended Object A and Deblended A'); plt.colorbar(i,shrink=sh)
    ax10 = fig.add_subplot(gs[5,6:10])
    j = ax10.imshow(fit_dbl_b.array-dbl_obj_im_b,interpolation='none',origin='lower',cmap='jet'); plt.title('Residual Of Fit To Deblended Object B and Deblended B'); plt.colorbar(j,shrink=sh)
    
    # Plot the residual of the fit to the deblended to the true unblended object
    ax11 = fig.add_subplot(gs[6,0:4])
    k = ax11.imshow(fit_dbl_a.array-true_im_a.array,interpolation='none',origin='lower',cmap='jet'); plt.title('Residual Of Fit To Deblended Object A and True Object A'); plt.colorbar(k,shrink=sh)
    ax12 = fig.add_subplot(gs[6,6:10])
    l = ax12.imshow(fit_dbl_b.array-true_im_b.array,interpolation='none',origin='lower',cmap='jet'); plt.title('Residual Of Fit To Deblended Object B and True Object B'); plt.colorbar(l,shrink=sh)

    plt.savefig(path + '/image_of_one_trial.png')
    plt.clf()
    plt.close()
