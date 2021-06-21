import numpy as np
from taufit import fit_drw 
from taufit import hampel_filter
import statsmodels.api as sm

#hampel filtering
def hampel_sort(mjd, mag, err):
    hampel_parse = 0
    hampel_mjd = []
    hampel_mag = []
    hampel_err = []
    
    while hampel_parse < len(mjd):
        mjd_new, mag_new, mask_bool = hampel_filter(np.array(mjd[hampel_parse]), np.array(mag[hampel_parse]), 365)
        err_parse = np.array(err)[hampel_parse]
        err_new = np.array(err_parse)[~mask_bool]
        
        hampel_mjd.append(mjd_new)
        hampel_mag.append(mag_new)
        hampel_err.append(err_new)
        hampel_parse += 1

    return hampel_mjd, hampel_mag, hampel_err

#return fit_drw plots
def timing_analysis(mjd, mag, err):
    
    hampel_mjd, hampel_mag, hampel_err = hampel_sort(mjd, mag, err)
    parse = 0
    
    while parse < len(hampel_mjd):
        gp, samples, fig = fit_drw(np.array(hampel_mjd)[parse]*u.day, np.array(hampel_mag)[parse]*u.mag, 
                                   yerr = np.array(hampel_err)[parse]*u.mag, 
                                   plot = True, nburn = 50, nsamp = 200)
        
        sigma_drw = np.median(np.sqrt(np.exp(samples[:,0])/2))
        sigma_wn =  np.sqrt(np.mean(hampel_err[parse])**2 + np.median(np.exp(samples[:,2]))**2)
        snr_val = sigma_drw/sigma_wn
        
        lbval, p_val = sm.stats.acorr_ljungbox(hampel_mag[parse])
        sigma_lb = p2sigma(p_val[-1])
        
        ax_lc = fig.axes[-2]
        ax_lc.text(0.05, 0.95, 'SNR = %f, sigma_LB = %f' %(snr_val, sigma_lb), transform = ax_lc.transAxes, fontsize = 15)
        
        fig.savefig('page_%d_sigma.pdf' %parse, dpi = 300, bbox_inches = 'tight')
        
        print('fit_drw has completed %s out of %d iterations' %(parse+1, len(mjd)))
        parse += 1
