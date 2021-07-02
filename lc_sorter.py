def lc_sorter(fits_file, ipac_file):
    lcFULL = fits.open(fits_file) #fits_file should be string, i.e 'LIU_lc.fits'
    lcdata = lcFULL[1].data
    
    lc_coord = ascii.read(ipac_file) #ipac_file should be string, i.e 'matched_IPAC_LIU.txt'
    lc_ra = np.array(lc_coord['ra'])*u.degree
    lc_de = np.array(lc_coord['dec'])*u.degree
    lc_matchcoord = SkyCoord(lc_ra, lc_de)
    
    band_datara = lcdata['ra']*u.degree
    band_datade = lcdata['dec']*u.degree
    band_datacoord = SkyCoord(band_datara, band_datade)
    
    lc_idx, band_idx, d2d, d3d = search_around_sky(lc_matchcoord, band_datacoord, 0.5*u.arcsec)
    
    #find where the the index in lc_idx changes, this change should indicate different coordinates and thus differents LCs.
    k = 0
    temp_change = []
    while k < len(lc_idx):
        if lc_idx[k] == lc_idx[k-1]:
            k += 1
        else:
            temp_change.append(k)
            k += 1
    temp_change.append(len(lc_idx))

    
    diff_idx = []
    z = 0
    while z < len(lc_idx):
        if lc_idx[z] == lc_idx[z-1]:
            z += 1
        else:
            diff_idx.append(lc_idx[z])
            z += 1
    
    temp_idx = [[] for x in range(len(lc_matchcoord))] #good indexes
    j = 0
    y = 0
    while j+1 < len(temp_change):
        if y in diff_idx:
            temp_idx[y] = band_idx[temp_change[j] : temp_change[j+1]] 
            j += 1
            y += 1
        else:
            y += 1
            
    coord_mask = []
    j = 0
    while j < len(temp_idx):
        if len(temp_idx[j]) == 0:
            j += 1
        else:
            coord_mask.append(j)
            j += 1
    
    print('sorted!')
    
    i = 0
    mjd = []
    mag = []
    err = []
    
    while i < len(temp_idx):
        mjd.append(lcdata['mjd'][temp_idx[i]])
        mag.append(lcdata['mag'][temp_idx[i]])
        err.append(lcdata['magerr'][temp_idx[i]])
        i += 1
    
    return mjd, mag, err, coord_mask

def argsort(mjd, mag, err):
    mjdfin = []
    magfin = []
    errfin = []
    
    k = 0
    while k < len(mjd):
        maskfin = np.argsort(mjd[k])
        mjdfin.append(mjd[k][maskfin])
        magfin.append(mag[k][maskfin])
        errfin.append(err[k][maskfin])
        k += 1
    
    return mjdfin, magfin, errfin

#######

def p2sigma(p):
    import numpy as np
    import scipy.stats as st
    log_p = np.log(p)
    if (log_p > -36):
        sigma = st.norm.ppf(1 - p/2)
    else:
        sigma = np.sqrt(np.log(2/np.pi) - 2*np.log(8.2) - 2*log_p)
    return sigma

########

import statsmodels.api as sm

def timing_analysis(mjd, mag, err, names, oridx, nburn_val, nsamp_val, var_filename):
    
    hampel_parse = 0
    hampel_mjd = []
    hampel_mag = []
    hampel_err = []
    
    while hampel_parse < len(mjd):
        mjd_new, mag_new, mask_bool = hampel_filter(np.array(mjd)[hampel_parse], np.array(mag)[hampel_parse], 365)
        err_parse = np.array(err)[hampel_parse]
        err_new = np.array(err_parse)[~mask_bool]
        
        hampel_mjd.append(mjd_new)
        hampel_mag.append(mag_new)
        hampel_err.append(err_new)
        hampel_parse += 1
    
    parse = 0
    
    tau_drw = []
    tau_drw_lo = []
    tau_drw_hi = []
    
    sigma_drwlist = []
    sigma_drw_lo = []
    sigma_drw_hi = []
    
    sigma_nlist = []
    sigma_n_lo = []
    sigma_n_hi = []
    
    sigma_wnlist = []
    SNR = []
    pvalue_list = []
    sigma_lblist = []
    
    while parse < len(hampel_mjd):
        gp, samples, fig = fit_drw(np.array(hampel_mjd)[parse]*u.day, np.array(hampel_mag)[parse]*u.mag, 
                                   yerr = np.array(hampel_err)[parse]*u.mag, target_name = names[parse] +' '+ oridx[parse], 
                                   plot = True, nburn = nburn_val, nsamp = nsamp_val)
        
        tau_val = 1/np.exp(np.median(samples[:,1]))
        tau_lo = tau_val - np.percentile(1/np.exp(samples[:,1]), 16)
        tau_hi = np.percentile(1/np.exp(samples[:,1]), 84) - tau_val
        tau_drw.append(tau_val)
        tau_drw_lo.append(tau_lo)
        tau_drw_hi.append(tau_hi)
        
        sigma_drw = np.median(np.sqrt(np.exp(samples[:,0])/2))
        sigmadrw_lo = np.median(np.sqrt(np.exp(np.percentile(samples[:,0], 16))/2))
        sigmadrw_hi = np.median(np.sqrt(np.exp(np.percentile(samples[:,0], 84))/2))
        sigma_drwlist.append(sigma_drw)
        sigma_drw_lo.append(sigmadrw_lo)
        sigma_drw_hi.append(sigmadrw_hi)
        
        sigma_n = np.median(np.sqrt(np.exp(samples[:,2])/2))
        sigman_lo = np.median(np.sqrt(np.exp(np.percentile(samples[:,2], 16))/2))
        sigman_hi = np.median(np.sqrt(np.exp(np.percentile(samples[:,2], 84))/2))
        sigma_nlist.append(sigma_n)
        sigma_n_lo.append(sigman_lo)
        sigma_n_hi.append(sigman_hi)
        
        sigma_wn =  np.sqrt(np.mean(hampel_err[parse])**2 + np.median(np.exp(samples[:,2]))**2)
        sigma_wnlist.append(sigma_wn)
        
        snr_val = sigma_drw/sigma_wn
        SNR.append(snr_val)
        rounded_snr = "{:.1f}".format(snr_val)
        
        try:
            lbval, p_val = sm.stats.acorr_ljungbox(hampel_mag[parse])
            pvalue_list.append(p_val[-1])
        
            sigma_lb = p2sigma(p_val[-1])
            sigma_lblist.append(sigma_lb)
            rounded_sigma = "{:.1f}".format(sigma_lb)
        
        except ValueError:
            pvalue_list.append('ValueError')
            sigma_lb = 'ValueError'
            sigma_lblist.append(sigma_lb)
            rounded_sigma = 0.0
            pass
        
        ax_lc = fig.axes[-2]
        ax_lc.text(0.03, 0.93, r'SNR = %s, $\sigma_{\rm{LB}}$ = %s' %(rounded_snr, rounded_sigma), transform = ax_lc.transAxes, fontsize = 20)
        
        fig.savefig('%s page_%d_sigma.pdf' %(names[parse], parse), dpi = 300, bbox_inches = 'tight')
        
        print('fit_drw has completed %s out of %d iterations' %(parse+1, len(mjd)))
        parse += 1
    
    ascii.write([tau_drw, tau_drw_lo, tau_drw_hi, sigma_drwlist, sigma_drw_lo, sigma_drw_hi,
                 sigma_nlist, sigma_n_lo, sigma_n_hi, sigma_wnlist, SNR, pvalue_list, sigma_lblist], 
                '%s.txt' %var_filename, 
                names=['tau', 'tau_lo', 'tau_hi', 'sigma_drw', 'sigma_drw_lo', 'sigma_drw_hi',
                       'sigma_n', 'sigma_n_lo', 'sigma_n_hi', 'sigma_wn', 'SNR', 'pvalue', 'sigma_lb'], format = 'ipac')
    
    return tau_drw
