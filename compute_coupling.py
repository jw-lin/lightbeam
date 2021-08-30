import numpy as np
import LPmodes
import matplotlib.pyplot as plt
from misc import resize,normalize,printProgressBar
from scipy.optimize import minimize,minimize_scalar
import dask
import dask.array as da
from dask.distributed import Client, progress
import h5py
import hcipy as hc
import AOtele
import numexpr as ne
from scipy.interpolate import UnivariateSpline
from numba import njit



def overlap(u0,u1,c = False):
    if not c:
        return np.abs(np.sum( u0*np.conj(u1) ))
    else:
        return np.sum(u0*np.conj(u1))

def plot_modes(ncore,nclad,wl0):
    core_rs = [2.2]
    xa,ya = np.linspace(-50,50,400),np.linspace(-50,50,400)
    xg,yg = np.meshgrid(xa,ya)

    for r in core_rs:
        alpha = r/10
        field = LPmodes.lpfield(xg,yg,0,1,r,wl0,ncore,nclad)
        field = normalize(field)
        plt.plot(xa,np.abs(field[int(len(field)/2)]),alpha=alpha,color='k')
    
    plt.show()

def center(u):
    c0 = (int(u.shape[0]/2)+1,int(u.shape[1]/2)+1)
    c = np.unravel_index(np.abs(u).argmax(),u.shape)
    u = np.roll(u,c0[0]-c[0],0)
    u = np.roll(u,c0[1]-c[1],1)
    return u

def coupling_vs_r(psf,ncore,nclad,wl0):
    core_rs = np.linspace(1,10,30)
    couples = []
    ws = []
    for r in core_rs:
        _r,c,w = scale_u0_lp01(psf,r,ncore,nclad,wl0)
        couples.append(c)
        ws.append(w)
    plt.plot(core_rs,couples)
    plt.xlabel("core radius (um)")
    plt.ylabel("maximum coupling into LP01")
    plt.show()
    plt.plot(core_rs,ws)
    plt.xlabel("core radius (um)")
    plt.ylabel("optimal field extent (um)")
    plt.show()

def get_power_in_mode(u0,rcore,ncore,nclad,mode,width,wl0):
    xa = np.linspace(-width,width,u0.shape[0],endpoint=False)
    ya = np.linspace(-width,width,u0.shape[1],endpoint=False)
    xg,yg = np.meshgrid(xa,ya)
    if mode[0] == 0:
        lf = LPmodes.lpfield(xg,yg,mode[0],mode[1],rcore,wl0,ncore,nclad)
        field = normalize(lf)
        _power = np.power(overlap(u0,field),2)
        return _power
    else:
        lf0 = LPmodes.lpfield(xg,yg,mode[0],mode[1],rcore,wl0,ncore,nclad,"cos")
        lf1 = LPmodes.lpfield(xg,yg,mode[0],mode[1],rcore,wl0,ncore,nclad,"sin")
        
        field0 = normalize(lf0)
        field1 = normalize(lf1)
        
        power0 = np.power(overlap(u0,field0),2)
        power1 = np.power(overlap(u0,field1),2)
        return power0+power1

def get_power_in_modes(width,rcore,ncore,nclad,u0,modes,wl0):
    _power = 0
    for mode in modes:
        _power += get_power_in_mode(u0,rcore,ncore,nclad,mode,width,wl0)
    return _power

def get_power_in_modes_neg(width,rcore,ncore,nclad,u0,modes,wl0):
    _power = 0
    for mode in modes:
        _power += get_power_in_mode(u0,rcore,ncore,nclad,mode,width,wl0)
    return -1*_power

def scale_u0_lp01(psfs,rcore,ncore,nclad,wl0):
    return scale_u0_fast(psfs,rcore,ncore,nclad,wl0,[(0,1)])

def LP01gauss(xg,yg,rcore,wl0,ncore,nclad):
    V = LPmodes.get_V(2*np.pi/wl0,rcore,ncore,nclad)
    w = rcore * (0.65 + 1.619/V**1.5 + 2.879/V**6 - (0.016 + 1.561 * V**(-7)))
    rsq = xg*xg + yg*yg
    return normalize( np.exp(-rsq/w**2) )
 
def scale_u0_fast(psfs,rcore,ncore,nclad,wl0,modes=None):
    '''assuming psfs is a smaller subset of all psfs, do a vectorized coupling calc and minimize'''
    if rcore==0:
        return 0
    k0 = 2*np.pi/wl0
    V = LPmodes.get_V(k0,rcore,ncore,nclad)
    if modes is None:
        modes = LPmodes.get_modes(V)
        #print(modes)
    res = minimize_scalar(compute_coupling,args=(psfs,modes,rcore,ncore,nclad,wl0,True),method='brent',bounds=(20,200),bracket=(20,90,200))
    #res = minimize(compute_avg_coupling_at_r,80,args=(psfs,modes,rcore,ncore,nclad))
    return np.abs(res.x)

def compute_coupling(width,psfs,modes,rcore,ncore,nclad,wl,opt_mode=False,full_output=False,_complex=False):
    '''return coupling given some parameters. opt_mode flag is for use w/ minimization funcs'''
    xa = np.linspace(-width,width,psfs[0].shape[0],endpoint=False)
    ya = np.linspace(-width,width,psfs[0].shape[1],endpoint=False)
    xg,yg = np.meshgrid(xa,ya)
    fields = []

    for mode in modes:
        if mode[0] == 0:
            lf = LPmodes.lpfield(xg,yg,mode[0],mode[1],rcore,wl,ncore,nclad)
            fields.append(normalize(lf))
        else:
            lf0 = LPmodes.lpfield(xg,yg,mode[0],mode[1],rcore,wl,ncore,nclad,"cos")
            lf1 = LPmodes.lpfield(xg,yg,mode[0],mode[1],rcore,wl,ncore,nclad,"sin")
            fields.append(normalize(lf0))
            fields.append(normalize(lf1))

    fields = np.array(fields)    
    overlaps = psfs[:,None] * fields[None,:] 


    if _complex:
        return np.sum(overlaps,axis=(2,3))

    powers = np.power(np.abs(np.sum(overlaps,axis=(2,3))),2)

    if full_output:
        # see how power is distributed among modes
        return powers

    total_powers = np.sum(powers,axis=1)
    if opt_mode:
        # this flag is for use w/ the focal plane width optimization
        return -np.mean(total_powers)
    return total_powers

def compute_coupling_vs_r(psfs,rcores,ncore,nclad,wl0,fname,width,rcore0,include_SMF=True,TT=0, opt_after=False, widths=None):
    """compute avg coupling into MM fibers of various sizes given an input list of psfs. save to hdf5"""
    np.random.seed(123456789)
    
    results = []
    ## compute avg coupling into selected fiber dimension (rcore0) and adjust focal plane scale to maximize coupling
 
    j = 0
    total = len(rcores)
    k0 = 2*np.pi/wl0

    tilted_psfs = np.zeros_like(psfs)

    # apply tt to all psf realizations
    focal_plane_widths = []    

    for i in range(len(rcores)):
        rcore = rcores[i]
        if opt_after:
            width = widths[i]
        
        np.random.seed(123456789)

        V = LPmodes.get_V(k0,rcore,ncore,nclad)
        modes = LPmodes.get_modes(V)

        if not opt_after:
            if rcore0 == 0 :
                width = scale_u0_fast(psfs,rcore,ncore,nclad,wl0)

        for i in range(len(psfs)):
            tilted_psfs[i] = shift_u(psfs[i],TT,width)

        if opt_after:
            if rcore0 == 0 :
                width = scale_u0_fast(tilted_psfs,rcore,ncore,nclad,wl0)

        focal_plane_widths.append(width)

        results.append(compute_coupling(width,tilted_psfs,modes,rcore,ncore,nclad,wl0))
        printProgressBar(j,total)
        j+=1

    smfc = 0
    #compute smf coupling
    if include_SMF:
        smfc = compute_SMF_coupling(psfs)

    # save to file

    with h5py.File(fname+".hdf5", "w") as f:
        f.create_dataset("coupling", data = results)
        f.create_dataset("rcores", data = rcores)
        f.create_dataset("rcore0", data = rcore0)
        f.create_dataset("ncore", data = ncore)
        f.create_dataset("nclad", data = nclad)
        f.create_dataset("wl0", data = wl0)
        f.create_dataset("w", data = width)
        f.create_dataset("focalplanewidths", data = focal_plane_widths)
        f.create_dataset("SMFcoupling", data = smfc)

    # show prelim results

    #results_avg = np.mean(results,axis=1)
    #results_std = np.std(results,axis=1)

    #plt.plot(rcores,results_avg,color='k',ls='None',marker='.')
    #plt.fill_between(rcores,results_avg-results_std,results_avg+results_std,color='0.75',alpha=0.5)
    #plt.xlim(2,12)
    #plt.ylim(0,1)

    #plt.show()

def get_IOR(wl):
    """ for fused silica """
    wl2 = wl*wl
    return np.sqrt(0.6961663 * wl2 / (wl2 - 0.0684043**2) + 0.4079426 * wl2 / (wl2 - 0.1162414**2) + 0.8974794 * wl2 / (wl2 - 9.896161**2) + 1)

def compute_coupling_vs_wl(f,wls,rcore0,fname,focal_plane_width,contrast = 5.5e-3):
    """ compute avg coupling vs wavelength. Since psfs change with wavelength, 
        the psfs are generated on the fly from saved DM positions and atmospheric 
        turbulence states """

    #to ensure accurate psf generation, an AOtele object identical to the one used to generate DMshapes must be passed in.
    
    DMshapes = f['DMs'][:]
    ts = f['ts'][:]
    _args = f['tele_args']
    wl0 = _args['wl0'][()]

    # construct the telescope saved in the file
    tele = AOtele.setup_tele_hdf5(_args=f['tele_args'])
    tele.get_perfect_wfs()

    wf_pupil = tele.wf_pupil_hires
    u_pupil = AOtele.get_u(wf_pupil)
    print("computing screens @ reference wavelength")
    
    #precompute the phase screens (I estimate 200Mb memory for 100 screens, 512x512)
    #i should really just save these in the code
    #current_time = 0
    """
    screens = []
    for i in range(len(t_arr)):
        tele.get_screen()
        if (i!= 0 and i%100==0):
            screens.append(tele.current_phase_screen)
    """
    screens = f['screens'][:]

    xa = ya = np.linspace(-focal_plane_width,focal_plane_width,u_pupil.shape[0])
    xg , yg = np.meshgrid(xa,ya)

    results = []

    print("computing couplings vs wavelength")
    #now do coupling calculations
    for i in range(len(wls)):
        printProgressBar(i,len(wls))
        wl = wls[i]
        wf_pupil.wavelength = wl*1.0e-6

        ncore = get_IOR(wl)
        nclad = ncore - contrast
        # get the available LP modes
        k = 2*np.pi/wl
        V = LPmodes.get_V(k,rcore0,ncore,nclad)
        modes = LPmodes.get_modes(V)

        mode_fields = []
        for mode in modes:
            if mode[0] == 0:
                lf = LPmodes.lpfield(xg,yg,mode[0],mode[1],rcore0,wl,ncore,nclad)
                mode_fields.append( normalize(lf) )
            else:
                lf0 = LPmodes.lpfield(xg,yg,mode[0],mode[1],rcore0,wl,ncore,nclad,"cos")
                lf1 = LPmodes.lpfield(xg,yg,mode[0],mode[1],rcore0,wl,ncore,nclad,"sin")
                
                mode_fields.append(normalize(lf0))
                mode_fields.append(normalize(lf1))
        
        #get focal plane fields
        us = []
        for i in range(len(ts)):
            t = ts[i]
            tele.DM.actuators = DMshapes[i]
            wf_focal = tele.propagate(wf_pupil,screen=screens[i])
            u = AOtele.get_u(wf_focal)
            us.append(u)

        #compute coupling
        mode_fields = np.array(mode_fields)
        us = np.array(us)
        overlaps = np.sum(us[None,:,:,:]*mode_fields[:,None,:,:],axis=(2,3))
        powers = np.power(np.abs(overlaps),2)
        couplings = np.sum(powers,axis=0)
        results.append(couplings)

    results = np.array(results)

    # save to file
    with h5py.File(fname+".hdf5", "w") as f:
        f.create_dataset("coupling", data = results)
        f.create_dataset("rcore0", data = rcore0)
        f.create_dataset("wls", data = wls)
        f.create_dataset("wl0", data = wl0)
        f.create_dataset("focal plane width", data = focal_plane_width)

    # show prelim results

    avgs = np.mean(results,axis=1)
    stds = np.std(results,axis=1)

    plt.plot(wls,avgs,color='k',ls='None',marker='.')

    plt.fill_between(wls,avgs-stds,avgs+stds,color='0.75',alpha=0.5)

    plt.axvspan(1.02-0.06,1.02+0.06,alpha=0.35,color="burlywood",zorder=-100)
    plt.axvspan(1.220-0.213/2,1.220+0.213/2,alpha=0.25,color="indianred",zorder=-100)

    plt.xlim(0.9,1.4)
    plt.ylim(0.,1.0)
    plt.xlabel("wavelength (um)")
    plt.ylabel("coupling efficiency")
    plt.show()

def compute_SMF_coupling(psfs):
    rcore = 2.2
    ncore = 1.4504
    nclad = 1.4415478347942532
    wl0 = 1.

    u0 = psfs[0]

    width = scale_u0_fast(psfs,rcore,ncore,nclad,wl0)
    print("optimal focal plane width for SMF: ",width)
    xa = np.linspace(-width,width,u0.shape[0],endpoint=False)
    ya = np.linspace(-width,width,u0.shape[1],endpoint=False)
    xg,yg = np.meshgrid(xa,ya)

    k0 = 2*np.pi/wl0

    V = LPmodes.get_V(k0,rcore,ncore,nclad)
    modes = LPmodes.get_modes(V)

    assert len(modes)==1, "there should only be one mode in an SMF!"

    lp01 = normalize(LPmodes.lpfield(xg,yg,0,1,rcore,wl0,ncore,nclad))

    avg_coupling = np.mean(np.power(np.abs(np.sum(lp01[None,:,:]*psfs,axis=(1,2))),2))

    return avg_coupling

def TT_test(psfs,rcore,ncore=None,nclad=None):
    shifts = np.linspace(0,0.1,50) #in units of half focal plane width

    ncore = 1.4504 if ncore is None else ncore
    nclad = 1.4504 - 5.5e-3 if nclad is None else nclad
    wl0 = 1.

    k0 = 2*np.pi/wl0
    V = LPmodes.get_V(k0,rcore,ncore,nclad)
    modes = LPmodes.get_modes(V)

    pixel_shifts = (shifts * psfs[0].shape[0]).astype(int)

    for shift in pixel_shifts:

        shifted_psfs = []

        for psf in psfs:
            theta = np.random.uniform(0,2*np.pi)
            shiftx = int(np.cos(theta)*shift)
            shifty = int(np.sin(theta)*shift)
            shifted_psfs.append(np.roll(np.roll(psf,shiftx,axis=0),shifty,axis=1))

        shifted_psfs = np.array(shifted_psfs)
        width = scale_u0_fast(shifted_psfs,4,ncore,nclad,wl0)
        
        pick =  shifted_psfs[19]

        coupling = compute_coupling(width,psfs,modes,rcore,ncore,nclad,wl0)#get_power_in_modes(width,rcore,ncore,nclad,pick,modes,wl0)
        print(np.mean(coupling))

def shift_u(u,rms_disp,width):

    if rms_disp == 0:
        return u

    dispx = np.random.normal(scale=rms_disp)
    dispy = np.random.normal(scale=rms_disp)

    resx = u.shape[0]
    resy = u.shape[1]

    shiftx = int(dispx * resx/(2*width))
    shifty = int(dispy * resy/(2*width))

    u = np.roll(u,shiftx, axis=0)
    u = np.roll(u,shifty, axis=1)

    if shiftx > 0:
        u[:,:shiftx] = 0
    elif shiftx < 0:
        u[:,shiftx:] = 0

    if shifty > 0:
        u[:,:shifty] = 0
    elif shifty < 0:
        u[:,shifty:] = 0

    return u

def compute_strehl(psfs,reference_psf):
    avg = np.zeros_like(psfs[0])
    for psf in psfs:
        avg += psf*np.conj(psf)
    
    avg /= len(psfs)
    avg = np.sqrt(avg)
    return np.max(avg*np.conj(avg)) / np.max(reference_psf*np.conj(reference_psf))

def get_strehls_for_TT(TTs,piaa):

    strehls = []

    if piaa: 
        fname = "SR70_GPIAA_paper.hdf5"
        reference_psf = np.load("reference_psf_piaa.npy")
        widths = h5py.File("./TTdata2/TT70_GPIAA_0.hdf5","r")['focalplanewidths'][:]
    else:
        fname = "SR70_NOPIAA_paper.hdf5"
        reference_psf = np.load("reference_psf_nopiaa.npy")
        widths = h5py.File("./TTdata2/TT70_NOPIAA_0.hdf5","r")['focalplanewidths'][:]
    
    f = h5py.File(fname ,"r")
    psfs = f['psfs'][:]

    for TT in TTs:
        np.random.seed(123456789)
        tilted_psfs = np.zeros_like(psfs)
        for i in range(len(psfs)):
            tilted_psfs[i] = shift_u(psfs[i],TT,widths[i])
        strehls.append(compute_strehl(tilted_psfs,reference_psf))
    strehls = np.array(strehls)
    return np.real(strehls)

def compute_couplings_vortex(psfs,psf0,rcore,ncore,nclad,wl):
    ''' compute coupling efficiency for each mode, assuming vortex data '''

    modes = LPmodes.get_modes(LPmodes.get_V(2*np.pi/wl,rcore,ncore,nclad))
    width = 109 #scale_u0_fast(np.array([psf0]),rcore,ncore,nclad,wl,modes=[(0,1)])
    print(width)

    couplings = compute_coupling(width,psfs,modes,rcore,ncore,nclad,wl,full_output=True,_complex=True)

    couplings = couplings.T

    couplings_re = np.real(couplings)
    couplings_im = np.imag(couplings)

    return couplings_re,couplings_im


if __name__ == "__main__":

    f = h5py.File("j3_vortex1_sym.hdf5" ,"r")
    psfs = f['psfs'][:]
    psf0 = f['psf_no_vortex'][:]
    amps = f['amps'][:]

    ncore = 1.4504 #
    nclad = 1.4504 - 5.5e-3
    rcore = 6.21

    wl = 1.0
    cs_re,cs_im = compute_couplings_vortex(psfs,psf0,rcore,ncore,nclad,wl)

    modes = LPmodes.get_modes(LPmodes.get_V(2*np.pi/wl,rcore,ncore,nclad))

    mode_names = []

    for mode in modes:
        if mode[0] == 0:
            mode_names.append("LP0"+str(mode[1]))
        else:
            mode_names.append("LP"+str(mode[0])+str(mode[1])+" cos")
            mode_names.append("LP"+str(mode[0])+str(mode[1])+" sin")
    
    cc = ['#377eb8', '#ff7f00', '#4daf4a',
                '#f781bf', '#a65628', '#984ea3',
                '#999999', '#e41a1c', '#dede00']
    
    #REAL

    idx = 0
    for c,n in zip(cs_re,mode_names):
        if n[-3:] == "sin":
            plt.plot(amps,c,label=n,color=cc[idx],ls='dashed')
        else:
            plt.plot(amps,c,label=n,color=cc[idx])
        
        if n[-3:] != "cos":
            idx+=1

    plt.xlabel("Zernike amplitude")
    plt.ylabel("Re(LP mode amplitude)")
    plt.title("charge 1, tilt y")
    plt.legend()
    plt.show()

    #IMAGINARY

    idx = 0
    for c,n in zip(cs_im,mode_names):
        if n[-3:] == "sin":
            plt.plot(amps,c,label=n,color=cc[idx],ls='dashed')
        else:
            plt.plot(amps,c,label=n,color=cc[idx])
        
        if n[-3:] != "cos":
            idx+=1

    plt.xlabel("Zernike amplitude")
    plt.ylabel("Im(LP mode amplitude)")
    plt.title("charge 1, tilt y")
    plt.legend()
    plt.show()

    """
    from bisect import bisect_left
    ## in wavelength space:
    # blue edge of Y located at 6 - 8 mode number transition when rcore = 6.21 um
    # red edge of J located at 3 - 6 mode # transition when rcore = 6.41 um 
    
    #psf = np.load("perfect_psf.npy")
    ncore = 1.4504 #
    nclad = 1.4504 - 5.5e-3
    wl0 = 1.0

    f = h5py.File("SR70_GPIAA_paper.hdf5" ,"r")
    psfs = f['psfs'][:]
    rcores = np.linspace(2,12,100)

    compute_coupling_vs_r(psfs,rcores,ncore,nclad,wl0,"SR70_GPIAA_paper_coupling_allopt",0,0)
    
    
    #psfs = f['psfs'][:]
    #psf = psfs[15]
    #TT_test(psfs,7.5)

    #print(get_strehls_for_TT([0,1,2,3,4,5],True))

    ### figuring out the reason for the jump in the 21-port lantern

    ncore = 1.4504 #
    nclad = 1.4504 - 5.5e-3

    wl0 = 1.0

    rcore = 5

    modes = LPmodes.get_modes(LPmodes.get_V(2*np.pi/wl0,rcore,ncore,nclad))
    print(modes)

    fname = "full_PIAA_test.hdf5"
    psfs = h5py.File(fname,"r")['psfs'][:]

    fname2 = "full_noPIAA_test.hdf5"
    psfs2 = h5py.File(fname2,"r")['psfs'][:]

    w =  59.100945850109376  #scale_u0_fast(psfs,rcore,ncore,nclad,wl0) #59.100945850109376 
    print('opt width: ', w)

    xa = np.linspace(-w,w,psfs[0].shape[0],endpoint=False)
    ya = np.linspace(-w,w,psfs[0].shape[1],endpoint=False)
    xg,yg = np.meshgrid(xa,ya)
    lf = normalize(LPmodes.lpfield(xg,yg,0,2,rcore,wl0,ncore,nclad))
    plt.imshow(np.abs(lf))
    plt.show()
    plt.imshow(np.arctan2(np.imag(lf),np.real(lf)),cmap="RdBu")
    plt.show()


    for i in range(100):

        print("overlaps w/ lp13: ", np.power(np.abs(overlap(psfs[i],lf)),2) , np.power(np.abs(overlap(psfs2[i],lf)),2))

        fig,axs = plt.subplots(1,3,sharey=True,frameon=False)
        axs[0].imshow(np.abs(psfs[i]))
        axs[0].set_title("PSF w/ PIAA")
        axs[1].imshow(np.abs(psfs2[i]))
        axs[1].set_title("PSF w/o PIAA")
        axs[2].imshow(np.abs(lf))
        axs[2].set_title("LP13")

        for ax in axs:
            ax.set_axis_off()
        plt.subplots_adjust(hspace=0,wspace=0)
        plt.show()

    #w = scale_u0_fast(psfs,rcore,ncore,nclad,wl0)
    #c = compute_coupling(w,psfs,modes,rcore,ncore,nclad,wl0,full_output=True)
    #print(modes)
    #print(np.mean(c))
    """

    """
    ncore = 1.4504 #
    nclad = 1.4504 - 5.5e-3

    wl0 = 1.0

    #19 mode data

    rcore = 21.8/2

    srs = [10,20,30,40,50,60,70]

    out_nopiaa = []
    out_piaa = []

    modes = LPmodes.get_modes(LPmodes.get_V(2*np.pi/wl0,rcore,ncore,nclad))

    for sr in srs:
        if sr == 10:
            fname = "full_PIAA_test.hdf5"
        else:
            fname = "full_PIAA_test_"+str(sr)+".hdf5"
        
        psfs = h5py.File(fname,"r")['psfs'][:]
        w = scale_u0_fast(psfs,rcore,ncore,nclad,wl0)
        c = compute_coupling(w,psfs,modes,rcore,ncore,nclad,wl0)
        out_piaa.append(np.mean(c))
        print(sr,np.mean(c))

    for sr in srs:
        if sr == 10:
            fname = "full_noPIAA_test.hdf5"
        else:
            fname = "full_noPIAA_test_"+str(sr)+".hdf5"
        
        psfs = h5py.File(fname,"r")['psfs'][:]
        w = scale_u0_fast(psfs,rcore,ncore,nclad,wl0)
        c = compute_coupling(w,psfs,modes,rcore,ncore,nclad,wl0)
        out_nopiaa.append(np.mean(c))
        print(sr,np.mean(c))

    """

    """
    V_cutoffs = np.array([2.4048,3.8317,5.1356,5.5201,6.38,7.0156,7.588,8.4172,8.6537,8.771])

    wl0 = 1.0 #um

    ## fiber params
    rcore = 4 #um
    ncore = 1.4504
    nclad = 1.4504 - 5.5e-3
    wl0 = 1.0 #um 

    k0 = 2*np.pi/wl0

    r_cutoffs = V_cutoffs / k0 / LPmodes.get_NA(ncore,nclad)
    print(r_cutoffs)

    f = h5py.File("SR10_GPIAA_paper.hdf5" ,"r")

    psfs = f['psfs'][:]

    #opt_width = scale_u0_all_2(psfs,rcore,ncore,nclad,wl0) #61.63445457276633#

    opt_width = scale_u0_fast(psfs,rcore,ncore,nclad,wl0)

    print("optimal focal plane width for MMF: ",opt_width)

    #wls = np.linspace(0.9,1.4,100)
    #compute_coupling_vs_wl(f,wls,rcore,"SR10_NOPIAA_1port_coupling_chromatic_paper",opt_width)
    """

    """
    V_cutoffs = np.array([2.4048,3.8317,5.1356,5.5201,6.38,7.0156,7.588,8.4172,8.6537,8.771])
    
    rms_s = [1,2,3,4,5] 

    f2 = h5py.File("./TTdata2/TT70_GPIAA_0.hdf5")
    widths = f2['focalplanewidths'][:]
    print(widths)

    reference_psf = np.load("reference_psf_piaa.npy")


    fname = "SR70_GPIAA_paper.hdf5"

    f = h5py.File(fname ,"r")
    psfs = f['psfs'][:]

    ## fiber params
    rcore = 0.0 #um
    ncore = 1.4504
    nclad = 1.4504 - 5.5e-3
    wl0 = 1.0 #um

    k0 = 2*np.pi/wl0

    r_cutoffs = V_cutoffs / k0 / LPmodes.get_NA(ncore,nclad)

    r = 2
    dr = 0.1 
    eps = 0.0025
    rcores = []
    last_idx = 1
  
    while r <= 12:
        current_idx = bisect_left(r_cutoffs,r)
        if current_idx != last_idx and current_idx < 10:
            rcores.append(r_cutoffs[current_idx] - eps)
            rcores.append(r_cutoffs[current_idx] + eps)
            last_idx = current_idx
        rcores.append(r) 
        r+=dr

    rcores = np.sort(rcores)

    opt_width = scale_u0_fast(psfs,rcore,ncore,nclad,wl0)
    print("optimal focal plane width for MMF: ",opt_width)

    #rcores = np.linspace(2,12,100)
    #compute_coupling_vs_r(psfs,rcores,ncore,nclad,wl0,"TT_noPIAA_"+str(rms)+"_coupling",opt_width,rcore,False)
 
    for rms in rms_s:
        compute_coupling_vs_r(psfs,rcores,ncore,nclad,wl0,"./TTdata2/TT70_GPIAA_optafter_"+str(rms),opt_width,rcore,False,rms,opt_after=True,widths=widths)
    """

    """
    V_cutoffs = np.array([2.4048,3.8317,5.1356,5.5201,6.38,7.0156,7.588,8.4172,8.6537,8.771])
    
    #rms_s = [0,0.2,0.4,0.6,0.8,1]
    rms_s = [1.2,1.4,1.6,1.8,2]
  
    fname = "SR70_noPIAA_paper"


    def compute_for_rms(rms):
        filename = "TT_noPIAA_"+str(rms)
        f = h5py.File(filename+".hdf5" ,"r")
        psfs = f['psfs'][:] 

        ## fiber params
        rcore = 0.0 #glonal opt.
        ncore = 1.4504
        nclad = 1.4504 - 5.5e-3
        wl0 = 1.0 #um

        k0 = 2*np.pi/wl0

        r_cutoffs = V_cutoffs / k0 / LPmodes.get_NA(ncore,nclad)

        r = 2
        dr = 0.1
        eps = 0.0025
        rcores = []
        last_idx = 1

        while r <= 12:
            current_idx = bisect_left(r_cutoffs,r)
            if current_idx != last_idx and current_idx < 10:
                rcores.append(r_cutoffs[current_idx] - eps)
                rcores.append(r_cutoffs[current_idx] + eps)
                last_idx = current_idx
            rcores.append(r)
            r+=dr

        rcores = np.sort(rcores)

        opt_width = scale_u0_fast(psfs,rcore,ncore,nclad,wl0)
        print("optimal focal plane width for MMF: ",opt_width)

        #rcores = np.linspace(2,12,100)
        #compute_coupling_vs_r(psfs,rcores,ncore,nclad,wl0,"TT_noPIAA_"+str(rms)+"_coupling",opt_width,rcore,False)
        compute_coupling_vs_r(psfs,rcores,ncore,nclad,wl0,filename+"_coupling_allopt.hdf5",opt_width,rcore,False)

    #from dask.distributed import Client, progress
    #client = Client(threads_per_worker=1, n_workers=2)

    #lazy_results = []
    for rms in rms_s:
        compute_for_rms(rms)
        #lazy_result = dask.delayed(compute_for_rms)(rms)
        #lazy_results.append(lazy_result)

    #dask.compute(*lazy_results)
    """
    '''
    # SMF coupling
    rcore = 2.2
    ncore = 1.4504
    nclad = 1.4415478347942532

    psfs = np.load("psfs_sr0pt2.npy")

    plt.imshow(np.abs(psfs[42]))
    plt.axis('off')
    plt.show()

    #print(compute_SMF_coupling(psfs))

    '''

    """
    #plotting PIAA boost vs strehl:


    mode = "_3mode"
    SRs = [10,20,30,40,50,60,70]
    dif1s = []
    dif2s = []
    smf1s = []
    smf2s = []
    smf3s = []
    smf4s = []

    for sr in SRs:

        if sr == 10:

            f1 = h5py.File("test_coupling_results.hdf5","r")
            f2 = h5py.File("test_coupling_results_PIAA.hdf5","r")

            f3 = h5py.File("test_coupling_results_3mode.hdf5","r")
            f4 = h5py.File("test_coupling_results_PIAA_3mode.hdf5","r")

        else:
            f1 = h5py.File("test_coupling_results_"+str(sr)+".hdf5","r")
            f2 = h5py.File("test_coupling_results_PIAA_"+str(sr)+".hdf5","r")

            f3 = h5py.File("test_coupling_results_"+str(sr)+mode+".hdf5","r")
            f4 = h5py.File("test_coupling_results_PIAA_"+str(sr)+mode+".hdf5","r")

        c1s = np.mean(f1["coupling"][:].T,axis=0)
        c2s = np.mean(f2["coupling"][:].T,axis=0)

        c3s = np.mean(f3["coupling"][:].T,axis=0)
        c4s = np.mean(f4["coupling"][:].T,axis=0)

        rs = f1["rcores"][:]

        cr1 = UnivariateSpline(rs,c1s,s=0)
        cr2 = UnivariateSpline(rs,c2s,s=0)
        cr3 = UnivariateSpline(rs,c3s,s=0)
        cr4 = UnivariateSpline(rs,c4s,s=0)
        
        smf1s.append( f1["SMFcoupling"][()])
        smf2s.append( f2["SMFcoupling"][()])
        smf3s.append( f1["SMFcoupling"][()])
        smf4s.append( f2["SMFcoupling"][()])

        dif1 = (cr2(6.21) - cr1(6.21))/cr1(4)
        dif2 = (cr4(4) - cr3(4))/cr3(4)

        dif1s.append(dif1)
        dif2s.append(dif2)
        print(sr,cr1(6.21),cr2(6.21))

    from scipy.stats import linregress
    fit1 = linregress(SRs,smf1s)
    fit2 = linregress(SRs,smf2s)

    l1 = lambda x: fit1[0]*x + fit1[1]
    l2 = lambda x: fit2[0]*x + fit2[1]

    print(l1(100),l2(100))

    xs = np.linspace(0,100,100)
    plt.plot(xs,l1(xs),ls='dashed',color='k')
    plt.plot(xs,l2(xs),ls='dashed',color='k')

    plt.plot(SRs,smf1s,marker='o',ls="None",color='steelblue',label="no piaa, opt. to 6 mode")
    plt.xlabel("Strehl ratio (%)")
    plt.ylabel("SMF coupling efficiency")

    plt.plot(SRs,smf2s,marker='o',ls="None",color='indianred',label = "piaa, opt. to 6 mode")

    

    #plt.plot(SRs,smf3s,marker='^',ls="None",color='steelblue',label = "no piaa, opt. to 3 mode")
    #plt.plot(SRs,smf4s,marker='^',ls="None",color='indianred',label = "no piaa, opt. to 3 mode")

    plt.legend(loc='best',frameon=False)
    plt.ylim(0,1)
    plt.show()
    
    plt.plot(SRs,dif1s,marker='o',ls="None",color='k',label = "opt. to 6 mode")
    plt.plot(SRs,dif2s,marker='^',ls="None",color='k',label = "opt. to 3 mode")

    plt.legend(loc='best',frameon=False)
    plt.xlabel("Strehl ratio (%)")
    plt.ylabel("PIAA coupling boost")
    #plt.ylim(0,0.15)
    plt.show()
    """

    ## PIAA methods comp
    """
    f1 = h5py.File("SR70_PIAALP_RT_0_coupling.hdf5","r")
    f2 = h5py.File("SR70_PIAALP_FR_0_coupling.hdf5","r")
    c1 = np.mean(f1['coupling'],axis=1)
    c2 = np.mean(f2['coupling'],axis=1)

    f3 = h5py.File("SR70_PIAALP_RT_1_3_coupling.hdf5","r")
    f4 = h5py.File("SR70_PIAALP_FR_1_3_coupling.hdf5","r")
    c3 = np.mean(f3['coupling'],axis=1)
    c4 = np.mean(f4['coupling'],axis=1)

    f5 = h5py.File("SR70_PIAALP_RM_0_coupling.hdf5","r")
    f6 = h5py.File("SR70_PIAALP_RM_1_3_coupling.hdf5","r")
    c5 = np.mean(f5['coupling'],axis=1)
    c6 = np.mean(f6['coupling'],axis=1)


    r = f1['rcores'][:]

    plt.plot(2*r,c1,color='steelblue',label='Fresnel, obstr. completely removed')
    plt.plot(2*r,c4,color='steelblue',label='Fresnel, obstr. shrunk by 1/3',ls='dashed')

    plt.plot(2*r,c2,color='indianred',label='ray traced, obstr. completely removed')
    plt.plot(2*r,c3,color='indianred',label='ray traced, obstr. shrunk by 1/3',ls='dashed')
    
    plt.plot(2*r,c5,color='forestgreen',label='remapped, obstr. completely removed')
    plt.plot(2*r,c6,color='forestgreen',label='remapped, obstr. shrunk by 1/3',ls='dashed')

    plt.ylim(0.45,0.75)
    plt.xlabel("core diameter (um)")
    plt.ylabel("coupling efficiency")
    plt.legend(frameon=False)
    plt.show()
    """

    ## obstruction size comp
    """
    f1 = h5py.File("SR70_PIAALP_RT_0_coupling.hdf5","r")
    f2 = h5py.File("SR70_PIAALP_FR_0_coupling.hdf5","r")
    c1 = np.mean(f1['coupling'],axis=1)
    c2 = np.mean(f2['coupling'],axis=1)

    f3 = h5py.File("SR70_PIAALP_RT_1_3_coupling.hdf5","r")
    f4 = h5py.File("SR70_PIAALP_FR_1_3_coupling.hdf5","r")
    c3 = np.mean(f3['coupling'],axis=1)
    c4 = np.mean(f4['coupling'],axis=1)

    f5 = h5py.File("SR70_PIAALP_RT_1_6_coupling.hdf5","r")
    f6 = h5py.File("SR70_PIAALP_FR_1_6_coupling.hdf5","r")
    c5 = np.mean(f5['coupling'],axis=1)
    c6 = np.mean(f6['coupling'],axis=1)

    f7 = h5py.File("SR70_PIAALP_RT_1_12_coupling.hdf5","r")
    f8 = h5py.File("SR70_PIAALP_FR_1_12_coupling.hdf5","r")
    c7 = np.mean(f7['coupling'],axis=1)
    c8 = np.mean(f8['coupling'],axis=1)


    r = f1['rcores'][:]
    d = 2*r

    plt.plot(d,c3,color='steelblue',label="RT , 1/3")
    plt.plot(d,c4,color='steelblue',ls='dashed',label="FR , 1/3")
    plt.plot(d,c5,color='forestgreen',label="RT , 1/6")
    plt.plot(d,c6,color='forestgreen',ls='dashed',label="FR , 1/6")
    plt.plot(d,c7,color='darkorange',label="RT , 1/12")
    plt.plot(d,c8,color='darkorange',ls='dashed',label="FR , 1/12")
    plt.plot(d,c1,color='indianred',label="RT , 0")
    plt.plot(d,c2,color='indianred',ls='dashed', label="FR , 0")

    plt.ylim(0.45,0.75)
    plt.xlabel("core diameter (um)")
    plt.ylabel("coupling efficiency")
    plt.legend(frameon=False)
    plt.show()


    # revised coupling calculations

    #fused silica @ 1um
    ncore = 1.4504
    nclad = 1.4504 - 5.5e-3
    rcore0 = 4.0

    f = h5py.File("SR70_PIAALP_RT_1_12.hdf5","r")

    psfs = f['psfs'][:]


    '''
    downsampled = []
    for psf in psfs:
        downsampled.append( 2*resize(psf,(512,512)) )

    downsampled = np.array(downsampled)
    '''


    downsampled = psfs
    opt_width = scale_u0_all_2(downsampled,rcore0,ncore,nclad,1.0) 

    print("optimal focal plane width for MMF: ",opt_width) 


    rcores = np.linspace(2,10,200)
    compute_coupling_vs_r_parallel2(downsampled,rcores,ncore,nclad,1.0,"SR70_PIAALP_RT_1_12_coupling",opt_width,rcore0)

    
    #coupling_vs_r_pupil("SR70_PIAA_1_6",rcores,ncore,nclad,1.0,opt_width,rcore,"SR70_PIAA_1_6_coupling",False)

    #####
    """



    #rcores = np.linspace(2,10,200)
    #coupling_vs_r_pupil("full_noPIAA_test",rcores,ncore,nclad,1.0,opt_width,rcore,"test_coupling_results")
    #coupling in pupil/focal tests below
    """
    f = h5py.File("quick_PIAA_test14.hdf5","r")
    f2 = h5py.File("quick_noPIAA_test14.hdf5","r")

    #psfs_unapo = np.load("psfs_oldgen.npy")
    psfs = f["psfs"]
    psfs2 = f["pupil_fields"]

    pupil_lps = np.load("pupil_lps.npy")
    pupil_lps_piaa = np.load("pupil_lps_piaa.npy")

    width = 56.29976221549987
    width2 = 56.189801742422716

    xa = ya = np.linspace(-width,width2,1024)
    xg,yg = np.meshgrid(xa,ya)

    i = 1

    rcore = 6.21
    modes = [(0,1),(1,1),(0,2),(2,1)]

    mode = (0,1)

    lpmode = normalize(LPmodes.lpfield(xg,yg,mode[0],mode[1],rcore,wl0,ncore,nclad))

    print(overlap(psfs[0],lpmode))

    print(overlap(psfs2[0],pupil_lps_piaa[0]))


    #psfs = np.load("psfs_k2y.npy")
    #w = 54.15074889973275

    #rcores = np.linspace(2,10,200)
    #compute_coupling_vs_r_parallel(resampled_psfs,rcores,ncore,nclad,6.21,wl0,"coupling_vs_r_PIAA_old")

    #54.239484645246364
    #54.15074889973275 @256 res
    """