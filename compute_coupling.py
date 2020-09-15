import numpy as np
import LPmodes
import matplotlib.pyplot as plt
from misc import resize,normalize
from scipy.optimize import minimize
import dask
from dask.distributed import Client, progress
import h5py
import hcipy as hc
import AOtele

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
    xa = np.linspace(-width,width,u0.shape[0])
    ya = np.linspace(-width,width,u0.shape[1])
    xg,yg = np.meshgrid(xa,ya)

    #xa = np.linspace(-width*3,width*3,u0.shape[0]*3)
    #ya = np.linspace(-width*3,width*3,u0.shape[1]*3)
    #xg,yg = np.meshgrid(xa,ya)

    s = u0.shape[0]

    if mode[0] == 0:
        lf = LPmodes.lpfield(xg,yg,mode[0],mode[1],rcore,wl0,ncore,nclad)
        field = normalize(lf)#[s:-s,s:-s]
        #field = normalize(LPmodes.lpfield(xg,yg,mode[0],mode[1],rcore,wl0,ncore,nclad))
        #field = normalize(LP01gauss(xg,yg,rcore,wl0,ncore,nclad))
        _power = np.power(overlap(u0,field),2)
        return _power
    else:
        #field0 = normalize(LPmodes.lpfield(xg,yg,mode[0],mode[1],rcore,wl0,ncore,nclad,"cos"))
        #field1 = normalize(LPmodes.lpfield(xg,yg,mode[0],mode[1],rcore,wl0,ncore,nclad,"sin"))
        lf0 = LPmodes.lpfield(xg,yg,mode[0],mode[1],rcore,wl0,ncore,nclad,"cos")
        lf1 = LPmodes.lpfield(xg,yg,mode[0],mode[1],rcore,wl0,ncore,nclad,"sin")
        
        field0 = normalize(lf0)#[s:-s,s:-s]
        field1 = normalize(lf1)#[s:-s,s:-s]
        
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

def scale_u0_all(u0,modes,rcore,ncore,nclad,wl0):
    res = minimize(get_power_in_modes_neg,80,args=(rcore,ncore,nclad,u0,modes,wl0))
    w = np.abs(res.x[0])
    _power = get_power_in_modes(w,rcore,ncore,nclad,u0,modes,wl0)
    return rcore,_power,w

def scale_u0_lp01(u0,rcore,ncore,nclad,wl0):
    return scale_u0_all(u0,[(0,1)],rcore,ncore,nclad,wl0)

def LP01gauss(xg,yg,rcore,wl0,ncore,nclad):
    V = LPmodes.get_V(2*np.pi/wl0,rcore,ncore,nclad)
    w = rcore * (0.65 + 1.619/V**1.5 + 2.879/V**6 - (0.016 + 1.561 * V**(-7)))

    rsq = xg*xg + yg*yg

    return normalize( np.exp(-rsq/w**2) )

def compute_coupling_vs_r_single(psf,w,rcores,ncore,nclad,rcore_ref,wl0):
    """compute coupling vs multiple MM fiber core sizes for a single psf"""

    k0 = 2*np.pi/wl0
    powers_all = []

    for rcore in rcores:
        V = LPmodes.get_V(k0,rcore,ncore,nclad)
        modes = LPmodes.get_modes(V)

        _p_all  = get_power_in_modes(w,rcore,ncore,nclad,psf,modes,wl0)
        powers_all.append(_p_all)

    return powers_all

def compute_coupling_vs_r_parallel(psfs,rcores,ncore,nclad,rcore0,wl0,fname):
    """compute avg coupling into MM fibers of various sizes given an input list of psfs. save to hdf5"""

    ncores = 8

    ## set up parallel stuff
    client = Client(threads_per_worker=1, n_workers=ncores)
    lazy_results = []

    ## now we need to pick a psf from the array of psfs and do an optimization to get the optimize the F#
    # which is arbitrary
    pick = 24
    #pick = 0
    
    print("selected psf for opt shown: ")
    plt.imshow(np.abs(psfs[pick]))
    plt.show()

    k0 = 2*np.pi/wl0

    V = LPmodes.get_V(k0,rcore0,ncore,nclad)
    modes = LPmodes.get_modes(V)
    opt = scale_u0_all(psfs[pick],modes,rcore0,ncore,nclad,wl0)
    focal_plane_width = opt[2]

    print("optimal focal plane width: " + str(focal_plane_width))

    for i in range(len(psfs)):
        psf = psfs[i]
        lazy_result = dask.delayed(compute_coupling_vs_r_single)(psf,focal_plane_width,rcores,ncore,nclad,rcore0,wl0)
        lazy_results.append(lazy_result)

    futures = dask.persist(*lazy_results)
    results = np.array(dask.compute(*futures))

    # save to file

    with h5py.File(fname+".hdf5", "w") as f:
        f.create_dataset("coupling", data = results)
        f.create_dataset("rcores", data = rcores)
        f.create_dataset("rcore0", data = rcore0)
        f.create_dataset("ncore", data = ncore)
        f.create_dataset("nclad", data = nclad)
        f.create_dataset("wl0", data = wl0)
        f.create_dataset("w", data = focal_plane_width)
        f.create_dataset("SMFcoupling", data = compute_SMF_coupling(psfs))

    # show prelim results

    results_avg = np.mean(results,axis=0)
    results_std = np.std(results,axis=0)

    plt.plot(rcores,results_avg,color='k',ls='None',marker='.')
    plt.fill_between(rcores,results_avg-results_std,results_avg+results_std,color='0.75',alpha=0.5)

    plt.show()

def parallel_avg_coupling(w,rcore,ncore,nclad,psfs,modes,wl0):
    lazy_powers = []

    for psf in psfs:
        lazy_powers.append( dask.delayed(get_power_in_modes)(w,rcore,ncore,nclad,psf,modes,wl0))

    futures = dask.persist(*lazy_powers)
    coupling = np.array(dask.compute(*futures))

    return -np.mean(coupling)

def scale_u0_all_2(psfs,rcore,ncore,nclad,wl0):
    k0 = 2*np.pi/wl0
    V = LPmodes.get_V(k0,rcore,ncore,nclad)
    modes = LPmodes.get_modes(V)

    res = minimize(parallel_avg_coupling,80,args=(rcore,ncore,nclad,psfs,modes,wl0))
    w = np.abs(res.x[0])
    return w

def compute_coupling_vs_r_parallel2(psfs,rcores,ncore,nclad,rcore0,wl0,fname,width=None):
    """compute avg coupling into MM fibers of various sizes given an input list of psfs. save to hdf5"""

    ncores = 8

    ## set up parallel stuff
    client = Client(threads_per_worker=1, n_workers=ncores)
    lazy_results = []

    ## compute avg coupling into selected fiber dimension (rcore0) and adjust focal plane scale to maximize coupling

    if width is None:
        opt = scale_u0_all_2(psfs,rcore0,ncore,nclad,wl0)
        focal_plane_width = opt
    else:
        focal_plane_width = width

    #k0 = 2*np.pi/wl0
    #V = LPmodes.get_V(k0,rcore0,ncore,nclad)
    #modes = LPmodes.get_modes(V)

    print("optimal focal plane width: " + str(focal_plane_width))

    for i in range(len(psfs)):
        psf = psfs[i]
        lazy_result = dask.delayed(compute_coupling_vs_r_single)(psf,focal_plane_width,rcores,ncore,nclad,rcore0,wl0)
        lazy_results.append(lazy_result)

    futures = dask.persist(*lazy_results)
    results = np.array(dask.compute(*futures))

    # save to file

    with h5py.File(fname+".hdf5", "w") as f:
        f.create_dataset("coupling", data = results)
        f.create_dataset("rcores", data = rcores)
        f.create_dataset("rcore0", data = rcore0)
        f.create_dataset("ncore", data = ncore)
        f.create_dataset("nclad", data = nclad)
        f.create_dataset("wl0", data = wl0)
        f.create_dataset("w", data = focal_plane_width)
        f.create_dataset("SMFcoupling", data = compute_SMF_coupling(psfs))

    # show prelim results

    results_avg = np.mean(results,axis=0)
    results_std = np.std(results,axis=0)

    plt.plot(rcores,results_avg,color='k',ls='None',marker='.')
    plt.fill_between(rcores,results_avg-results_std,results_avg+results_std,color='0.75',alpha=0.5)

    plt.show()


def get_IOR(wl):
    """ for fused silica """
    wl2 = wl*wl
    return np.sqrt(0.6961663 * wl2 / (wl2 - 0.0684043**2) + 0.4079426 * wl2 / (wl2 - 0.1162414**2) + 0.8974794 * wl2 / (wl2 - 9.896161**2) + 1)

def compute_coupling_at_wl(u,w,rcore,wl):
    """get coupling between psf generated by tele and MM fiber at given wl, assuming some rcore"""

    ncore = get_IOR(wl)
    nclad = ncore - 5.5e-3

    k = 2*np.pi/wl
    V = LPmodes.get_V(k,rcore,ncore,nclad)
    modes = LPmodes.get_modes(V)

    c = get_power_in_modes(w,rcore,ncore,nclad,u,modes,wl)

    return c

def compute_coupling_vs_wl_parallel(tele,DMshapes,ts,wls,wl0,rcore0,fname):
    """ compute avg coupling vs wavelength. Since psfs change with wavelength, 
        the psfs are generated on the fly from saved DM positions and atmospheric 
        turbulence states """

    #to ensure accurate psf generation, an AOtele object identical to the one used to generate DMshapes must be passed in.
    
    ncores = 8

    ## set up parallel stuff
    client = Client(threads_per_worker=1, n_workers=ncores)

    # get a perfect psf for Strehl calculations
    wf_pupil = hc.Wavefront(tele.ap,wavelength = wl0*1.e-6)
    wf_focus_perfect = tele.propagator.forward(wf_pupil)
    u_focus_perfect = AOtele.get_u(wf_focus_perfect)

    out = []

    #psf focal plane physical size
    w = None

    j = 0

    import copy

    for i in range(len(ts)):

        if i%100 == 0 and i!=0:
            print("psfs completed: ",j)

            tele.atmos.t = ts[i]
            tele.DM.actuators = DMshapes[i]

            # we need to select a psf to do F# opt
            # by default let's just use #100, the first psf we analyze
            if i == 100:

                u = tele.get_PSF(wf_pupil)
                print("first psf @ 1um has Strehl", AOtele.get_strehl(u,u_focus_perfect))
                plt.imshow(np.abs(u))
                plt.show()
                
                ncore = get_IOR(wl0)

                V = LPmodes.get_V(2*np.pi/wl0,rcore0,ncore,ncore-5.5e-3)
                modes = LPmodes.get_modes(V)

                r,p,focal_plane_width = scale_u0_all(u,modes,rcore0,ncore,nclad,wl0)
                print('optimal psf size: ',focal_plane_width)

            couplings = []

            # most likely can be sped up. I don't know how to parallelize the psf creation since the tele object can't be shared between threads.
            for wl in wls:
                wf = hc.Wavefront(tele.ap,wavelength = wl*1.e-6)
                u = tele.get_PSF(wf)
                c = dask.delayed(compute_coupling_at_wl)(u,focal_plane_width,rcore0,wl)
                couplings.append(c)
            
            futures = dask.persist(*couplings)
            results = np.array(dask.compute(*futures))
            out.append(results)
            
            j+=1
    
    out = np.array(out)

    # save to file

    ncores = get_IOR(wls)
    with h5py.File(fname+".hdf5", "w") as f:
        f.create_dataset("coupling", data = out)
        f.create_dataset("rcore0", data = rcore0)
        f.create_dataset("ncores", data = ncores)
        f.create_dataset("nclads", data = ncores - 5.5e-3)
        f.create_dataset("wls", data = wls)
        f.create_dataset("wl0", data = wl0)
        f.create_dataset("w", data = focal_plane_width)

    # show prelim results

    avgs = np.mean(out,axis=0)
    stds = np.std(out,axis=0)

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

    ncpucores = 8

    ## set up parallel stuff
    client = Client(threads_per_worker=1, n_workers=ncpucores)
    lazy_results = []

    ## now we need to pick a psf from the array of psfs and do an optimization to get the optimize the F#
    # choosing 42 is arbitrary
    pick = 42

    print("selected psf for opt shown: ")
    plt.imshow(np.abs(psfs[pick]))
    plt.show()

    k0 = 2*np.pi/wl0

    V = LPmodes.get_V(k0,rcore,ncore,nclad)
    modes = LPmodes.get_modes(V)
    print("modes: ", modes)
    opt = scale_u0_all(psfs[pick],modes,rcore,ncore,nclad,wl0)

    focal_plane_width = opt[2]

    for psf in psfs:
        lazy_result = dask.delayed(get_power_in_modes)(focal_plane_width,rcore,ncore,nclad,psf,modes,wl0)
        lazy_results.append(lazy_result)

    futures = dask.persist(*lazy_results)
    results = np.array(dask.compute(*futures))

    return np.mean(results)


def coupling_vs_r_pupilplane_single(tele:AOtele.AOtele,pupilfields,rcore,ncore,nclad,wl0):
    k0 = 2*np.pi/wl0

    fieldshape = pupilfields[0].shape

    #need to generate the available modes
    V = LPmodes.get_V(k0,rcore,ncore,nclad)
    modes = LPmodes.get_modes(V)
    lpfields = []

    for mode in modes:
        if mode[0] == 0:
            lpfields.append( normalize(LPmodes.lpfield(xg,yg,mode[0],mode[1],rcore,wl0,ncore,nclad)) )

        else:
            lpfields.append( normalize(LPmodes.lpfield(xg,yg,mode[0],mode[1],rcore,wl0,ncore,nclad,"cos")) )
            lpfields.append( normalize(LPmodes.lpfield(xg,yg,mode[0],mode[1],rcore,wl0,ncore,nclad,"sin")) )
    
    lppupilfields = []
    
    #then backpropagate
    for field in lpfields:
        wf = hc.Wavefront(hc.Field(field.flatten(),tele.focalgrid),wavelength = wl0*1.e-6)
        pupil_wf = tele.back_propagate(wf,True)
        lppupilfields.append( pupil_wf.electric_field.reshape(fieldshape) )
    
    lppupilfields = np.array(lppupilfields)
    pupilfields = np.array(pupilfields)

    #compute total overlap
    powers = np.sum( pupilfields * lppupilfields , axes = (1,2) )

    return np.mean(powers),np.stddev(powers)

if __name__ == "__main__":
    ## in wavelength space:
    # blue edge of Y located at 6 - 8 mode number transition when rcore = 6.21 um
    # red edge of J located at 3 - 6 mode # transition when rcore = 6.41 um
    
    wl0 = 1.0 #um

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

    #_norm3 = np.sqrt(28444444444444.445)
    #_norm = np.sqrt(256000000000000.3)
    #_norm = np.sqrt(113777777777777.69)
    #power norm fac: 28444444444444.445+0j

    _norm2 = np.sqrt(455111111111111.1)

    #fused silica @ 1um
    ncore = 1.4504
    nclad = 1.4504 - 5.5e-3

    f = h5py.File("quick_PIAA_test13.hdf5","r")
    f2 = h5py.File("quick_noPIAA_test12.hdf5","r")

    #psfs_unapo = np.load("psfs_oldgen.npy")
    psfs = f["psfs"]
    psfs2 = f2["psfs"]

    print(psfs2[0].shape)

    """

    resampled_psfs = []
    for psf in psfs:
        resampled_psfs.append(normalize(resize(psf,(256,256))))

    """
    i = 1

    print(np.sum(psfs[i]*np.conj(psfs[i])))

    rcore = 6.21
    modes = [(0,1),(1,1),(0,2),(2,1)]

    #rcore = 4
    #modes = [(0,1),(1,1)]

    _w = scale_u0_all(psfs[i],modes,rcore,ncore,nclad,1)[2]
    _w2 = scale_u0_all(psfs2[i],modes,rcore,ncore,nclad,1)[2]

    print(_w,_w2)

    print(get_power_in_modes(_w,rcore,ncore,nclad,psfs[i],modes,1))
    print(get_power_in_modes(_w2,rcore,ncore,nclad,psfs2[i],modes,1))

    
    plt.imshow(np.abs(psfs[i]))
    plt.show()

    plt.imshow(np.abs(psfs2[i]))
    plt.show()




    #psfs = np.load("psfs_k2y.npy")
    #w = 54.15074889973275

    #rcores = np.linspace(2,10,200)
    #compute_coupling_vs_r_parallel(resampled_psfs,rcores,ncore,nclad,6.21,wl0,"coupling_vs_r_PIAA_old")

    #54.239484645246364
    #54.15074889973275 @256 res

    """
    seed = 123456789
    tele = AOtele.make_tele(0.108=,seed)
    DMshapes = np.load("DM_sr0pt2_fp0pt114.npy")
    wl_arr = np.linspace(1.3,1.4,10)
    freq = 1000
    run_for = 10
    t_arr = np.linspace(0,run_for, int(run_for*freq) + 1)    

    compute_coupling_vs_wl_parallel(tele,DMshapes,t_arr,wl_arr,1.,6.21,"coupling_vs_wl_sr0pt2")
    """