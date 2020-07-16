import numpy as np
import LPmodes
from bisect import bisect_left
import matplotlib.pyplot as plt
import h5py

def c_vs_d(fname,sr=None):
    fig, (ax0,ax1) = plt.subplots(2,1,sharex=True,figsize=(8,5))

    f = h5py.File(fname, 'r')
    results = f["coupling"][:]
    rcores = f["rcores"][:]
    ncore = f["ncore"][()]
    nclad = f["nclad"][()]
    wl0 = f["wl0"][()]
    rcore0 = f["rcore0"][()]
    SMFcoupling = f["SMFcoupling"][()]

    k0 = 2*np.pi / wl0
    diams = 2 * rcores
    Vs = LPmodes.get_V(k0,rcores,ncore,nclad)

    nummodes=[]
    for V in Vs:
        total = 0
        modes = LPmodes.get_modes(V)
        for mode in modes:
            if mode[0]==0:
                total+=1
            else:
                total+=2
        nummodes.append(total)

    results_avg = np.mean(results,axis=0)
    results_std = np.std(results,axis=0)

    ax0.plot(diams,results_avg,color='steelblue')
    if results.ndim == 2:
        ax0.fill_between(diams,results_avg-results_std,results_avg+results_std,color='0.75',alpha=0.5)

    ax1.plot(diams,nummodes,color='indianred')
    ax1.set_ylabel("number of modes")

    all_modes = LPmodes.get_modes(Vs[-1])
    locs = []

    sixmode_loc = NotImplementedError

    for mode in all_modes:
        if mode == (0,1):
            locs.append(rcores[0])
        else:
            _V = LPmodes.get_mode_cutoffs(mode[0],mode[1])[-1]
            idx = bisect_left(Vs,_V)
            locs.append(rcores[idx])
            if mode == (3,1):
                sixmode_loc = rcores[idx]

    for r,mode in zip(locs,all_modes):
        dy = 0.7
        y = mode[0]*dy + 0.5
        ax0.axvline(x=2*r,color='0.5',ls='dotted')
        ax1.axvline(x=2*r,color='0.5',ls='dotted')
        ax1.annotate("({},{})".format(mode[0],mode[1]),(2*r,y))

    ax1.set_xlabel(r"core diameter ($\mu$m)")
    ax0.set_ylabel("coupling efficiency")

    ax0.axvline(x = 2 * rcore0,color='k')
    ax1.axvline(x = 2 * rcore0,color='k')

    ax0.set_ylim(0,1)
    ax1.set_ylim(ymin=0)

    plt.subplots_adjust(hspace=0,wspace=0)

    ax0.annotate("SMF coupling", (18,SMFcoupling+0.03))
    ax0.axhline(y = SMFcoupling,color='k',ls = 'dashed')


    ax0.set_yticks(np.linspace(0,1,6))
    ax0.set_yticks(np.linspace(0,1,11),minor=True)
    ax0.yaxis.grid(which="both",alpha=0.5)
    #ax1.yaxis.grid()

    ax0.axvspan(4,2*sixmode_loc,color='beige',alpha=0.5,zorder=-100)
    ax1.axvspan(4,2*sixmode_loc,color='beige',alpha=0.5,zorder=-100)
    if sr is not None:
        ax0.set_title("coupling vs core diam. for "+sr+"% Strehl beam")

    plt.show()

    f.close()

def c_vs_wl(fname):
    f = h5py.File(fname, 'r')
    results = f["coupling"][:]
    rcore0 = f["rcore0"][()]
    ncores = f["ncores"][:]
    nclads = f["nclads"][:]
    wls = f["wls"][:]
    wl0 = f["wl0"][()]

    results_avg = np.mean(results,axis=0)
    results_std = np.std(results,axis=0)

    Vs = LPmodes.get_V(2*np.pi/wls,rcore0,ncores,nclads)

    nummodes=[]
    for V in Vs:
        total = 0
        modes = LPmodes.get_modes(V)
        for mode in modes:
            if mode[0]==0:
                total+=1
            else:
                total+=2
        nummodes.append(total)

    plt.plot(wls,results_avg,color='k',ls='None',marker='.')

    plt.fill_between(wls,results_avg-results_std,results_avg+results_std,color='0.75',alpha=0.5)

    # outline J band, Y band
    plt.axvspan(1.02-0.06,1.02+0.06,alpha=0.35,color="burlywood",zorder=-100)
    plt.axvspan(1.220-0.213/2,1.220+0.213/2,alpha=0.25,color="indianred",zorder=-100)

    # outline 6 mode region
    last = nummodes[0]
    print(nummodes)
    for i in range(len(nummodes)):
        if last != nummodes[i]:
            plt.axvline(x = wls[i], color='k', ls = 'dashed')
        last = nummodes[i]

    plt.xlim(0.9,1.4)
    plt.ylim(0.,1.)
    plt.xlabel("wavelength (um)")
    plt.ylabel("coupling efficiency")
    plt.show()

if __name__ == "__main__":
    c_vs_d("coupling_vs_r_oldgen.hdf5","12")