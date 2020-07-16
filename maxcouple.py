import numpy as np
import LPmodes
import matplotlib.pyplot as plt
from bisect import bisect_left
from misc import resize,normalize
from scipy.optimize import minimize
import dask
from dask.distributed import Client, progress

def overlap(u0,u1,c = False):
    if not c:
        return np.abs(np.sum( u0*np.conj(u1) ))
    else:
        return np.sum(u0*np.conj(u1))

def plot_modes(ncore,nclad,wl0):
    core_rs = [2.2]#np.linspace(1,10,10)
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

def get_power_in_mode(u0,rcore,ncore,nclad,mode,width,norm=None):
    if norm == None:
        norm = overlap(u0,u0)
    xa = np.linspace(-width,width,u0.shape[0])
    ya = np.linspace(-width,width,u0.shape[1])
    xg,yg = np.meshgrid(xa,ya)
    _power=0
    if mode[0] == 0:
        field = normalize(LPmodes.lpfield(xg,yg,mode[0],mode[1],rcore,wl0,ncore,nclad))
        power = np.power(overlap(u0,field),2)
        return power
    else:
        field0 = normalize(LPmodes.lpfield(xg,yg,mode[0],mode[1],rcore,wl0,ncore,nclad,"cos"))
        field1 = normalize(LPmodes.lpfield(xg,yg,mode[0],mode[1],rcore,wl0,ncore,nclad,"sin"))
        power0 = np.power(overlap(u0,field0),2)
        power1 = np.power(overlap(u0,field1),2)
        return power0+power1

def get_power_in_modes(width,rcore,ncore,nclad,u0,modes,norm=None):
    _power = 0
    for mode in modes:
        _power += get_power_in_mode(u0,rcore,ncore,nclad,mode,width,norm)
    return _power

def get_power_in_modes_neg(width,rcore,ncore,nclad,u0,modes,norm=None):
    _power = 0
    for mode in modes:
        _power += get_power_in_mode(u0,rcore,ncore,nclad,mode,width,norm)
    return -1*_power

def scale_u0_all(u0,modes,rcore,ncore,nclad,norm=None):

    res = minimize(get_power_in_modes_neg,100,args=(rcore,ncore,nclad,u0,modes,norm))
    w = np.abs(res.x[0])
    _power = get_power_in_modes(w,rcore,ncore,nclad,u0,modes,norm)
    return rcore,_power,w

def scale_u0_lp01(u0,rcore,ncore,nclad,wl0):
    norm = overlap(u0,u0)
    return scale_u0_all(u0,[(0,1)],rcore,ncore,nclad,norm)

def opt_single(k0,rcore,ncore,nclad,u0,norm):
    V = LPmodes.get_V(k0,rcore,ncore,nclad)
    modes = LPmodes.get_modes(V)
    res = minimize(get_power_in_modes_neg,100,args=(rcore,ncore,nclad,u0,modes,norm))
    w = np.abs(res.x[0])
    _power = get_power_in_modes(w,rcore,ncore,nclad,u0,modes,norm)
    return(rcore,w,_power)

if __name__ == "__main__":
    ncore = 1.4504
    nclad = 1.4504 - 5.5e-3

    wl0 = 1.
    k0 = 2*np.pi/wl0

    fname0 = "psfabb_lo.npy"
    u0 = np.load(fname0)
    u0 = center(u0)
    u0 = resize(u0,(401,401))
    u0 = normalize(u0)
    norm = overlap(u0,u0)

    rcores = np.linspace(2,30,300)
    lazy_results = []

    for rcore in rcores:
        client = Client(threads_per_worker=1, n_workers=4)
        lazy = dask.delayed(opt_single)(k0,rcore,ncore,nclad,u0,norm)
        lazy_results.append(lazy)

    futures = dask.persist(*lazy_results)
    results = dask.compute(*futures)
    np.savetxt("maxcoupling.txt",results)
