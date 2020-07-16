'''feeds into main.py establishing default set up for the simulation'''
import numpy as np
import optics
from mesh import RectMesh3D
from prop import Prop3D
from misc import read_rsoft,normalize,resize,overlap,getslices,gauss,write_rsoft,overlap_nonu,norm_nonu
import LPmodes
import argparse
import matplotlib.pyplot as plt
from config_batch import *
import dask
from dask.distributed import Client, progress

def main(wl,u0f):
    u0 = np.load(u0f)

    optic = optics.lant5big(rcore,rclad,rjack,ncore,nclad,njack,offset,zex,final_scale=1/scale)

    mesh = RectMesh3D(xw0,yw0,zw,ds,dz,num_PML,xw_func,yw_func)
    xg,yg = mesh.xy.xg,mesh.xy.yg

    mesh.xy.max_iters = max_remesh_iters
    mesh.sigma_max = sig_max

    xslice = getslices(savex,mesh.xa)
    yslice = getslices(savey,mesh.ya)
    zslice = getslices(savez,mesh.za)
    xyslice = np.s_[xslice,yslice]

    prop = Prop3D(wl,mesh,optic,n0)
    u = prop.prop2end(u0,xyslice=xyslice,zslice=zslice,u1_func = u1_func,writeto=writeto,ucrit=ucrit,remesh_every=remesh_every,dynamic_n0=dynamic_n0)

    xg,yg = np.meshgrid(mesh.xy.xa,mesh.xy.ya,indexing='ij')

    w = mesh.xy.get_weights()

    ## final field power decomposition (add more to modes as necessary) ##

    mode0 = norm_nonu(LPmodes.lpfield(xg,yg,0,1,6.5/2,wl0,ncore,nclad),w)
    mode1 = norm_nonu(LPmodes.lpfield(xg-50,yg,0,1,6.5/2,wl0,ncore,nclad),w)
    mode2 = norm_nonu(LPmodes.lpfield(xg,yg-50,0,1,6.5/2,wl0,ncore,nclad),w)
    mode3 = norm_nonu(LPmodes.lpfield(xg+50,yg,0,1,6.5/2,wl0,ncore,nclad),w)
    mode4 = norm_nonu(LPmodes.lpfield(xg,yg+50,0,1,6.5/2,wl0,ncore,nclad),w)

    modes = [mode0,mode1,mode2,mode3,mode4]

    output = []

    for i in range(len(modes)):
        output.append(np.power(overlap_nonu(u,modes[i],w),2))
    
    output.append(overlap_nonu(u,u,w))

    return output

if __name__ == "__main__":
    client = Client(threads_per_worker=4, n_workers=16)

    lazy_results = []

    for i in range(len(wl0s)):
        wl0 = wl0s[i]
        u0_fname = u0_header+str(i)+".npy"

        lazy_result = dask.delayed(main)(wl0,u0_fname)
        lazy_results.append(lazy_result)

    futures = dask.persist(*lazy_results)
    results = dask.compute(*futures)

    np.savetxt("output.txt",results)







        

