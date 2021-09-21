''' example script for running the beamprop code in prop.py'''
import numpy as np
from mesh import RectMesh3D
from prop import Prop3D
from misc import normalize,overlap,getslices,overlap_nonu,norm_nonu
import LPmodes
import matplotlib.pyplot as plt
from config_example import *

if __name__ == "__main__":

    # mesh initialization (required)
    mesh = RectMesh3D(xw0,yw0,zw,ds,dz,num_PML,xw_func,yw_func)
    xg,yg = mesh.xy.xg,mesh.xy.yg

    mesh.xy.max_iters = max_remesh_iters
    mesh.sigma_max = sig_max

    # propagator initialization (required)
    prop = Prop3D(wl0,mesh,optic,n0)

    print('launch field')
    plt.imshow(np.real(u0))
    plt.show()

    # run the propagator (required)
    u,u0 = prop.prop2end(u0,xyslice=None,zslice=None,u1_func = u1_func,writeto=writeto,ref_val=ref_val,remesh_every=remesh_every,dynamic_n0=dynamic_n0,fplanewidth=fplanewidth)

    # compute power in output ports (optional)

    xg,yg = np.meshgrid(mesh.xy.xa,mesh.xy.ya,indexing='ij')

    w = mesh.xy.get_weights()

    xg0,yg0 = np.meshgrid(mesh.xy.xa0,mesh.xy.ya0,indexing='ij')
    w0 = mesh.xy.dx0*mesh.xy.dy0

    modes = []
    for x,y in zip(xpos,ypos):
        mode = norm_nonu(LPmodes.lpfield(xg-x,yg-y,0,1,2.2,wl0,ncore,nclad),w)
        modes.append(mode)
    
    modes0 = [] 
    for x,y in zip(xpos,ypos):
        mode = normalize(LPmodes.lpfield(xg0-x,yg0-y,0,1,2.2,wl0,ncore,nclad),w0)
        modes0.append(mode)

    SMFpower=0
    print("final field power decomposition:")
    for i in range(len(modes)):
        _p = np.power(overlap_nonu(u,modes[i],w),2)
        print("mode"+str(i)+": ", _p)
        SMFpower += _p
    
    print("total power in SMFs: ", SMFpower)

    # plotting (optional)
    print("final field dist:")
    plt.imshow(np.abs(u0)[num_PML:-num_PML,num_PML:-num_PML]) 
    plt.show()
