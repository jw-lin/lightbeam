'''feeds into main.py establishing default set up for the simulation'''
import numpy as np
import optics
from mesh import RectMesh3D
from prop import Prop3D
from misc import read_rsoft,normalize,resize,overlap,getslices,gauss,write_rsoft,overlap_nonu,norm_nonu
import LPmodes
import argparse
import matplotlib.pyplot as plt
from config import *

parser = argparse.ArgumentParser("propagate field through system specified in top of runbpm.py")
parser.add_argument("--wl0",nargs="?",type=float)
parser.add_argument('--plot2D',action='store_true')
args = parser.parse_args()

if __name__ == "__main__":
    optic = optics.lant5big(rcore,rclad,rjack,ncore,nclad,njack,offset,zex,final_scale=1/scale)

    if args.wl0 is not None:
        wl0 = args.wl0

    mesh = RectMesh3D(xw0,yw0,zw,ds,dz,num_PML,xw_func,yw_func)
    xg,yg = mesh.xy.xg,mesh.xy.yg

    u0 =  np.load(u0_fname)

    mesh.xy.max_iters = max_remesh_iters
    mesh.sigma_max = sig_max

    xslice = getslices(savex,mesh.xa)
    yslice = getslices(savey,mesh.ya)
    zslice = getslices(savez,mesh.za)
    xyslice = np.s_[xslice,yslice]

    prop = Prop3D(wl0,mesh,optic,n0)

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

    print("final field power decomposition:")
    for i in range(len(modes)):
        print("mode"+str(i)+": ", np.power(overlap_nonu(u,modes[i],w),2))

    if args.plot2D:
        field = prop.field
        sx,sy,sz = len(savex),len(savey),len(savez)
        
        if sz==1:
            #xy mode
            extent = (*savex,*savey)
            title = r"$z = "+str(savez[0]) + "$"
            hlabel = r"$x$"
            vlabel = r"$y$"
        elif sx==1:
            #yz mode
            extent = (*savey,*savez)
            title = r"$x = "+str(savex[0]) + "$"
            hlabel = r"$y$"
            vlabel = r"$z$"
        elif sy==1:
            #xz mode
            extent = (*savex,*savez)
            title = r"$y = "+str(savey[0]) + "$"
            hlabel = r"$x$"
            vlabel = r"$z$"
        
        plt.xlabel(hlabel)
        plt.ylabel(vlabel)
        plt.title(title)
        plt.imshow(np.abs(field)/np.max(np.abs(field)),extent=extent,cmap="jet", origin = "lower")
        plt.axis("auto")
        plt.show()

        plt.plot(mesh.za,prop.totalpower,label="total power")
        plt.ylabel("throughput")
        plt.xlabel(r"$z$")
        plt.show()

        if u1_func is not None:
            plt.plot(mesh.za,prop.power,label="core0 power")
            
            plt.ylabel("throughput")
            plt.xlabel(r"$z$")
            plt.show()
