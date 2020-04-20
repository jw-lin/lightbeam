'''feeds into main.py establishing default set up for the simulation'''
import numpy as np
import optics
from mesh import RectMesh3D
from prop import Prop3D
from misc import read_rsoft,normalize,resize,overlap,getslices,gauss,write_rsoft
import LPmodes
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser("propagate field through system specified in top of runbpm.py")
parser.add_argument("--wl0",nargs="?",type=float)
parser.add_argument('--plot2D',action='store_true')
args = parser.parse_args()

## to do ## 

# this config file needs to be completely rewritten into a more user-friendly format
# the slice/subvolume selection for the field data is fucked

## current features ##
# FD-BPM (ADI for 3D, GD method for 4th order transversal accuracy, trapezoidal rule for 2nd order longitudinal accuracy)
# Fast(ish) tridiagonal matrix solver
# An adaptive grid that can grow with tapered waveguides
# Smoothing (antialiasing) of IOR distribution for better convergence


################################
###       BPM Settings       ###
################################


################################
## define the optical system  ##

zex = 13800
scale = 110/437
rcore = 6.5/2 * scale
rclad = 148/2 * scale
rjack = 437/2 * scale
ncore = 1.45397
nclad = 1.444
njack = 1.4431
offset = 50 * scale

#lant = optics.lant5(rcore,rclad,rjack,ncore,nclad,njack,offset,zex,final_scale=1/scale)
lant = optics.lant5big(rcore,rclad,rjack,ncore,nclad,njack,offset,zex,final_scale=1/scale)

#######################################
## set optical item being simulated) ##
optic = optics.OpticSys([],njack) #lant#

########################################
## set the free space wavelength (um) ##
wl0 = 1.55 if args.wl0 is None else args.wl0

#########################
## number of PML cells ##
PML = 12

###################################
## sampling grid parameters (um) ##
xw = 440
yw = 440
zw = 13800
ds = 2
dz = 10

#################################################
## set how transversal size of mesh grows w/ z ##

xw_func = lambda z: 440 #(440 - 120) * z/zex + 120
yw_func = xw_func

mesh = RectMesh3D(xw,yw,zw,ds,dz,PML,xw_func,yw_func)
xg,yg = mesh.xy.xg,mesh.xy.yg

############################
## set launch field (req) ##
u0 = np.load("tg_launch_his.npy")# np.load("PSF0hi.npy") #

########################
## truncate the field ##
xa_in = np.linspace(-xw/2,xw/2,u0.shape[0])
ya_in = np.linspace(-yw/2,yw/2,u0.shape[1])
xg_in,yg_in = np.meshgrid(xa_in,ya_in)
rsq = xg_in*xg_in + yg_in*yg_in
init_rad = 110/2.
u0[rsq>=init_rad**2] = 0

u1_func = lambda xg,yg :  normalize(LPmodes.lpfield(xg,yg,0,1,6.5/2,wl0,ncore,nclad))

#####################
## reference index ##
n0 = 1.4488739430524165 # this is the effective IOR of sm mode at end of lantern
dynamic_n0 = False #this computes a power-weighted avergae of the actual IOR, sets avg to n0

#####################
## mesh refinement ##
ucrit = 4.e-7 # the lower the slower!
remesh_every = 10 #^^^
max_refinement_ratio = 128

mesh.xy.max_ratio = max_refinement_ratio
mesh.xy.max_iters = 8

################################
## slicing of simulation data ##

# save the entire data-cube or just a slice

savex = [-220-ds*PML,220+2*PML] #this stuff's kinda broken atm
savey = [0]
savez = [0,13800]

##################################################################
## (optional) which file the program should write field dist to ##
writeto = None #also broken

###########################
###    actual code      ###
###########################

if __name__ == "__main__":
    xslice = getslices(savex,mesh.xa)
    yslice = getslices(savey,mesh.ya)
    zslice = getslices(savez,mesh.za)
    xyslice = np.s_[xslice,yslice]

    prop = Prop3D(wl0,mesh,optic,n0)

    u0 = prop.prop2end(u0,xyslice=xyslice,zslice=zslice,u1_func = u1_func,writeto=writeto,ucrit=ucrit,remesh_every=remesh_every,dynamic_n0=dynamic_n0)

    xg,yg = np.meshgrid(mesh.xy.xa0,mesh.xy.ya0,indexing='ij')

    ######################################################################
    ## final field power decomposition (add more to modes as necessary) ##

    mode0 = normalize(LPmodes.lpfield(xg,yg,0,1,6.5/2,wl0,ncore,nclad))
    mode1 = normalize(LPmodes.lpfield(xg-50,yg,0,1,6.5/2,wl0,ncore,nclad))
    mode2 = normalize(LPmodes.lpfield(xg,yg-50,0,1,6.5/2,wl0,ncore,nclad))
    mode3 = normalize(LPmodes.lpfield(xg+50,yg,0,1,6.5/2,wl0,ncore,nclad))
    mode4 = normalize(LPmodes.lpfield(xg,yg+50,0,1,6.5/2,wl0,ncore,nclad))

    modes = [mode0,mode1,mode2,mode3,mode4]

    if len(modes)>0:
        print("final field power decomposition:")
        for i in range(len(modes)):
            print("mode"+str(i)+": ", np.power(overlap(u0,modes[i]),2)/overlap(u0,u0) )

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
