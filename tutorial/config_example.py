''' example configuration file for run_bpm_example.py '''
################################
## free space wavelength (um) ##
################################

wl0 = 1.55

########################
## lantern parameters ##
########################

zex = 30000 # length of lantern, in um
scale = 1/4 # how much smaller the input end is wrt the output end
rcore = 4.5 * scale # how large the lantern cores are, at the input (um)
rclad = 16.5 # how large the lantern cladding is, at the input (um)
ncore = 1.4504 + 0.0088 # lantern core refractive index
nclad = 1.4504 # cladding index
njack = 1.4504 - 5.5e-3 # jacket index

###################################
## sampling grid parameters (um) ##
###################################

xw0 = 128 # simulation zone x width (um)
yw0 = 128 # simulation zone y width (um)
zw = zex
ds = 1 # base grid resolution (um)
dz = 3 # z stepping resolution (um)

#############################
## mesh refinement options ##
#############################

ref_val = 1e-4 # controls adaptive meshing. lower -> more careful
remesh_every = 50 # how many z-steps to go before recomputing the adaptive mesh
max_remesh_iters = 6 # maximum amount of subdivisions when computing the adaptive mesh

xw_func = None # optional functions which allow the simulation zone to "grow" with z, which may save on computation time
yw_func = None

##################
## PML settings ##
##################

num_PML = 12
sig_max = 3. + 0.j

######################
## set launch field ##
######################
import numpy as np
import matplotlib.pyplot as plt
from lightbeam import LPmodes
from lightbeam.misc import normalize

xa = np.linspace(-xw0/2,xw0/2,int(xw0/ds)+1)
ya = np.linspace(-yw0/2,yw0/2,int(yw0/ds)+1)
xg,yg  = np.meshgrid(xa,ya)

u0 = normalize(LPmodes.lpfield(xg,yg,2,1,rclad,wl0,nclad,njack,'cos'))

fplanewidth = 0 # manually reset the width of the input field. set to 0 to match field extent with grid extent.

#####################
## reference index ##
#####################

n0 = 1.4504 
dynamic_n0 = False

###################
## monitor field ##
###################

monitor_func = None

#############################
## write out field dist to ##
#############################

writeto = None

# generate optical element
from lightbeam import optics
optic = optics.lant19(rcore,rclad,ncore,nclad,njack,rclad/3,zex,final_scale=1/scale)


#######################
## initial core locs ##
#######################

xpos_i = optic.core_locs[:,0]
ypos_i = optic.core_locs[:,1]

#####################
## final core locs ##
#####################

xpos = xpos_i / scale
ypos = ypos_i / scale
