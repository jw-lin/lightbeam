################################
## free space wavelength (um) ##
################################

import numpy as np
wl0s = np.linspace(1,2.5,31)
#wl0s = [1.45,1.5,1.55,1.6]

########################################
## lantern parameters (assumed 5core) ##
########################################

zex = 13800
scale = 110/437
rcore = 6.5/2 * scale
rclad = 148/2 * scale
rjack = 437/2 * scale
ncore = 1.45397
nclad = 1.444
njack = 1.4431
offset = 50 * scale

###################################
## sampling grid parameters (um) ##
###################################

xw0 = 440
yw0 = 440
zw = 13800
ds = 2
dz = 2

######################################
## transverse mesh growth functions ##
######################################

xw_func = lambda z: (xw0 - 120) * z/zex + 120
yw_func = xw_func

#############################
## mesh refinement options ##
#############################

ucrit = 2.e-7
remesh_every = 20
max_remesh_iters = 8

##################
## PML settings ##
##################

num_PML = 12
sig_max = 3. + 0.j

###############################
## set launch field npy file ##
###############################

u0_header = "psf"

#####################
## reference index ##
#####################

n0 = 1.4488739430524165
dynamic_n0 = False

#############################################
## slicing of simulation data (broken atm) ##
#############################################

savex = [-220-ds*num_PML,220+2*num_PML]
savey = [0]
savez = [0,zex]

###################
## monitor field ##
###################

u1_func = None

#############################
## write out field dist to ##
#############################

writeto = None