''' Example tests showcasing the potential savings in computation time offered by AMR. The waveguide used in this test is a 3 port lantern. '''

import matplotlib.pyplot as plt
import numpy as np
import optics
from mesh import RectMesh3D
import LPmodes
from misc import norm_nonu, normalize, overlap_nonu
from prop import Prop3D

def compute_port_power(ds,AMR=False,ref_val=2e-4,max_iters=5,remesh_every=50):
    ''' Compute the output port powers for a 3 port lantern, given some simulation parameters. '''
    # wavelength
    wl = 1.0 #um

    # mesh 
    xw = 64 #um
    yw= 64 #um
    zw = 10000 #um
    num_PML = int(4/ds) # number of cells
    dz = 1

    mesh = RectMesh3D(xw,yw,zw,ds,dz,num_PML)
    mesh.xy.max_iters = max_iters

    xg,yg = mesh.xg[num_PML:-num_PML,num_PML:-num_PML] , mesh.yg[num_PML:-num_PML,num_PML:-num_PML]

    # optic (3 port lantern)
    taper_factor = 4
    rcore = 2.2/taper_factor # INITIAL core radius
    rclad = 4
    nclad = 1.4504
    ncore = nclad + 0.0088
    njack = nclad - 5.5e-3

    lant = optics.lant3big(rcore,rclad,ncore,nclad,njack,rclad/2,zw,final_scale=taper_factor)

    def launch_field(x,y):
        return normalize(np.exp(10.j*x*wl/xw)*LPmodes.lpfield(x,y,0,1,rclad,wl,ncore,nclad))

    # propagation

    prop = Prop3D(wl,mesh,lant,nclad)

    if AMR:
        u , u0 = prop.prop2end(launch_field,ref_val=ref_val,remesh_every=remesh_every)
    else:
        u = u0 = prop.prop2end_uniform(launch_field(xg,yg))

    xg,yg = np.meshgrid(mesh.xy.xa,mesh.xy.ya,indexing='ij')

    w = mesh.xy.get_weights()

    # get the output port powers

    output_powers = []
    for pos in lant.final_core_locs:
        _m = norm_nonu(LPmodes.lpfield(xg-pos[0],yg-pos[1],0,1,rcore*taper_factor,wl,ncore,nclad),w)
        output_powers.append(np.power(overlap_nonu(_m,u,w),2))

    return np.array(output_powers)

# example of how to use
output = compute_port_power(1/64,False)
print(output)
