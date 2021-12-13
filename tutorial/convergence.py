''' Example tests showcasing the potential savings in computation time offered by AMR. The waveguide used in this test is a 3 port lantern. '''

import matplotlib.pyplot as plt
import numpy as np
from lightbeam import optics,LPmodes
from lightbeam.mesh import RectMesh3D
from lightbeam.misc import norm_nonu,normalize,overlap_nonu
from lightbeam.prop import Prop3D


from scipy.optimize import curve_fit
outs = np.load('tutorial/convergence.npy')
outs = np.vstack( (outs, np.array([0.30718035, 0.53222632, 0.14978008])))
times = np.array([45,67,99,236,665,2842,14447])

diffs = np.sqrt(np.sum(np.power(outs[:-1,:] - outs[-1,:],2),axis=1))

f = lambda x,a,b:a*x+b
popt,pcov = curve_fit(f,np.log(times),np.log(diffs))
plt.figure(figsize=(4,3))
plt.plot(times,np.exp(f(np.log(times),*popt)),color='0.5',ls='dashed',lw=2)

plt.plot(1115,0.000119196,marker='.',ms=9,color='steelblue',label='AMR',ls='None')
plt.loglog(times,diffs,ls="None",marker='.',color='k',ms=9,label='uniform')
plt.xlabel('computation time (s)')
plt.ylabel('total error')
plt.legend(frameon=False)
plt.tight_layout()
plt.show()

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
#output = compute_port_power(1/64,False)
#print(output)
