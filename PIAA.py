import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, UnivariateSpline
import hcipy as hc
import LPmodes
from scipy.special import jn,kn
from scipy.integrate import quad
from scipy.optimize import brentq

class PaddedWavefront(hc.Wavefront):
    def __init__(self, electric_field, wavelength=1, input_stokes_vector=None,pad_factor=1):
        shape = electric_field.shaped.shape
        assert shape[0] == shape[1] , "simulation region must be square!"
        assert shape[0]%2 == 0 , "simulation region must be an even number of pixels across!"

        res = shape[0]

        new_field = np.pad(electric_field.shaped,int(res*(pad_factor-1)/2)).flatten()

        self.radius = electric_field.grid[-1,0]
        self.total_radius = self.radius * pad_factor
        self.pad_factor = pad_factor
        self.beam_res = res
        self.res = pad_factor*res

        padded_electric_field = hc.Field(new_field,hc.make_pupil_grid(pad_factor*res,2*pad_factor*self.radius ))

        super().__init__(padded_electric_field, wavelength, input_stokes_vector)
    
    @staticmethod
    def remove_pad(wf,pad):
        res = wf.electric_field.shaped.shape[0]
        radius = wf.electric_field.grid[-1,0]
        beamres = int(res/pad)
        padpix = int( (pad-1)*beamres/2 )
        field = wf.electric_field.shaped[padpix:-padpix,padpix:-padpix]
        grid = hc.make_pupil_grid(beamres,2*radius)
        return hc.Wavefront(hc.Field(field.flatten(),grid),wavelength = wf.wavelength)

def make_remapping_gauss(res,obs,trunc=0.95,sigma='auto'):
    """make remapping arrays corresponding to gaussian INTENSITY output"""

    ra = np.linspace(0,1,res) #note that r=1 technically maps to infinity

    with np.errstate(divide='ignore'):
        ra_map =  np.sqrt(-2 * np.log(1 - trunc*np.power(ra,2)))
        ra_map[0] = 0
    
    if sigma == 'auto':
        #normalize so pre and post remapping have same domain
        ra_map /= np.max(ra_map)
    else:
        ra_map *= sigma
    ra_obs = np.linspace(obs,1,res)

    return ra_obs, ra_map

def make_remapping_gauss_annulus(res,obs,a,b,sigma='auto'):
    """make remapping arrays that map to gaussian 'annulus' with inner truncations a and outer truncation b (in units of sigma)"""
    ra = np.linspace(obs,1,res)
    out = np.sqrt( - 2*np.log( np.exp(-0.5*a*a) - (ra*ra-obs*obs)/(1-obs*obs) * (np.exp(-0.5*a*a) - np.exp(-0.5*b*b))) )

    if sigma == 'auto':
        out /= (np.max(out))
    else:
        out *= sigma
    
    return ra,out

def make_remapping_gauss_annulus_2(res,obs,a,b):

    ra = np.linspace(obs,1,res)

    num = np.exp( a**2 + b**2 ) * (1 - obs*obs)

    den = np.exp( a**2 ) * (ra*ra - obs*obs) + np.exp(b**2) * (1 - ra*ra)
    out = np.sqrt(np.log(num/den))
    out/=np.max(out)
    return ra,out

def LP02_back(rcore,ncore,nclad,wl,dist):
    '''compute back propped LP02 mode. wl and rcore same units'''
    k0 = 2*np.pi/wl
    V = LPmodes.get_V(k0,rcore,ncore,nclad)
    b = LPmodes.get_b(0,2,V)
    u = V*np.sqrt(1-b)
    v = V*np.sqrt(b) 

    def _inner_(rho):
        d = rho*2*np.pi*rcore/(wl*dist)
        term1 = (u * jn(1,u)*jn(0,d) - d * jn(0,u)*jn(1,d)) / (u**2-d**2)
        term2 = jn(0,u)/kn(0,u) * (d*kn(0,v)*jn(1,d) - v * kn(1,v)*jn(0,d)) / (v**2+d**2)

        return term1 - term2
    
    return _inner_

def LP0m_back(m,rcore,ncore,nclad,wl,dist,_norm=1):
    '''return a function for the backpropagated LP0m mode'''

    #convert to um, better for the integrator
    r_sc = rcore*1e6
    wl_sc = wl*1e6
    d_sc = dist*1e6

    k0 = 2*np.pi/wl_sc
    V = LPmodes.get_V(k0,r_sc,ncore,nclad)
    b = LPmodes.get_b(0,m,V)
    u = V*np.sqrt(1-b)
    v = V*np.sqrt(b) 

    def _inner_(rho):
        d = rho*2*np.pi*r_sc/(wl_sc*d_sc)
        term1 = (u * jn(1,u)*jn(0,d) - d * jn(0,u)*jn(1,d)) / (u**2-d**2)
        term2 = jn(0,u)/kn(0,v) * (d*kn(0,v)*jn(1,d) - v * kn(1,v)*jn(0,d)) / (v**2+d**2)

        return term1 - term2
    
    # compute normalization prefactor

    def _integrand_(rho):
        return np.power(_inner_(rho),2)*rho

    norm_fac = np.sqrt( _norm/(quad(_integrand_,0,np.inf,limit=100)[0]*2*np.pi) )

    def _inner2_(rho):
        d = rho*2*np.pi*r_sc/(wl_sc*d_sc)
        term1 = (u * jn(1,u)*jn(0,d) - d * jn(0,u)*jn(1,d)) / (u**2-d**2)
        term2 = jn(0,u)/kn(0,v) * (d*kn(0,v)*jn(1,d) - v * kn(1,v)*jn(0,d)) / (v**2+d**2)

        return norm_fac * (term1 - term2)

    return _inner2_

def make_remapping_LP0m(m,res,obs,beam_radius,rcore,ncore,nclad,wl,dist,inner_trunc=0,outer_trunc=None):
    if outer_trunc is None:
        outer_trunc = beam_radius

    back_lp0m_func = LP0m_back(m,rcore,ncore,nclad,wl,dist)

    r_in = beam_radius*obs
    inner_trunc = r_in * inner_trunc
    ra = np.linspace(r_in,beam_radius,res)

    def compute_encircled_power(_r,_norm=1):
        integrand = lambda r: r * np.power(back_lp0m_func(r*1e6),2)
        return _norm*quad(integrand,inner_trunc,_r)[0]*2*np.pi*1e12
    
    newnorm = 1/compute_encircled_power(outer_trunc)

    r_apos = []
    for r in ra:
        if r == r_in:
            r_apos.append(inner_trunc)
            continue
        if r == beam_radius:
            r_apos.append(outer_trunc)
            continue

        f = lambda rho: compute_encircled_power(rho,newnorm) - (r*r-r_in*r_in)/(beam_radius*beam_radius-r_in*r_in)

        r_apo = brentq(f,inner_trunc,outer_trunc)
        r_apos.append(r_apo)

    _r = np.linspace(inner_trunc,outer_trunc,100)
    _p = np.vectorize(compute_encircled_power)(_r)

    #plt.title("encircled power")
    #plt.plot(_r,_p)
    
    #plt.show()

    return ra, np.array(r_apos)

def make_PIAA_lenses(r1,r2,n1,n2,L):
    """ compute sag progiles for PIAA lenses to map r1 to r2 
        
        args: 
        r1,r2: remapping function (in array form)
        n1,n2: lens refractive index
        L: distance between lenses

        returns:
        z1,z2: height profiles of lenses
    """
    z1 = np.zeros_like(r1)
    z2 = np.zeros_like(r2)

    z1[0] = 0
    z2[0] = L

    for i in range(len(r1)-1):
        _r1,_r2 = r1[i], r2[i]
        B = z2[i] - z1[i]
        A = _r1 - _r2
        slope1 = A / ( n1 * np.sqrt(A**2 + B**2) - B )
        slope2 = A / ( n2 * np.sqrt(A**2 + B**2) - B )
        z1[i+1] = -1*slope1 * (r1[i+1]-_r1) + z1[i]
        z2[i+1] = -1*slope2 * (r2[i+1]-_r2) + z2[i]

    return z1,z2

def raytrace(r1,r2,z1,z2,n1,n2,L,skip=8):
    """use Snell's law to ray trace light paths between lenses"""

    #mirror lens profiles
    z1_p = np.concatenate( (z1[::-1][:-1],z1 ) )
    z2_p = np.concatenate( (z2[::-1][:-1],z2 ) )
    r1_p = np.concatenate( (-1*r1[::-1][:-1],r1 ) )
    r2_p = np.concatenate( (-1*r2[::-1][:-1],r2 ) )

    lens_thickness = 0.005

    #plt.axes().set_aspect('equal', 'datalim')

    #plot lens 1
    plt.plot(z1_p,r1_p,color='steelblue',lw=2)
    plt.plot((z1_p[-1]-lens_thickness,z1_p[-1]-lens_thickness), (r1_p[0],r1_p[-1]) ,color='steelblue',lw=2)

    plt.plot(z2_p,r2_p,color='steelblue',lw=2)
    plt.plot((z2_p[-1]+2*lens_thickness,z2_p[-1]+2*lens_thickness), (r2_p[0],r2_p[-1]) ,color='steelblue',lw=2)

    interpz1 = UnivariateSpline(r1,z1,s=0,k=1)
    interpz2 = UnivariateSpline(r2,z2,s=0,k=1)

    slope1 = interpz1.derivative(1)
    slope2 = interpz2.derivative(1)

    ray_rlaunch = r1[:-1]
    ray_zlaunch = z1[-1] - 2*lens_thickness

    for i in range(0,len(ray_rlaunch),skip):
        rayr = ray_rlaunch[i]
        _z1,_z2 = z1[i],z2[i]
        plt.plot( (ray_zlaunch,_z1 ), (rayr,rayr), color='k',lw=0.75)
        plt.plot( (ray_zlaunch,_z1 ), (-rayr,-rayr), color='k',lw=0.75)

        #first snell's law
        _slope = slope1(rayr)

        theta1 = np.arctan2(-1*_slope,1)
        theta2 = -np.arcsin(n1*np.sin(theta1)) + theta1

        _s = np.tan(theta2)

        ray_func1 = lambda z,z0,r0,: _s * (z-z0) + r0 

        z_between = np.linspace(_z1,_z2,100)

        plt.plot(z_between,ray_func1(z_between,_z1,rayr),color='k',lw=0.75)
        plt.plot(z_between,-1*ray_func1(z_between,_z1,rayr),color='k',lw=0.75)

        #second snell's law

        new_rayr = ray_func1(_z2,_z1,rayr)
        _slope2 = slope2(new_rayr)

        theta_l2 = np.arctan2(-1*_slope2,1)

        theta3 = theta_l2 - theta2
        theta4 = np.arcsin(np.sin(theta3)/n2) - theta_l2

        _s2 = np.tan(theta4)

        ray_func2 = lambda z,z0,r0,: _s2 * (z-z0) + r0 

        z_between_2 = np.linspace(_z2,z2[-1]+3*lens_thickness,20)

        plt.plot(z_between_2,ray_func2(z_between_2,_z2,new_rayr),color='k',lw=0.75)
        plt.plot(z_between_2,-1*ray_func2(z_between_2,_z2,new_rayr),color='k',lw=0.75)

    plt.show()

def prop_through_lens(wf,z,n):
    wf = wf.copy()

    k = 2*np.pi / wf.wavelength
    #thickness = np.max(z) - np.min(z)
    phase_delay = k*(n-1)*z #k*n*z + k * (thickness - z) 
    wf.electric_field *= np.exp(1j * phase_delay.flatten())

    return wf

def form_lens_height_maps(r1,r2,z1,z2,extent,res):
    """compute height of lens over uniform grid spanning simulation zone"""


    half_pixel = extent/res
    xa = ya = np.linspace(-extent+half_pixel,extent-half_pixel,res) # offset matches HCIPy's grids
    xg , yg = np.meshgrid(xa,ya)
    rg = np.sqrt(xg*xg+yg*yg)

    z1_func = UnivariateSpline(r1,z1,k=1,s=0,ext='const')
    z2_func = UnivariateSpline(r2,z2,k=1,s=0,ext='const')

    z1g = z1_func(rg)
    z1g-=np.min(z1g)

    z2g = z2_func(rg)*-1
    z2g-=np.min(z2g)

    return z1g,z2g

def fresnel_apodizer(pupil_grid,radius,sep,r1,r2,z1,z2,IOR1,IOR2):
    # pupil_grid: regular Cartesian grid across beam
    # pad: padding factor for Fresnel prop section
    # r1,r2: aperture remapping arrays (r1,r2 are normalized)
    # z1,z2: 1D lens sag profiles
    # IOR, IOR2: refractive indices of first and second lens
    # res is resolution across the beam

    #radius = pupil_grid.x[-1]
    res = pupil_grid.shape[0]

    #r1 *= radius 
    #r2 *= radius

    z1g , z2g = form_lens_height_maps(r1,r2,z1,z2,radius,res)

    prop =  hc.AngularSpectrumPropagator(pupil_grid,sep)

    def _inner_(wf):
        wf = prop_through_lens(wf,z1g,IOR1)
        wf = prop(wf)
        wf = prop_through_lens(wf,z2g,IOR2)
        return wf
    
    def _inner_backwards(wf):
        wf = prop_through_lens(wf,-z2g,IOR2)
        wf = prop.backward(wf)
        wf = prop_through_lens(wf,-z1g,IOR1)
        return wf

    return _inner_,_inner_backwards

if __name__ == "__main__":
    #plt.style.use("dark_background")

    rcore = 6.21
    ncore = 1.4504
    nclad = 1.4504 - 5.5e-3
    wl0 = 1.0
    sep = 25139
    beam_radius = 0.013/2*1e6
    piaa_sep = 0.12
    inner_trunc=0

    xa = ya = np.linspace(-6500,6500,1000)
    xg , yg = np.meshgrid(xa,ya)

    rg = np.sqrt(xg*xg+yg*yg)


    f = LP0m_back(2,rcore,ncore,nclad,wl0,sep)

    out = f(rg)*-1

    from misc import normalize

    out = normalize(out)

    plt.imshow( out)
    plt.show()

    r3,r4 = make_remapping_gauss_annulus(256,0.24,0,3)
    #r3,r4 = make_remapping_gauss_annulus_2(256,0.24,0,3/np.sqrt(2))
    r3 *= beam_radius*1e-6
    r4 *= beam_radius*1e-6
    z3,z4 = make_PIAA_lenses(r3,r4,1.48,1.48,piaa_sep)

    raytrace(r3,r4,z3,z4,1.48,1.48,piaa_sep,8)


    r1,r2 = make_remapping_LP0m(2,200,0.24,beam_radius,rcore,ncore,nclad,wl0,sep,inner_trunc,None)

    r1/=1e6
    r2/=1e6

    plt.plot(np.zeros_like(r1),r1,ls='None',color='white',marker='.')
    plt.plot(np.ones_like(r2),r2,ls='None',color='white',marker='.')
    plt.show()

    z1,z2 = make_PIAA_lenses(r1,r2,1.48,1.48,piaa_sep)

    plt.plot(z1,r1)
    plt.plot(z2,r2)
    plt.axis('equal')
    plt.show()


    r1,r2 = make_remapping_gauss_annulus_2(100,0.24,0.24/np.sqrt(2),3/np.sqrt(2))
    _r1,_r2 = make_remapping_gauss_annulus(100,0.24,0.24,3)

    plt.plot(np.zeros_like(r1),r1,marker='.',color='white',ls="None")
    plt.plot(np.ones_like(_r2),_r2,marker='.',color='steelblue',ls="None")
    plt.plot(np.ones_like(r2),r2,marker='.',color='white',ls="None")
    
    plt.ylim(0,1)
    plt.show()
