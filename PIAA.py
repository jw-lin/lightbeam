import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, UnivariateSpline
from misc import resize2,timeit
import hcipy as hc

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

def make_remapping_gauss_annulus_2(res,obs,a,b,sigma=1):

    ra = np.linspace(obs,1,res)

    num = np.exp( a**2 + b**2 ) * (1 - obs*obs)

    den = np.exp( a**2 ) * (ra*ra - obs*obs) + np.exp(b**2) * (1 - ra*ra)
    out = sigma * np.sqrt(np.log(num/den))
    return ra,out

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

    plt.axes().set_aspect('equal', 'datalim')

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
        plt.plot( (ray_zlaunch,_z1 ), (rayr,rayr), color='white',lw=0.75)
        plt.plot( (ray_zlaunch,_z1 ), (-rayr,-rayr), color='white',lw=0.75)

        #first snell's law
        _slope = slope1(rayr)

        theta1 = np.arctan2(-1*_slope,1)
        theta2 = -np.arcsin(n1*np.sin(theta1)) + theta1

        _s = np.tan(theta2)

        ray_func1 = lambda z,z0,r0,: _s * (z-z0) + r0 

        z_between = np.linspace(_z1,_z2,100)

        plt.plot(z_between,ray_func1(z_between,_z1,rayr),color='white',lw=0.75)
        plt.plot(z_between,-1*ray_func1(z_between,_z1,rayr),color='white',lw=0.75)

        #second snell's law

        new_rayr = ray_func1(_z2,_z1,rayr)
        _slope2 = slope2(new_rayr)

        theta_l2 = np.arctan2(-1*_slope2,1)

        theta3 = theta_l2 - theta2
        theta4 = np.arcsin(np.sin(theta3)/n2) - theta_l2

        _s2 = np.tan(theta4)

        ray_func2 = lambda z,z0,r0,: _s2 * (z-z0) + r0 

        z_between_2 = np.linspace(_z2,z2[-1]+3*lens_thickness,20)

        plt.plot(z_between_2,ray_func2(z_between_2,_z2,new_rayr),color='white',lw=0.75)
        plt.plot(z_between_2,-1*ray_func2(z_between_2,_z2,new_rayr),color='white',lw=0.75)

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

    xa = ya = np.linspace(-extent,extent,res)
    xg , yg = np.meshgrid(xa,ya)
    rg = np.sqrt(xg*xg+yg*yg)

    z1_func = UnivariateSpline(r1,z1,k=3,s=0,ext='const')
    z2_func = UnivariateSpline(r2,z2,k=3,s=0,ext='const')

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

    r1 *= radius 
    r2 *= radius

    z1g , z2g = form_lens_height_maps(r1,r2,z1,z2,radius,res)

    prop =  hc.AngularSpectrumPropagator(pupil_grid,sep)

    def _inner_(wf):
        wf = prop_through_lens(wf,z1g,IOR1)
    
        wf = prop(wf)
        
        wf = prop_through_lens(wf,z2g,IOR2)
        wf.total_power=1
        return wf
    
    def _inner_backwards(wf):
        wf = prop_through_lens(wf,-z2g,IOR2)
        
        wf = prop.backward(wf)
        
        wf = prop_through_lens(wf,-z1g,IOR1)
        wf.total_power=1
        return wf

    return _inner_,_inner_backwards

if __name__ == "__main__":
    plt.style.use("dark_background")

    radius = 0.013/2 #5
    sep =  0.12#10
    IOR = 1.48 #acrylic at 1 um
    res = 600 #600 seems to be the critical value
    pad = 1
    pupil_grid = hc.make_pupil_grid(res,2*radius)

    r1,r2 = make_remapping_gauss_annulus(res,0.23,0,3)
    z1,z2 = make_PIAA_lenses(r1*radius,r2*radius,IOR,IOR,sep) ## fix radius dependence here!

    apodizer = fresnel_apodizer(pupil_grid,sep,pad,r1,r2,z1,z2,IOR,IOR)

    keck_pupil_hires = np.array(fits.open("pupil_KECK_high_res.fits")[0].data,dtype=np.float32)
    ap_arr = resize2(keck_pupil_hires, (res,res)).flatten()
    ap = hc.Field(ap_arr,pupil_grid)

    plt.imshow(ap.real.reshape((600,600)))
    plt.show()
    wf = PaddedWavefront(ap,wavelength=1.e-6, pad_factor=pad)
    wf.total_power = 1*pad*pad

    wf = apodizer(wf)

    #hc.imshow_field(wf.electric_field)
    #plt.show()

    p = wf.power.reshape(pad*res,pad*res)
    fig,ax = plt.subplots(figsize=(4,4))
    ax.axis("off")
    plt.xlim(-radius,radius)
    plt.ylim(-radius,radius)
    plt.imshow(wf.power.reshape(pad*res,pad*res),vmax=3e-5,extent=(-pad*radius,pad*radius,-pad*radius,pad*radius))

    plt.show()

    """
    plt.xlim(-radius,radius)
    plt.plot(np.linspace(-pad*radius,pad*radius,pad*res),p[int(pad/2*res)])
    plt.ylim(0,1.2e-4)
    plt.show()
    """

    """
    plt.xlim(-pad*radius,pad*radius)
    plt.plot(ya, p[:,int(2*res)])
    plt.show()
    """