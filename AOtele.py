import numpy as np
import hcipy as hc
import matplotlib.pyplot as plt
from misc import normalize,resize,printProgressBar
from os import path
import time
import LPmodes
import h5py

def get_u(wf):
    field = wf.electric_field
    size = int(np.sqrt(field.shape[0]))
    shape=(size,size)
    reals,imags = np.array(np.real(field)),np.array(np.imag(field))
    u = reals + 1.j*imags
    u = u.reshape(shape)
    u = normalize(u)
    return u

def get_phase(wf):
    phase = wf.phase
    size = int(np.sqrt(phase.shape[0]))
    shape=(size,size)
    phase = phase.reshape(shape)
    return phase

def get_strehl(u,u_perfect):
    return np.max(u*np.conj(u))/np.max(u_perfect*np.conj(u_perfect))

class AOtele: 
    
    #imaged star parameters
    zero_magnitude_flux = 3.9e10 #phot/s
    star_mag = 0

    def __init__(self, diameter, fnum, wavelength, num_DM_acts = 30, wavelength_0 = None, obstr_frac = 0.3):

        if wavelength_0 is None:
            wavelength_0 = wavelength

        self.diameter = diameter
        self.fnum = fnum
        self.num_acts = num_DM_acts
        self.obstr_frac = obstr_frac
        self.wavelength = wavelength
        
        num_pupil_pixels = 60*2
        pupil_pixel_samps = 56*2
        pupil_grid_diam = diameter * num_pupil_pixels / pupil_pixel_samps
        
        self.pupil_grid = hc.make_pupil_grid(num_pupil_pixels,diameter = pupil_grid_diam)
        ap = hc.make_obstructed_circular_aperture(diameter,obstr_frac)
        self.ap = hc.evaluate_supersampled(ap,self.pupil_grid,6)

        act_spacing = diameter / num_DM_acts
        influence_funcs = hc.make_gaussian_influence_functions(self.pupil_grid,num_DM_acts,act_spacing)
        self.DM = hc.DeformableMirror(influence_funcs)
        
        self.pwfs = hc.PyramidWavefrontSensorOptics(self.pupil_grid,wavelength_0=wavelength_0)
        
        self.detector = hc.NoiselessDetector()
        self.dt = 1 #integration time in seconds (?)

        self.atmos = None
        self.coherence_length = None

        # linear resolution in pixels is 2 * q * num_airy
        self.focal_grid = hc.make_focal_grid(q = 12, num_airy = 20, f_number=fnum, reference_wavelength=wavelength_0)

        self.ref_image = None
        self.rmat = None

        self.propagator = hc.FraunhoferPropagator(self.pupil_grid,self.focal_grid,focal_length=diameter*fnum)

        self.t_arr = None
        self.DMshapes = None


    def read_out(self, wf, poisson = False):
        self.detector.integrate(wf,self.dt)

        if poisson:
            image = hc.large_poisson(self.detector.read_out()).astype(np.float)
        else:
            image = self.detector.read_out()
        image /= image.sum()
        return image

    def calibrate_DM(self, rcond = 0.1, fname_append=None):
        """Calculate and save the imagae reconstruction matrix for the AO system"""
        
        if fname_append is not None:
            fname = "rmat_"+str(self.DM.num_actuators)+"_"+str(rcond).replace(".","")+"_"+fname_append
        else:
            fname = "rmat_"+str(self.DM.num_actuators)+"_"+str(rcond).replace(".","")

        #first we make a reference image for an unaberrated wavefront passing through the pwfs
        wf = hc.Wavefront(self.ap,wavelength = self.wavelength)
        wf.total_power = 1
        wf_pwfs = self.pwfs.forward(wf)

        self.ref_image = self.read_out(wf_pwfs)

        try:
            rmat = np.load(fname+str(".npy"))
            print("loaded cached rmat")
        except:

            #compute the interaction matrix relating incoming abberations to wfs response
            probe_disp = 0.01 * self.wavelength
            slopes = []

            for i in range(self.DM.num_actuators):
                _str = "working on mode "+str(i)+" out of "+str(self.DM.num_actuators)
                print(_str)
                slope = 0

                for s in (-1,1):

                    disps = np.zeros((self.DM.num_actuators,))
                    disps[i] = s*probe_disp
                    self.DM.actuators = disps

                    wf_dm = self.DM.forward(wf)              #pass through DM
                    wf_dm_pwfs = self.pwfs.forward(wf_dm)    #pass through wfs
                    
                    image = self.read_out(wf_dm_pwfs)
                    slope += s * (image - self.ref_image) / (2*probe_disp)
                
                slopes.append(slope)

            basis = hc.ModeBasis(slopes)
            rmat = hc.inverse_tikhonov(basis.transformation_matrix,rcond = rcond, svd = None)
            np.save(fname,rmat)
            
            self.DM.actuators = np.zeros((self.DM.num_actuators,))
        
        self.rmat = rmat
        return rmat

    def make_turb(self,fp0=0.2,wl0=1e-6,outer_scale_length=20,vel=10,seed = None):
        """create a single atmospheric layer according to params (SI units)"""

        if seed is None:
            #fix seed so turbulence results are the same when code is run later
            seed = 123456789

        np.random.seed(seed)
        vel_vec = np.array([vel,0])

        Cn2 = hc.Cn_squared_from_fried_parameter(fp0,wl0)
        layer = hc.InfiniteAtmosphericLayer(self.pupil_grid,Cn2,outer_scale_length,vel_vec)

        self.atmos = layer

        # save the atmos params
        self.fp0 = fp0
        self.seed = seed
        self.wl0 = wl0
        self.OSL = outer_scale_length
        self.vel = vel

        return layer
    
    def run_AO_step(self,wf,leakage,gain,t,dt):
        #norm = self.propagator(wf).power.max()

        self.atmos.t = t #seconds
        

        wf_turb = self.atmos.forward(wf)
        wf_dm = self.DM.forward(wf_turb)            #prop through dm
        wf_pyr = self.pwfs.forward(wf_dm)           #prop through pwfs

        wfs_image = self.read_out(wf_pyr,poisson=False)

        diff_image = wfs_image-self.ref_image

        #leaky integrator correction algorithm
        self.DM.actuators = (1-leakage) * self.DM.actuators - gain * self.rmat.dot(diff_image)

        _wf = self.propagator.forward(wf_dm)    #prop wf to science plane
        return _wf

    def run_closed_loop(self,leakage,gain,fileout,run_for = 10,freq = 1000,save_every = 100, wl=None):
        """run AO system for specified time and save the actuator positions at each timestep"""

        #reset DM
        self.DM.actuators[:] = 0.

        dt = 1/freq
        t_arr = np.linspace(0,run_for, int(run_for*freq) + 1)

        if wl is None:
            wf = hc.Wavefront(self.ap,wavelength = self.wavelength)
        else:
            wf = hc.Wavefront(self.ap,wavelength = wl)

        wf.total_power = self.zero_magnitude_flux * 10**(- self.star_mag/2.5) * dt

        perfect_wf = self.propagator.forward(wf)
        perfect_u = get_u(perfect_wf)

        num_saves = int(len(t_arr)/save_every)

        DMshapes = np.empty((num_saves,len(self.DM.actuators)))
        avg = np.zeros_like(perfect_u,dtype=np.float)
        psf_arr = np.zeros((num_saves,perfect_u.shape[0],perfect_u.shape[1]),dtype=np.complex128)
        times = []

        j = 0
        for i in range(len(t_arr)):

            _wf = self.run_AO_step(wf,leakage,gain,t_arr[i],dt)

            u = get_u(_wf)
            avg += np.real(u*np.conj(u))

            if i != 0 and i%save_every == 0:
                psf_arr[j] = u
                DMshapes[j] = self.DM.actuators
                times.append(t_arr[i])
                j += 1

                printProgressBar(j,num_saves)

        avg = normalize(np.sqrt(avg))
        s = get_strehl(avg,perfect_u)

        with h5py.File(fileout+".hdf5", "w") as f:
            f.create_dataset("DMs",data=DMshapes)
            f.create_dataset("psfs",data=psf_arr)
            f.create_dataset("ts",data=times)

            #add more args here if you want to make them available to used
            _args = f.create_group("tele_args")
            _args.create_dataset("fp0",data=tele.fp0)
            _args.create_dataset("wl0",data=tele.wl0)
            _args.create_dataset("OSL",data=tele.OSL)
            _args.create_dataset("vel",data=tele.vel)
            
            _args.create_dataset("diam",data=tele.diameter)
            _args.create_dataset("fnum",data = tele.fnum)
            _args.create_dataset("num_acts",data = tele.num_acts)
            _args.create_dataset("obstr_frac",data = tele.obstr_frac)
            _args.create_dataset("seed",data=tele.seed,dtype = int)

        return u, s
    
    def get_abb_wf(self,wl=None):
        if wl is None:
            wl = self.wavelength
        wf = hc.Wavefront(self.ap,wavelength = wl)
        wf_turb = self.atmos.forward(wf)
        return wf_turb

    def fix_DM_and_prop(self,acts,t,wl=None):
        self.DM.actuators = acts

        if wl is None:
            wl = self.wavelength

        wf = hc.Wavefront(self.ap,wavelength = wl)
        u_p = get_u(self.propagator.forward(wf))

        self.atmos.t = t #seconds

        wf_turb = self.atmos.forward(wf)
        wf_dm = self.DM.forward(wf_turb)            #prop through dm
        _wf = self.propagator.forward(wf_dm)

        u = get_u(_wf)
        return u, get_strehl(u,u_p)
    
    def get_PSF(self,wf):
        ''' need to set DM and atmos state before calling this '''
        wf_turb = self.atmos.forward(wf)
        wf_dm = self.DM.forward(wf_turb)
        _wf = self.propagator.forward(wf_dm)
        return get_u(_wf)

def setup_tele(fp=0.2,seed=123456789,diam = 4.2,fnum = 9.5,num_acts = 30,ref_wl=1.e-6,_args = None):
    """make a telescope object AND set up turbulence in a reproducible manner, you can also pass in 
       _args, an hdf5 group saved with data files encoding the specs of the telescope object used 
       to create the data."""

    if _args is not None:
        #unpack
        fp0 = _args['fp0'][()]
        wl0 = _args['wl0'][()]
        OSL = _args['OSL'][()]
        vel = _args['vel'][()]

        diam = _args['diam'][()]
        fnum = _args['fnum'][()]
        num_acts = _args['num_acts'][()]
        obstr_frac = _args['obstr_frac'][()]
        seed = _args['seed'][()]

        tele = AOtele(diam,fnum,wl0,num_acts,wl0,obstr_frac)
        tele.calibrate_DM(rcond=0.1)
        tele.make_turb(fp0,wl0,OSL,vel,seed)

        print(fp0,wl0,OSL,vel,diam,fnum,num_acts,obstr_frac,seed)

    else:

        tele = AOtele(diam,fnum,ref_wl,num_acts)
        tele.calibrate_DM(rcond = 0.1)
        tele.make_turb(fp0 = fp,seed=seed)

    return tele

def generate_psfs(tele,freq,run_for,save_every,fileout,wl):

    leak = 0.01
    gain = 0.5
    print("making psfs...")
    u, s = tele.run_closed_loop(leak,gain,fileout,run_for,freq,save_every,wl)

if __name__ == "__main__":
    
    ### configurations ###

    fileout = "refactor_test3"

    ## telescope physical specs ##
    tele_diam = 4.2 # meters
    tele_focal_ratio = 9.5
    num_acts_across_pupil = 30
    ref_wavelength = 1.0e-6 # meters

    ## random number generation seed ##
    seed = 123456789

    ## atmosphere specs ##
    coherence_length = 0.108 # meters

    ## simulation runtime specs ##
    freq = 1000 # Hz
    run_for = 1 # seconds
    save_every = 100 # save every 100th DM shape

    ## leaky int controller params
    leak = 0.01
    gain = 0.5

    ## desired psf wavelength
    wl = 1.0e-6

    ### run the sim ###

    tele = setup_tele(coherence_length,seed,tele_diam,tele_focal_ratio,num_acts_across_pupil,ref_wavelength)
    generate_psfs(tele,freq,run_for,save_every,fileout,wl)
    
    """
    f = h5py.File("refactor_test3.hdf5","r")

    tele2 = setup_tele(_args = f['tele_args'])

    for DM,t in zip(f['DMs'][:],f['ts'][:]):
        psf,s = tele2.fix_DM_and_prop(DM,t,2.0e-6)
        plt.imshow(np.abs(psf))
        plt.show()

    """