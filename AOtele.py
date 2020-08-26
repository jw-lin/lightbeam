import numpy as np
import hcipy as hc
import matplotlib.pyplot as plt
from misc import normalize,resize,printProgressBar,resize2
from os import path
import time
import LPmodes
import h5py
import screen
from astropy.io import fits
from Pupil_apodization import Apodizer2

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

        self.reference_wavelength = wavelength_0

        self.diameter = diameter
        self.fnum = fnum
        self.num_acts = num_DM_acts
        self.obstr_frac = obstr_frac
        self.wavelength = wavelength
        
        num_pupil_pixels = 128#60*2
        pupil_pixel_samps = 120#56*2

        self.pupil_plane_res = pres = (num_pupil_pixels,num_pupil_pixels)

        pupil_grid_diam = diameter * num_pupil_pixels / pupil_pixel_samps
        self.pupil_grid_diam = pupil_grid_diam
        self.pupil_sample_rate = pupil_grid_diam / num_pupil_pixels
        
        self.pupil_grid = hc.make_pupil_grid(num_pupil_pixels,diameter = pupil_grid_diam)
        #ap = hc.make_obstructed_circular_aperture(diameter,obstr_frac)
        #self.ap = hc.evaluate_supersampled(ap,self.pupil_grid,6)

        keck_pupil_hires = np.array(fits.open("pupil_KECK_high_res.fits")[0].data,dtype=np.float32)
        ap_arr = resize2(keck_pupil_hires,pres).flatten()
        self.ap = hc.Field(ap_arr.flatten(),self.pupil_grid)

        self.apo_map = fits.open("Pre_to_post_apodized_Keck_pupil_radius_0.fits")[0].data * (num_pupil_pixels/2-1)/122 
        #-1 since Nem's code 
        #occasionally throws index errors
        #when I don't include it

        apo_int = fits.open("2D_apodized_intensity_profile_Keck_0.fits")[0].data
        self.apo_int = resize2(apo_int[-1024-122:1024+122,-1024-122:1024+122],pres)

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

        self.psgen = None

    def gen_wf(self,wl=None):
        if wl is None:
            wl = self.wavelength
        return hc.Wavefront(self.ap,wl)

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
        wf = hc.Wavefront(self.ap,wavelength = self.reference_wavelength)

        wf.total_power = 1

        wf_pwfs = self.pwfs.forward(wf)

        self.ref_image = self.read_out(wf_pwfs)

        try:
            print("trying to load reconstruction matrix from "+fname)
            rmat = np.load(fname+str(".npy"))
            print("loaded cached rmat")
        except:
            print("no rmat file found, computing reconstruction matrix")
            #compute the interaction matrix relating incoming abberations to wfs response
            probe_disp = 0.01 * self.wavelength
            slopes = []

            for i in range(self.DM.num_actuators):
                printProgressBar(i,self.DM.num_actuators)

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
    
    def make_turb_alt(self,r0,wl0,T,v=10,seed = None):
        "set up phase screen generator according Mike's code. flow direction is fixed straight up because i am lazy."

        if seed is None:
            seed = 123456789
        # save the atmos params
        self.fp0 = r0
        self.seed = seed
        self.wl0 = wl0
        self.OSL = "N/A"
        self.vel = v

        size = self.pupil_grid_diam
        sampling_scale = self.pupil_sample_rate

        self.psgen = screen.PhaseScreenGenerator(size, sampling_scale, v, 0, T, r0, wl0, wl0, seed=seed)

    def propagate_through_turb_alt(self,wf):
        '''propagate through turbulence, assuming alternate turbulence method (Mike's code) has been set up'''
        wf = wf.copy()
        phase_screen = self.psgen.generate() * self.reference_wavelength / wf.wavelength
        wf.electric_field *= np.exp(1j * phase_screen.flatten())

        return wf
    
    def apodize(self,wf):
        wf = wf.copy()
        reals = wf.real.reshape(self.pupil_plane_res)
        imags = wf.imag.reshape(self.pupil_plane_res)

        reals_apo = Apodizer2(reals,self.apo_map,self.apo_int)
        imags_apo = Apodizer2(imags,self.apo_map,self.apo_int)

        wf.electric_field = hc.Field( (reals_apo + 1.j * imags_apo).flatten(), self.pupil_grid)
        return wf

    def gen_psf_from_turb(self,r0,wl0,wl,T=0.01,v=10,sr=0.5):
        """this function cheats by just adding a perfect psf and a seeing-limited one together lol"""

        size = self.pupil_grid_diam
        sampling_scale = self.pupil_sample_rate

        psgen = screen.PhaseScreenGenerator(size, sampling_scale, v, 0, T, r0, wl0, wl)
        phase_screen  = psgen.generate()

        wf0 = hc.Wavefront(self.ap,wavelength = wl)

        perfect = get_u(self.propagator.forward(wf0))

        wf0.electric_field *= np.exp(1j * phase_screen.flatten())

        seeing_limited = get_u(self.propagator.forward(wf0))

        plt.imshow(np.abs(perfect))
        plt.show()
        plt.imshow(np.abs(seeing_limited))
        plt.show()

        out = np.sqrt(sr) * perfect + np.sqrt(1-sr) * seeing_limited

        plt.imshow(np.abs(out))
        plt.show()

        return out

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

    def run_AO_step_alt(self,wf,leakage,gain,apo=False):
        """this is like run_AO_step_alt except with custom phase screens from Mike's code. note that the time step is set when self.psgen is initialized"""

        wf_turb = self.propagate_through_turb_alt(wf)

        wf_dm = self.DM.forward(wf_turb)            #prop through dm
        wf_pyr = self.pwfs.forward(wf_dm)           #prop through pwfs

        wfs_image = self.read_out(wf_pyr,poisson=False) #DONT ENABLE POISSON it messes w/ random number generation!

        diff_image = wfs_image-self.ref_image

        #leaky integrator correction algorithm
        self.DM.actuators = (1-leakage) * self.DM.actuators - gain * self.rmat.dot(diff_image)

        if apo:
            wf_dm = self.apodize(wf_dm)

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
    
    def run_closed_loop_alt(self,leakage,gain,fileout,run_for = 10,freq = 1000,save_every = 100, wl=None,apo=False):
        """like the original function but use's Mike's phase screen code"""

        #reset DM
        self.DM.actuators[:] = 0.

        dt = 1/freq
        t_arr = np.linspace(0,run_for, int(run_for*freq) + 1)

        if wl is None:
            wf = hc.Wavefront(self.ap,wavelength = self.wavelength)
        else:
            wf = hc.Wavefront(self.ap,wavelength = wl)

        if apo:
            wf_apo = self.apodize(wf)
            wf_apo.total_power = 1. #self.zero_magnitude_flux * 10**(- self.star_mag/2.5) * dt
            wf.total_power = 1. #self.zero_magnitude_flux * 10**(- self.star_mag/2.5) * dt
            perfect_wf = self.propagator.forward(wf_apo)
            
        else:
            wf.total_power = 1. #self.zero_magnitude_flux * 10**(- self.star_mag/2.5) * dt
            perfect_wf = self.propagator.forward(wf)
        
        perfect_u = get_u(perfect_wf)

        num_saves = int(len(t_arr)/save_every)

        DMshapes = np.empty((num_saves,len(self.DM.actuators)))
        avg = np.zeros_like(perfect_u,dtype=np.float)
        psf_arr = np.zeros((num_saves,perfect_u.shape[0],perfect_u.shape[1]),dtype=np.complex128)
        times = []

        print("simulating AO system...")

        j = 0
        for i in range(len(t_arr)):

            _wf = self.run_AO_step_alt(wf,leakage,gain,apo)

            u = get_u(_wf)
            

            if i != 0 and i%save_every == 0:
                avg += np.real(u*np.conj(u))

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
            _args.create_dataset("dt",data=dt)
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

    def fix_DM_and_prop(self,acts,wl=None):
        self.DM.actuators = acts

        if wl is None:
            wl = self.wavelength

        wf = hc.Wavefront(self.ap,wavelength = wl)
        u_p = get_u(self.propagator.forward(wf))

        wf_turb = self.propagate_through_turb_alt(wf)
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

    else:

        tele = AOtele(diam,fnum,ref_wl,num_acts)
        tele.calibrate_DM(rcond = 0.1)
        tele.make_turb(fp0 = fp,seed=seed)

    return tele

def setup_tele_alt(fp=0.2,diam = 4.2,fnum = 9.5,num_acts = 30,ref_wl=1.e-6,dt=0.001,seed=123456789,apo=False,_args = None):

    if _args is not None:
        #unpack
        fp0 = _args['fp0'][()]
        wl0 = _args['wl0'][()]
        OSL = _args['OSL'][()]
        vel = _args['vel'][()]
        dt = _args['dt'][()]
        diam = _args['diam'][()]
        fnum = _args['fnum'][()]
        num_acts = _args['num_acts'][()]
        obstr_frac = _args['obstr_frac'][()]
        seed = _args['seed'][()]

        tele = AOtele(diam,fnum,wl0,num_acts,wl0,obstr_frac)
        tele.calibrate_DM(rcond=0.1)
        tele.make_turb_alt(fp0,wl0,dt,vel,seed)
    else:

        tele = AOtele(diam,fnum,ref_wl,num_acts)
        tele.calibrate_DM(rcond = 0.1)
        tele.make_turb_alt(fp,ref_wl,dt)

    return tele

def generate_psfs(tele,freq,run_for,save_every,fileout,wl):

    leak = 0.01
    gain = 0.5
    print("making psfs...")
    u, s = tele.run_closed_loop(leak,gain,fileout,run_for,freq,save_every,wl)
    return u,s

if __name__ == "__main__":
    

    ### configurations ###

    fileout = "apo_test"

    ## telescope physical specs ##
    tele_diam = 10.0 #4.2 # meters
    tele_focal_ratio = 15.0 #9.5
    num_acts_across_pupil = 30
    ref_wavelength = 1.0e-6 # meters

    ## random number generation seed ##
    seed = 34591485

    ## atmosphere specs ##
    coherence_length = 0.125 # meters

    ## simulation runtime specs ##
    freq = 1000 # Hz
    run_for = 10 # seconds
    save_every = 100 # save every 10th DM shape

    ## leaky int controller params
    leak = 0.01
    gain = 0.5

    ## desired psf wavelength
    wl = 2.2e-6

    ### run the sim ###

    """
    tele = setup_tele_alt(coherence_length,tele_diam,tele_focal_ratio,num_acts_across_pupil,ref_wavelength,1/freq,seed = seed)
    
    u,s = tele.run_closed_loop_alt(leak,gain,"Kband",run_for,freq,save_every,apo=False,wl=wl)

    print(s)
    plt.imshow(np.abs(u))
    plt.show()
    """

    """
    #planar wf masked by aperture
    wf = tele.gen_wf()

    hc.imshow_field(wf.electric_field)
    plt.show()

    #turbulence induced abberation
    wf = tele.propagate_through_turb_alt(wf)

    hc.imshow_field(wf.electric_field)
    plt.show()

    #partial AO correction
    wf = tele.DM.forward(wf)

    hc.imshow_field(wf.electric_field)
    plt.show()

    #apodization
    wf = tele.apodize(wf)

    hc.imshow_field(wf.electric_field)
    plt.show()

    #propagation to focal plane
    wf = tele.propagator.forward(wf)

    hc.imshow_field(wf.electric_field)
    plt.show()

    
    #generate_psfs(tele,freq,run_for,save_every,fileout,wl)
    """


    f = h5py.File("Kband.hdf5","r")

    tele = setup_tele_alt(_args = f['tele_args'])

    wf = hc.Wavefront(tele.ap,1.e-6)
    up = get_u(tele.propagator.forward(wf))
    plt.imshow(np.abs(up))
    plt.show()

    tele.psgen.generate()
    tele.psgen.generate()

    psfs_k2y = []
    avg = np.zeros((480,480))

    for DM in f['DMs']:
        for i in range(save_every-1):
            tele.psgen.generate()
        u,s = tele.fix_DM_and_prop(DM,wl=1.e-6)
        avg += np.real(u * np.conj(u))
        psfs_k2y.append(u)
        
    avg = normalize(np.sqrt(avg))

    print(get_strehl(avg,up))

    psfs_k2y = np.array(psfs_k2y)
    #np.save("psfs_k2y",psfs_k2y)
