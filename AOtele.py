import numpy as np
import hcipy as hc
import matplotlib.pyplot as plt
from misc import normalize,resize,printProgressBar,resize2,overlap
from os import path
import time
import LPmodes
import h5py
import screen
from astropy.io import fits
import PIAA

def get_u(wf,norm=1):
    field = wf.electric_field/norm
    size = int(np.sqrt(field.shape[0]))
    shape=(size,size)
    reals,imags = np.array(np.real(field)),np.array(np.imag(field))
    u = reals + 1.j*imags
    u = u.reshape(shape)
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

    low_pupil_res = 128 # this resolution is used for the DM correction algorithm
    hi_pupil_res = 512 # this resolution is used for propagation

    def __init__(self, diameter, fnum, wavelength, num_DM_acts = 30, wavelength_0 = None, obstr_frac = 0.242):

        if wavelength_0 is None:
            wavelength_0 = wavelength

        self.reference_wavelength = wavelength_0

        self.diameter = diameter
        self.fnum = fnum
        self.num_acts = num_DM_acts
        self.obstr_frac = obstr_frac
        self.wavelength = wavelength
        
        ## setting up low and high res pupil grids. the low res grid is used for to only calibrate/control the DM

        num_pupil_pixels = self.low_pupil_res
        pupil_pixel_samps = self.low_pupil_res*0.95 #the keck pupil fits seems be padded with zeros around border by ~5%

        self.pupil_plane_res = (num_pupil_pixels,num_pupil_pixels)

        pupil_grid_diam = diameter * num_pupil_pixels / pupil_pixel_samps
        self.pupil_grid_diam = pupil_grid_diam
        self.pupil_sample_rate = pupil_grid_diam / num_pupil_pixels
        
        self.pupil_grid = hc.make_pupil_grid(self.low_pupil_res,diameter = pupil_grid_diam)
        self.pupil_grid_hires = hc.make_pupil_grid(self.hi_pupil_res,diameter = pupil_grid_diam)

        ## now set up the actual pupil fields

        keck_pupil_hires = np.array(fits.open("pupil_KECK_high_res.fits")[0].data,dtype=np.float32)
        ap_arr = resize2(keck_pupil_hires, (self.low_pupil_res,self.low_pupil_res) )
        ap_arr_hires = resize2(keck_pupil_hires, (self.hi_pupil_res,self.hi_pupil_res) )
        
        self.ap = hc.Field(ap_arr.flatten(),self.pupil_grid)
        self.ap_hires = hc.Field(ap_arr_hires.flatten(),self.pupil_grid_hires)

        ## we need to make two DMs, one sampled on the low res pupil grid and another on the hi res pupil grid

        act_spacing = diameter / num_DM_acts
        influence_funcs = hc.make_gaussian_influence_functions(self.pupil_grid,num_DM_acts,act_spacing)
        
        self.DM = hc.DeformableMirror(influence_funcs)
        
        influence_funcs_hires = hc.make_gaussian_influence_functions(self.pupil_grid_hires,num_DM_acts,act_spacing)
        self.DM_hires = hc.DeformableMirror(influence_funcs_hires)

        ## make the rest of our optics (besides PIAA/collimator)

        self.pwfs = hc.PyramidWavefrontSensorOptics(self.pupil_grid,wavelength_0=wavelength_0)
        
        self.detector = hc.NoiselessDetector()
        self.dt = 1 #integration time in seconds (?)

        ## focal grid set up. linear resolution in pixels is 2 * q * num_airy
        self.focal_grid = hc.make_focal_grid(q = 16*2, num_airy = 16, f_number = fnum, reference_wavelength=wavelength_0)

        self.ref_image = None
        self.rmat = None

        ## pupil -> focal and focal -> pupil propagators 

        self.propagator = hc.FraunhoferPropagator(self.pupil_grid,self.focal_grid,focal_length=diameter*fnum)
        self.propagator_backward = hc.FraunhoferPropagator(self.focal_grid,self.pupil_grid,focal_length=-diameter*fnum)
        
        ## misc other stuff that is useful to cache/save
        self.t_arr = None
        self.DMshapes = None
        self.psgen = None
        self.apodize = None
        self.apodize_backwards = None
        self.collimate = None
        self.collimator_grid = None

        self.current_phase_screen = None
        self.wf_pupil,self.wf_pupil_hires,self.wf_focal = None,None,None

    def init_collimator(self,beam_radius,col_res=512):
        """the collimator can run on a higher res than regular pupil plane so that aliasing is prevented when also using PIAA"""

        print("setting up collimator...")
        self.collimator_grid = hc.make_pupil_grid(col_res, diameter = beam_radius*2)
        self.propagator = hc.FraunhoferPropagator(self.collimator_grid,self.focal_grid,focal_length=self.fnum*2*beam_radius) # what's the focal length after PIAA???

        def _inner_(wf):
            reals,imags = wf.real,wf.imag
            
            reals = np.reshape(reals, (self.hi_pupil_res,self.hi_pupil_res) )
            imags = np.reshape(imags, (self.hi_pupil_res,self.hi_pupil_res) )

            if self.hi_pupil_res != col_res:
                reals = resize2(reals,(col_res,col_res)).flatten()
                imags = resize2(imags,(col_res,col_res)).flatten()
            else:
                reals = reals.flatten()
                imags = imags.flatten()

            new_wf = hc.Wavefront(hc.Field(reals+1.j*imags,self.collimator_grid), wf.wavelength)
            new_wf.total_power = 1
            return new_wf

        self.collimate = _inner_
        print("collimator setup complete")

    def init_PIAA(self,beam_radius,sep,inner=0,outer=3,IOR=1.48,pad=1):
        print("setting up PIAA lenses...")
        collimator_grid = self.collimator_grid if self.collimator_grid is not None else self.pupil_grid

        #adjusting the focal length so that PIAA and no PIAA psfs are same size on screen (the coeff is empirically determined)
        self.propagator = hc.FraunhoferPropagator(self.collimator_grid,self.focal_grid,focal_length=self.fnum*2*beam_radius*0.55) 
        
        r1,r2 = PIAA.make_remapping_gauss_annulus(self.pupil_plane_res[0],self.obstr_frac,inner,outer)
        z1,z2 = PIAA.make_PIAA_lenses(r1*beam_radius,r2*beam_radius,IOR,IOR,sep)

        self.apodize,self.apodize_backwards = PIAA.fresnel_apodizer(collimator_grid,beam_radius,sep,pad,r1,r2,z1,z2,IOR,IOR)
        print("PIAA setup complete")

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

    def calibrate_DM(self, rcond = 0.1, fname_append=None, force_new = False):
        """Calculate and save the imagae reconstruction matrix for the AO system"""
        from os import path

        if fname_append is not None:
            fname = "rmat_"+str(self.DM.num_actuators)+"_"+str(rcond).replace(".","")+"_"+fname_append
        else:
            fname = "rmat_"+str(self.DM.num_actuators)+"_"+str(rcond).replace(".","")

        #first we make a reference image for an unaberrated wavefront passing through the pwfs
        wf = hc.Wavefront(self.ap,wavelength = self.reference_wavelength)

        wf.total_power = 1

        wf_pwfs = self.pwfs.forward(wf)

        self.ref_image = self.read_out(wf_pwfs,poisson=False)

        if path.exists(fname+".npy") and not force_new:
            print("trying to load reconstruction matrix from "+fname)
            rmat = np.load(fname+str(".npy"))
            print("loaded cached rmat")
        else:
            print("computing reconstruction matrix")
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

            print("matrix inversion...")
            basis = hc.ModeBasis(slopes)
            rmat = hc.inverse_tikhonov(basis.transformation_matrix,rcond = rcond, svd = None)
            np.save(fname,rmat)
            
            self.DM.actuators = np.zeros((self.DM.num_actuators,))
        
        self.rmat = rmat
        return rmat
    
    def make_turb(self,r0,wl0,T,v=10,seed = None):
        """ set up phase screen generator according Mike's code. flow direction is fixed straight up because i am lazy. will
            will be replaced with my phase screen generator when testing is complete 
        """

        if seed is None:
            seed = 123456789
        # save the atmos params
        self.fp0 = r0
        self.seed = seed
        self.wl0 = wl0
        self.OSL = "N/A"
        self.vel = v
        self.sampling_freq = T

        size = self.pupil_grid_diam
        sampling_scale = size / self.hi_pupil_res
        self.psgen = screen.PhaseScreenGenerator(size, sampling_scale, v, 0, T, r0, wl0, wl0, seed=seed)

    def get_screen(self):
        '''advances timestep and gets the next phase screen'''
        self.current_phase_screen = self.psgen.generate()

    def propagate_through_turb(self,wf,downsample=False):
        '''propagate through turbulence, assuming alternate turbulence method (Mike's code) has been set up'''
        wf = wf.copy()
        phase_screen = self.current_phase_screen * self.reference_wavelength / wf.wavelength

        if downsample:
            # if not doing full optical propagation, downsample the phase screen
            phase_screen = resize(phase_screen, (self.low_pupil_res,self.low_pupil_res) )
        wf.electric_field *= np.exp(1j * phase_screen.flatten())
        return wf
    
    def update_DM(self,wf_lowres,leakage,gain):
        '''takes a low-res pupil plane wavefront and updates DM according to leaky integrator'''
        
        wf_turb = self.propagate_through_turb(wf_lowres,True)
        wf_dm = self.DM.forward(wf_turb)
        wf_pyr = self.pwfs.forward(wf_dm)

        wfs_image = self.read_out(wf_pyr,poisson=False) #DONT ENABLE POISSON it messes w/ random number generation!
        diff_image = wfs_image-self.ref_image

        #leaky integrator correction algorithm
        self.DM.actuators = (1-leakage) * self.DM.actuators - gain * self.rmat.dot(diff_image)
    
    def propagate(self,wf_hires):
        self.DM_hires.actuators = self.DM.actuators

        _wf = self.propagate_through_turb(wf_hires,False)
        _wf = self.DM_hires.forward(_wf)

        if self.collimate is not None:
            _wf = self.collimate(_wf)

        if self.apodize is not None:
            _wf = self.apodize(_wf)

        _wf = self.propagator.forward(_wf)

        return _wf
    
    def save_args_to_file(self,f):
            
        _args = f.create_group("tele_args")
        _args.create_dataset("fp0",data=tele.fp0)
        _args.create_dataset("wl0",data=tele.wl0)
        _args.create_dataset("OSL",data=tele.OSL)
        _args.create_dataset("vel",data=tele.vel)
        _args.create_dataset("dt",data=tele.dt)
        _args.create_dataset("diam",data=tele.diameter)
        _args.create_dataset("fnum",data = tele.fnum)
        _args.create_dataset("num_acts",data = tele.num_acts)
        _args.create_dataset("obstr_frac",data = tele.obstr_frac)
        _args.create_dataset("seed",data=tele.seed,dtype = int)
        _args.create_dataset("sample_freq",data=tele.sampling_freq)

    def get_perfect_wfs(self,wl=None):
        """compute and save perfect pupil plane and focal plane wavefronts"""

        if wl is None:
            wf_pupil = hc.Wavefront(self.ap,wavelength = self.wavelength)
            wf_pupil_hires = hc.Wavefront(self.ap_hires,wavelength = self.wavelength)
        else:
            wf_pupil = hc.Wavefront(self.ap,wavelength = wl)
            wf_pupil_hires = hc.Wavefront(self.ap_hires,wavelength = wl)

        wf_pupil.total_power = 1
        wf_pupil_hires.total_power = 1

        wf_focal = wf_pupil_hires.copy()

        if self.collimate is not None:
            wf_focal = self.collimate(wf_focal)
        
        if self.apodize is not None:
            wf_focal = self.apodize(wf_focal)
        
        wf_focal = self.propagator.forward(wf_focal) 
        # note that the total power in the focal plane is not 1. some power gets deflected 
        # away due to turbulence.

        self.wf_pupil,self.wf_pupil_hires,self.wf_focal = wf_pupil,wf_pupil_hires,wf_focal
        return wf_pupil,wf_pupil_hires,wf_focal

    def run_closed_loop(self,leakage,gain,fileout,run_for = 10,freq = 1000,save_every = 100, wl=None):
        """like the original function but use's Mike's phase screen code"""

        #reset DM
        self.DM.actuators[:] = 0.

        t_arr = np.linspace(0,run_for, int(run_for*freq) + 1)

        wf_pupil,wf_pupil_hires,wf_focal = self.wf_pupil,self.wf_pupil_hires,self.wf_focal
        
        perfect_u = get_u(wf_focal)

        print("perfect psf of current optical system: ")
        plt.imshow(np.abs(perfect_u))
        plt.show()

        num_saves = int(len(t_arr)/save_every)

        DMshapes = np.empty((num_saves,len(self.DM.actuators)))
        avg = np.zeros_like(perfect_u,dtype=np.float)
        psf_arr = np.zeros((num_saves,perfect_u.shape[0],perfect_u.shape[1]),dtype=np.complex128)
        times = []

        norm = np.sqrt(np.sum(perfect_u*np.conj(perfect_u)))

        print("simulating AO system...")

        j = 0
        for i in range(len(t_arr)):

            self.get_screen()
            self.update_DM(wf_pupil,leakage,gain)

            full_prop = (i!= 0 and i%save_every==0)

            if full_prop:
                # do a full optical propagation
                _wf = self.propagate(wf_pupil_hires)
                u = get_u(_wf,norm)
                avg += np.real(u*np.conj(u))
                psf_arr[j] = u
                DMshapes[j] = self.DM.actuators
                times.append(t_arr[i])
                j += 1
                printProgressBar(j,num_saves)

        avg/=j
        avg = np.sqrt(avg)

        s = get_strehl(avg,perfect_u/norm)

        with h5py.File(fileout+".hdf5", "w") as f:
            f.create_dataset("DMs",data=DMshapes)
            f.create_dataset("psfs",data=psf_arr)
            f.create_dataset("ts",data=times)

            self.save_args_to_file(f)

        return u, avg, s

    def back_propagate(self,wf,include_PIAA=False):
        '''take a wf at the focal plane and back-propagate it to the pupil plane'''
        #wf = self.propagator.backward(wf)
        wf = self.propagator.backward(wf)
        #PIAA
        if self.apodize_backwards is not None and include_PIAA:
            wf = self.apodize_backwards(wf)

        return wf
    
    def optimize_focal_ratio(self,rcore):
        '''find the optimal focal ratio for coupling into a mmf of certain size'''
        return NotImplementedError

    """
    def generate_pupil_plane_lpmodes(self,modes):
        xg = self.focal_grid

        for mode in modes:
            if mode[0] == 0:
                lf = LPmodes.lpfield(xg,yg,mode[0],mode[1],rcore,wl0,ncore,nclad)
                field = normalize(lf)#[s:-s,s:-s]
                #field = normalize(LPmodes.lpfield(xg,yg,mode[0],mode[1],rcore,wl0,ncore,nclad))
                #field = normalize(LP01gauss(xg,yg,rcore,wl0,ncore,nclad))
                _power = np.power(overlap(u0,field),2)
                return _power
            else:
                #field0 = normalize(LPmodes.lpfield(xg,yg,mode[0],mode[1],rcore,wl0,ncore,nclad,"cos"))
                #field1 = normalize(LPmodes.lpfield(xg,yg,mode[0],mode[1],rcore,wl0,ncore,nclad,"sin"))
                lf0 = LPmodes.lpfield(xg,yg,mode[0],mode[1],rcore,wl0,ncore,nclad,"cos")
                lf1 = LPmodes.lpfield(xg,yg,mode[0],mode[1],rcore,wl0,ncore,nclad,"sin")
                
                field0 = normalize(lf0)#[s:-s,s:-s]
                field1 = normalize(lf1)#[s:-s,s:-s]
    """

def setup_tele(fp=0.2,diam = 4.2,fnum = 9.5,num_acts = 30,ref_wl=1.e-6,dt=0.001,seed=123456789,apo=False,_args = None):

    if _args is not None:
        #unpack
        fp0 = _args['fp0'][()]
        wl0 = _args['wl0'][()]
        OSL = _args['OSL'][()]  #OSL not used atm, since phase screen generator is currently running Kolmogorov, not von Karman
        vel = _args['vel'][()]
        dt = _args['dt'][()]
        diam = _args['diam'][()]
        fnum = _args['fnum'][()]
        num_acts = _args['num_acts'][()]
        obstr_frac = _args['obstr_frac'][()]
        seed = _args['seed'][()]

        tele = AOtele(diam,fnum,wl0,num_acts,wl0,obstr_frac)
        tele.make_turb(fp0,wl0,dt,vel,seed)
    else:
        tele = AOtele(diam,fnum,ref_wl,num_acts)
        tele.make_turb(fp,ref_wl,dt,seed=seed)

    return tele

def generate_psfs(tele,freq,run_for,save_every,fileout,wl):
    leak = 0.01
    gain = 0.5
    print("making psfs...")
    u, s = tele.run_closed_loop(leak,gain,fileout,run_for,freq,save_every,wl)
    return u,s

if __name__ == "__main__":

    plt.style.use("dark_background")
    
    #import keckpupil
    #pupil = keckpupil.generate_keck_pupil(1024,0.011)

    #plt.imshow(pupil)
    #plt.show()

    ### configurations ###

    fileout = "quick_PIAA_test13"

    ## telescope physical specs ##
    tele_diam = 10.0 #4.2 # meters
    tele_focal_ratio = 3 * 56.29976221549987/48 #this mainly sets the sampling at focal plane, dx = lambda F # I have set this to the optical f # for
    num_acts_across_pupil = 30
    ref_wavelength = 1.0e-6 # meters

    ## random number generation seed ##
    seed = 123456789#34591485

    ## atmosphere specs ##
    coherence_length = 0.182#0.169 # meters

    ## simulation runtime specs ##
    freq = 1000 # Hz
    run_for = 0.5 # seconds
    save_every = 100 # save every 100th DM shape

    ## leaky int controller params
    leak = 0.01
    gain = 0.5

    ## desired psf wavelength
    wl = 1.0e-6

    ## PIAA params
    beam_radius = 0.013/2 # meters
    lens_sep = 0.12 # meters

    ### run the sim ###

    tele = setup_tele(coherence_length,tele_diam,tele_focal_ratio,num_acts_across_pupil,ref_wavelength,1/freq,seed = seed)

    tele.calibrate_DM()
    tele.init_collimator(beam_radius)
    tele.init_PIAA(beam_radius,lens_sep)

    #optics now set up, we can compute the perfect wfs
    tele.get_perfect_wfs()

    """
    # LP mode stuff
    ncore = 1.4504
    nclad = 1.4504 - 5.5e-3

    focalgrid = tele.focal_grid

    r = 16*16*2

    xg,yg = focalgrid.coords[0].reshape((r,r)),focalgrid.coords[1].reshape((r,r))

    LP_field = LPmodes.lpfield(xg*1e6,yg*1e6,0,1,6.21,1,ncore,nclad)

    LP = hc.Wavefront(hc.Field(LP_field.flatten(),focalgrid),wavelength=1.e-6)

    LP_field2 = LPmodes.lpfield(xg*1e6,yg*1e6,0,2,6.21,1,ncore,nclad)

    LP2 = hc.Wavefront(hc.Field(LP_field2.flatten(),focalgrid),wavelength=1.e-6)

    pupil_LP = tele.back_propagate(LP,True)
    pupil_LP2 = tele.back_propagate(LP2,True)

    field1 = pupil_LP.electric_field.shaped
    field2 = pupil_LP2.electric_field.shaped

    import compute_coupling as cc

    wf_pupil,wf_focal = tele.get_perfect_wfs()
    wf_pupil = tele.collimate(wf_pupil)

    hc.imshow_field(wf_pupil.electric_field)
    plt.show()

    wf_pupil_recon = tele.back_propagate(wf_focal,True)
    hc.imshow_field(wf_pupil_recon.electric_field)
    plt.show()

    _fieldlp = cc.normalize(LP.electric_field.shaped)
    _fieldpsf = cc.normalize(wf_focal.electric_field.shaped)

    _fieldlpp = cc.normalize(pupil_LP.electric_field.shaped)
    _fieldpup = cc.normalize(wf_pupil.electric_field.shaped)
    _fieldlpp2 = cc.normalize(pupil_LP2.electric_field.shaped)

    print(cc.overlap(_fieldlp,_fieldpsf))
    print(cc.overlap(_fieldlpp,_fieldpup))
    print(cc.overlap(_fieldlpp,_fieldlpp2))

    hc.imshow_field(pupil_LP.electric_field)
    plt.show()
    hc.imshow_field(pupil_LP2.electric_field)
    plt.show()
    """
    

    u,avg,s = tele.run_closed_loop(leak,gain,fileout,run_for,freq,save_every,wl=wl)



    print(s)
    plt.imshow(np.abs(u))
    plt.show()

    #planar wf masked by aperture
    wf = tele.wf_pupil_hires

    hc.imshow_field(wf.electric_field)
    plt.show()

    #turbulence induced abberation
    wf = tele.propagate_through_turb(wf)

    hc.imshow_field(wf.electric_field)
    plt.show()

    #partial AO correction
    wf = tele.DM_hires.forward(wf)

    hc.imshow_field(wf.electric_field)
    plt.show()

    #beam shrinking
    wf = tele.collimate(wf)
    hc.imshow_field(wf.electric_field)
    plt.show()

    #apodization
    wf = tele.apodize(wf)
    hc.imshow_field(wf.electric_field)
    plt.show()

    u = get_u(wf)
    plt.imshow(np.abs(u*np.conj(u)))
    plt.show()

    #propagation to focal plane
    wf = tele.propagator.forward(wf)

    hc.imshow_field(wf.electric_field)
    plt.show()

    u = get_u(wf)
    plt.imshow(np.abs(u*np.conj(u)))
    plt.show()
    
