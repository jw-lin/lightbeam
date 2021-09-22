from hcipy.mode_basis.zernike import zernike
import numpy as np
import hcipy as hc
import matplotlib.pyplot as plt
from misc import normalize,resize,printProgressBar,resize2,overlap
from os import path
import time
import lightbeam.LPmodes as LPmodes
import h5py
import screen
from astropy.io import fits
import lightbeam.PIAA as PIAA
import zernike as zk

def get_u(wf:hc.Wavefront):
    '''extract 2D array of complex field data from hcipy wavefront object'''
    field = wf.electric_field * np.sqrt(wf.grid.weights)
    size = int(np.sqrt(field.shape[0]))
    shape=(size,size)
    reals,imags = np.array(np.real(field)),np.array(np.imag(field))
    u = reals + 1.j*imags
    u = u.reshape(shape)
    return u

def get_phase(wf:hc.Wavefront):
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

    #lo_pupil_res = 128 # this resolution is used for the DM correction algorithm
    #hi_pupil_res = 1024 # this resolution is used for propagation

    def __init__(self, diameter, fnum, wavelength, num_DM_acts = 30, wavelength_0 = None, obstr_frac = 0.244, fast_boot = False, hi_pupil_res = 512, lo_pupil_res = 128, remove_TT = False):

        self.remove_TT = remove_TT

        self.hi_pupil_res = hi_pupil_res
        self.lo_pupil_res = lo_pupil_res

        if wavelength_0 is None:
            wavelength_0 = wavelength

        self.reference_wavelength = wavelength_0

        self.diameter = diameter
        self.fnum = fnum
        self.num_acts = num_DM_acts
        self.obstr_frac = obstr_frac
        self.wavelength = wavelength
        
        ## setting up low and high res pupil grids. the low res grid is used for to only calibrate/control the DM

        num_pupil_pixels = self.lo_pupil_res
        pupil_pixel_samps = self.lo_pupil_res #the keck pupil fits seems be padded with zeros around border by ~5%

        self.pupil_plane_res = (num_pupil_pixels,num_pupil_pixels)

        pupil_grid_diam = diameter * num_pupil_pixels / pupil_pixel_samps
        self.pupil_grid_diam = pupil_grid_diam
        self.pupil_sample_rate = pupil_grid_diam / num_pupil_pixels
        
        self.pupil_grid = hc.make_pupil_grid(self.lo_pupil_res,diameter = pupil_grid_diam)
        self.pupil_grid_hires = hc.make_pupil_grid(self.hi_pupil_res,diameter = pupil_grid_diam)

        ## now set up the actual pupil fields

        keck_pupil_hires = np.array(fits.open("pupil_KECK_high_res.fits")[0].data,dtype=np.float32)[50:-50,50:-50]

        ap_arr = resize2(keck_pupil_hires, (self.lo_pupil_res,self.lo_pupil_res) )
        ap_arr_hires = resize2(keck_pupil_hires, (self.hi_pupil_res,self.hi_pupil_res) )
        eps = 1e-6
        ap_arr[np.abs(ap_arr)<eps] = 0
        ap_arr_hires[np.abs(ap_arr_hires)<eps] = 0
        self.ap_data = ap_arr_hires.reshape((hi_pupil_res,hi_pupil_res))
        self.ap = hc.Field(ap_arr.flatten(),self.pupil_grid)
        self.ap_hires = hc.Field(ap_arr_hires.flatten(),self.pupil_grid_hires)

        ## we need to make two DMs, one sampled on the low res pupil grid and another on the hi res pupil grid

        if not fast_boot:
            # this stuff can be skipped if we're not using the telescope to sim ao correction
            act_spacing = diameter / num_DM_acts
            influence_funcs = hc.make_gaussian_influence_functions(self.pupil_grid,num_DM_acts,act_spacing)

            self.DM = hc.DeformableMirror(influence_funcs)
            
            influence_funcs_hires = hc.make_gaussian_influence_functions(self.pupil_grid_hires,num_DM_acts,act_spacing)
            self.DM_hires = hc.DeformableMirror(influence_funcs_hires)

            ## make the rest of our optics (besides PIAA/collimator)

            self.pwfs = hc.PyramidWavefrontSensorOptics(self.pupil_grid,wavelength_0=wavelength_0)
            
            self.detector = hc.NoiselessDetector()

        num_airy = 16
        ## focal grid set up. linear resolution in pixels is 2 * q * num_airy\
        self.q, self.num_airy = int(self.hi_pupil_res/num_airy/2), num_airy
        self.focal_grid = hc.make_focal_grid(q = self.q, num_airy = num_airy, f_number = fnum, reference_wavelength=wavelength_0)
        #self.focal_grid = hc.make_focal_grid(q = 16, num_airy = 16, f_number = fnum, reference_wavelength=wavelength_0)
        
        self.ref_image = None
        self.rmat = None

        ## pupil -> focal and focal -> pupil propagators 

        self.propagator = hc.FraunhoferPropagator(self.pupil_grid,self.focal_grid,focal_length=diameter*fnum)
        self.focal_length = diameter*fnum
        ## misc other stuff that is useful to cache/save
        self.t_arr = None
        self.DMshapes = None
        self.psgen = None
        self.apodize = None
        self.apodize_backwards = None
        self.collimate = None
        self.collimator_grid = None
        self.beam_radius = None

        self.current_phase_screen = None
        self.wf_pupil,self.wf_pupil_hires,self.wf_focal = None,None,None

        self.PIAA_args,self.col_args = None,None

        self.inject_TT =  None

    def get_args(self):
        fp0 = self.fp0
        wl0 = self.reference_wavelength
        OSL = 0
        vel = self.vel
        dt = self.dt
        diam = self.diameter
        fnum = self.fnum
        num_acts = self.num_acts
        obstr_frac = self.obstr_frac
        seed = self.seed
        col_args,PIAA_args = self.col_args,self.PIAA_args

        if col_args is None:
            col_args = (0,0)
        if PIAA_args is None:
            PIAA_args = (0,0,0,0)
        else:
            PIAA_args = PIAA_args[1:]

        return [fp0,wl0,OSL,vel,dt,diam,fnum,num_acts,obstr_frac,seed,*col_args,*PIAA_args]

    def rescale(self,fnum):
        """ change the F# of the optical system"""

        self.fnum = fnum 
        self.focal_grid = hc.make_focal_grid(q = self.q, num_airy = self.num_airy, f_number = fnum, reference_wavelength=self.reference_wavelength)

        if self.collimate is None and self.apodize is None:
            # no collimation or piaa
            self.propagator = hc.FraunhoferPropagator(self.pupil_grid,self.focal_grid,focal_length=self.diameter*fnum)
        elif self.collimate is not None and self.apodize is None:
            # just collimation
            self.propagator = hc.FraunhoferPropagator(self.collimator_grid,self.focal_grid,focal_length=fnum*2*self.beam_radius)
        else:
            # piaa (which needs collimation, from both simulation and practical consideratrions)
            self.propagator = self.propagator = hc.FraunhoferPropagator(self.collimator_grid,self.focal_grid,focal_length=fnum*2*self.beam_radius*0.55) 


    def init_collimator(self,beam_radius):
        """the collimator runs on a higher res than the low res grid used for DM correction to prevent phase aliasing"""
        
        col_res = self.hi_pupil_res 

        self.beam_radius = beam_radius

        print("setting up collimator...")
        self.collimator_grid = hc.make_pupil_grid(col_res, diameter = beam_radius*2)

        self.propagator = hc.FraunhoferPropagator(self.collimator_grid,self.focal_grid,focal_length=self.fnum*2*beam_radius) # what's the focal length after PIAA???
        self.focal_length = self.fnum*2*beam_radius
        def _inner_(wf):
            _power = wf.total_power
            new_wf = hc.Wavefront(hc.Field(wf.electric_field,self.collimator_grid), wf.wavelength)
            # make sure power is conserved
            new_wf.total_power = _power
            return new_wf

        self.collimate = _inner_
        print("collimator setup complete")

    def init_PIAA(self,beam_radius,sep,inner=0,outer=3,IOR=1.48,mode="FR"):
        ''' make a PIAA lens system that maps planar wavefronts to truncated Gaussians.
            beam radius is preserved, with the inner edge at <inner * obstruction fraction> and 
            the outer edge of of the truncated Gaussian is at <outer * sigma>. 
        '''

        #modes: FR, RT,RM

        self.PIAA_args = (beam_radius,sep,inner,outer,IOR)

        #Olivier's suggestion, shrink by a factor of 3 (mult by outer to get it in units of sigma)
        inner = self.obstr_frac * inner * outer 

        print("setting up PIAA lenses...")
        collimator_grid = self.collimator_grid if self.collimator_grid is not None else self.pupil_grid

        #adjusting the focal length so that PIAA and no PIAA psfs are same size on screen (the coeff is empirically determined)
        self.propagator = hc.FraunhoferPropagator(self.collimator_grid,self.focal_grid,focal_length=self.fnum*2*beam_radius*0.55) 
        self.focal_length = self.fnum*2*beam_radius*0.55
        r1,r2 = PIAA.make_remapping_gauss_annulus(self.pupil_plane_res[0],self.obstr_frac,inner,outer)
        
        r1 *= beam_radius
        r2 *= beam_radius
        
        z1,z2 = PIAA.make_PIAA_lenses(r1,r2,IOR,IOR,sep)

        if mode == "FR":
            self.apodize,self.apodize_backwards = PIAA.fresnel_apodizer(collimator_grid,beam_radius,sep,r1,r2,z1,z2,IOR,IOR)

        print("PIAA setup complete")

    def init_PIAA_LP0m(self,m,beam_radius,lens_sep,rcore,ncore,nclad,wl=None,IOR=1.48,inner_trunc=0.0,outer_trunc=-1.0,use_RT=False):
        ''' all units meters. initialize PIAA lenses which remap to a (potentially truncated) LP0m mode. outer_trunc=-1.0 sets PIAA lens to preserve beam size
            (roughly)
        '''
        self.PIAA_args = (beam_radius,lens_sep,inner_trunc,outer_trunc,IOR)
        if outer_trunc == -1.0:
            outer_trunc = None
        print("setting up PIAA lenses...")
        collimator_grid = self.collimator_grid
        #adjusting the focal length so that PIAA and no PIAA psfs are same size on screen (the coeff is empirically determined)
        self.propagator = hc.FraunhoferPropagator(self.collimator_grid,self.focal_grid,focal_length=self.fnum*2*beam_radius*0.55) 
        self.focal_length = self.fnum*2*beam_radius*0.55

        if wl is None:
            wl = self.reference_wavelength
        r1,r2 = PIAA.make_remapping_LP0m(m,1024,self.obstr_frac,beam_radius,rcore,ncore,nclad,wl,self.focal_length,inner_trunc,outer_trunc)

        #plt.plot(np.zeros_like(r1),r1,color='white',marker='.',ls='None')
        #plt.plot(np.ones_like(r2),r2,color='white',marker='.',ls='None')
        #plt.show()

        z1,z2 = PIAA.make_PIAA_lenses(r1,r2,IOR,IOR,lens_sep)
        
        if not use_RT:
            self.apodize,self.apodize_backwards = PIAA.fresnel_apodizer(collimator_grid,beam_radius,lens_sep,r1,r2,z1,z2,IOR,IOR)

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
        wf = hc.Wavefront(self.ap,wavelength=self.reference_wavelength)

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
        self.dt = T

        size = self.pupil_grid_diam
        sampling_scale = size / self.hi_pupil_res

        if self.remove_TT == True:
            filt = zk.high_pass(3)
            self.psgen = screen.PhaseScreenGenerator(size, sampling_scale, v, 0, T, r0, wl0, wl0, seed=seed,filter_func=filt,filter_scale=self.diameter/2)
        else:    
            self.psgen = screen.PhaseScreenGenerator(size, sampling_scale, v, 0, T, r0, wl0, wl0, seed=seed)

    def get_screen(self):
        '''advances timestep and gets the next phase screen'''
        self.current_phase_screen = self.psgen.generate()

    def propagate_through_turb(self,wf,downsample=False,screen=None):
        '''propagate through turbulence, assuming alternate turbulence method (Mike's code) has been set up'''
        wf = wf.copy()
        if  screen is None:
            phase_screen = self.current_phase_screen * self.reference_wavelength / wf.wavelength
        else:
            phase_screen = screen * self.reference_wavelength / wf.wavelength

        if downsample:
            # if not doing full optical propagation, downsample the phase screen
            phase_screen = resize(phase_screen, (self.lo_pupil_res,self.lo_pupil_res) )
        wf.electric_field *= np.exp(1j * phase_screen.flatten())
        return wf
    
    def setup_TT_injector(self,rms,wavelength=None):
        '''rms amplitude is at reference wavelength'''

        if wavelength is None:
            wavelength = self.reference_wavelength

        xa=ya=np.linspace(-1,1,self.hi_pupil_res)
        xg,yg=np.meshgrid(xa,ya)

        tip = zk.Zj_cart(2)(xg,yg) * self.reference_wavelength / wavelength
        tilt = zk.Zj_cart(3)(xg,yg) * self.reference_wavelength / wavelength

        self.rs = np.random.RandomState(seed=123456789)

        def _inner_(wf:hc.Wavefront):
            tip_amp = self.rs.normal(scale=rms)
            tilt_amp = self.rs.normal(scale=rms) 

            wf = wf.copy()
            wf.electric_field *= np.exp(1.j*tip_amp*tip.flatten())
            wf.electric_field *= np.exp(1.j*tilt_amp*tilt.flatten())
            return wf
        
        self.inject_TT = _inner_
        return

    def update_DM(self,wf_lowres,leakage,gain):
        '''takes a low-res pupil plane wavefront and updates DM according to leaky integrator'''
        
        wf_turb = self.propagate_through_turb(wf_lowres,True)
        wf_dm = self.DM.forward(wf_turb)
        wf_pyr = self.pwfs.forward(wf_dm)

        wfs_image = self.read_out(wf_pyr,poisson=False) #DONT ENABLE POISSON it messes w/ random number generation!
        diff_image = wfs_image-self.ref_image

        #leaky integrator correction algorithm
        self.DM.actuators = (1-leakage) * self.DM.actuators - gain * self.rmat.dot(diff_image)
    
    def propagate(self,wf_hires,screen=None,get_TT_stats=False):
        self.DM_hires.actuators = self.DM.actuators

        _wf = self.propagate_through_turb(wf_hires,False,screen)
        _wf = self.DM_hires.forward(_wf)
        
        #optionally, now add extra tip tilt
        if self.inject_TT is not None:
            _wf = self.inject_TT(_wf)

        if self.collimate is not None:
            _wf = self.collimate(_wf)

        if self.apodize is not None:
            _wf = self.apodize(_wf)

        _wf = self.propagator.forward(_wf)

        return _wf
    
    def half_propagate(self,wf_hires):
        self.DM_hires.actuators = self.DM.actuators
        _wf = self.propagate_through_turb(wf_hires,False)
        _wf = self.DM_hires.forward(_wf)
        if self.collimate is not None:
            _wf = self.collimate(_wf)
        return _wf

    def save_args_to_file(self,f):
            
        _args = f.create_group("tele_args")
        _args.create_dataset("fp0",data=self.fp0)
        _args.create_dataset("wl0",data=self.wl0)
        _args.create_dataset("OSL",data=self.OSL)
        _args.create_dataset("vel",data=self.vel)
        _args.create_dataset("dt",data=self.dt)
        _args.create_dataset("diam",data=self.diameter)
        _args.create_dataset("fnum",data = self.fnum)
        _args.create_dataset("num_acts",data = self.num_acts)
        _args.create_dataset("obstr_frac",data = self.obstr_frac)
        _args.create_dataset("seed",data = self.seed,dtype = int)
        _args.create_dataset("res",data = self.hi_pupil_res,dtype = int)
        _args.create_dataset("remove_TT",data = self.remove_TT,dtype = bool)


        if self.collimate is not None:
            _args.create_dataset("beam_radius",data=self.beam_radius)

        if self.apodize is not None:
            _args.create_dataset("PIAA_args",data=self.PIAA_args)
        
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
        
        perfect_p = get_u(wf_pupil_hires)
        perfect_u = get_u(wf_focal)

        num_saves = int(len(t_arr)/save_every)

        DMshapes = np.empty((num_saves,len(self.DM.actuators)))
        avg = np.zeros_like(perfect_u,dtype=np.float)
        psf_arr = np.zeros((num_saves,perfect_u.shape[0],perfect_u.shape[1]),dtype=np.complex128)
        pupil_arr = np.zeros((num_saves,perfect_p.shape[0],perfect_p.shape[1]),dtype=np.complex128)
        screens = []
        times = []

        print("simulating AO system...")

        j = 0
        for i in range(len(t_arr)):
            self.get_screen()
            self.update_DM(wf_pupil,leakage,gain)

            full_prop = (i!= 0 and i%save_every==0)

            if full_prop:
                # do a full optical propagation
                _wf = self.propagate(wf_pupil_hires)

                # only do collimation to test pupil plane coupling
                
                u = get_u(_wf)
                #u2 = get_u(_wf2)

                avg += np.real(u*np.conj(u))
                psf_arr[j] = u
                #pupil_arr[j] = u2
                DMshapes[j] = self.DM.actuators
                screens.append(self.current_phase_screen)
                times.append(t_arr[i])
                j += 1
                printProgressBar(j,num_saves)

        avg/=j
        avg = np.sqrt(avg)

        s = get_strehl(avg,perfect_u)

        with h5py.File(fileout+".hdf5", "w") as f:
            f.create_dataset("DMs",data=DMshapes)
            f.create_dataset("psfs",data=psf_arr)
            f.create_dataset("ts",data=times)
            f.create_dataset("screens",data=np.array(screens))
            #f.create_dataset("pupil_fields",data=pupil_arr)

            self.save_args_to_file(f)

        return u, avg, s

    def make_animation(self,leakage,gain,fileout,run_for = 1,freq = 1000,save_every = 1):
        #reset DM
        self.DM.actuators[:] = 0.
        wf_pupil,wf_pupil_hires,wf_focal = self.wf_pupil,self.wf_pupil_hires,self.wf_focal
        out_shape = get_u(self.wf_focal).shape
        print(out_shape)

        from matplotlib.animation import FuncAnimation

        fig = plt.figure(figsize=(8,8),frameon=False)

        im = plt.imshow(np.zeros(out_shape), cmap='magma', animated=True,vmax=3e-2,vmin=0)

        num_saves = int(run_for*freq / save_every)

        def init():
            im.set_data(np.zeros(out_shape))
            return [im]

        def update_fig(i):
            for j in range(save_every):
                self.get_screen()
                self.update_DM(wf_pupil,leakage,gain)

            _wf = self.propagate(wf_pupil_hires)
            u = get_u(_wf)
            psf = np.real(u*np.conj(u))
            im.set_data(np.sqrt(psf))

            printProgressBar(i,num_saves)    

            return [im]

        anim = FuncAnimation(fig, update_fig,init_func = init,frames=num_saves, blit=True)
        anim.save('test.mp4', fps=60,extra_args=['-vcodec', 'libx264'])
        plt.show()
        
    def get_TT_stats(self,leakage,gain,run_for = 10,freq = 1000,save_every = 100):

        xa=ya=np.linspace(-1,1,self.hi_pupil_res)
        xg,yg=np.meshgrid(xa,ya)
        ds = 2 / self.hi_pupil_res

        ztip = zk.Zj_cart(2)(xg,yg)
        ztilt = zk.Zj_cart(3)(xg,yg)
    
        #reset DM
        self.DM.actuators[:] = 0.
        t_arr = np.linspace(0,run_for, int(run_for*freq) + 1)
        wf_pupil,wf_pupil_hires,wf_focal = self.wf_pupil,self.wf_pupil_hires,self.wf_focal
        num_saves = int(len(t_arr)/save_every)
        perfect_u = get_u(wf_focal)
        tips = []
        tilts = []
        us = []
        avg = np.zeros_like(perfect_u,dtype=np.float)

        print("getting tip-tilt statistics...")
        
        j=0
        for i in range(len(t_arr)):
            self.get_screen()
            self.update_DM(wf_pupil,leakage,gain)

            compute = (i!= 0 and i%save_every==0)

            if compute:
                self.DM_hires.actuators = self.DM.actuators

                ## get TT
                _wf = self.propagate_through_turb(wf_pupil_hires,False)
                _wf = self.DM_hires.forward(_wf)
                if self.inject_TT is not None:
                    _wf = self.inject_TT(_wf)
                phase_cor = _wf.phase.reshape((self.hi_pupil_res,self.hi_pupil_res))
                tip = zk.inner_product(phase_cor,ztip,ds)
                tilt = zk.inner_product(phase_cor,ztilt,ds)

                tips.append(tip)
                tilts.append(tilt)

                ## do an actual prop
                _wf = self.propagate(wf_pupil_hires)
                u = get_u(_wf)
                us.append(u)

                avg += np.real(u*np.conj(u))

                j += 1
                printProgressBar(j,num_saves)

        avg/=j
        avg = np.sqrt(avg)
        s = get_strehl(avg,perfect_u)

        tilts = np.array(tilts)
        tips = np.array(tips)

        print("tip variance: ", np.var(tips))
        print("tilt variance: ", np.var(tilts))
        print("strehl: ",s)

        plt.plot(np.arange(len(tips)),tips)
        plt.plot(np.arange(len(tilts)),tilts)
        plt.show()
        return tips,tilts,s,us

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
    
    def generate_pupil_plane_lpmodes(self,modes,rcore,ncore,nclad,wl=None,f=None,downsample=False,include_PIAA=True):
        res = self.hi_pupil_res
        wl0 = self.wavelength if wl is None else wl
        
        xg = self.focal_grid.coords[0].reshape((res,res))
        yg = self.focal_grid.coords[1].reshape((res,res))
        fields = []

        for mode in modes:
            if mode[0] == 0:
                lf = LPmodes.lpfield(xg,yg,mode[0],mode[1],rcore,wl0,ncore,nclad)
                fields.append( normalize(lf) )
            else:
                lf0 = LPmodes.lpfield(xg,yg,mode[0],mode[1],rcore,wl0,ncore,nclad,"cos")
                lf1 = LPmodes.lpfield(xg,yg,mode[0],mode[1],rcore,wl0,ncore,nclad,"sin")
                
                fields.append(normalize(lf0))
                fields.append(normalize(lf1))
        
        pupil_fields = []

        for field in fields:
            _wf = hc.Wavefront(hc.Field(field.flatten(),self.focal_grid),wl0)
            _wf.total_power = 1 #this should be okay to do because the lpmodes fall off exponentially fast
            wf_ = self.back_propagate(_wf,include_PIAA)

            if downsample:
                pupil_fields.append( 2*resize(get_u(wf_),(int(res/2),int(res/2))))
            else:
                pupil_fields.append(get_u(wf_))
        
        if f is not None:
            f.create_dataset('lp_pupil_fields',data=pupil_fields)

        return pupil_fields
    
def setup_tele_hdf5(fp=0.2,diam = 4.2,fnum = 9.5,num_acts = 30,ref_wl=1.e-6,dt=0.001,seed=123456789,res=512,apo=False,_args = None,fast_boot=False):

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
        res = _args['res'][()]

        beam_radius = None
        PIAA_args = None

        if 'beam_radius' in _args:
            beam_radius = _args['beam_radius'][()]
        if 'PIAA_args' in _args:
            PIAA_args = _args['PIAA_args']

        tele = AOtele(diam,fnum,wl0,num_acts,wl0,obstr_frac,fast_boot=fast_boot,hi_pupil_res=res)
        tele.make_turb(fp0,wl0,dt,vel,seed)

        if beam_radius is not None:
            tele.init_collimator(beam_radius)
        if PIAA_args is not None:
            tele.init_PIAA(*PIAA_args)

    else:
        tele = AOtele(diam,fnum,ref_wl,num_acts,fast_boot=fast_boot,hi_pupil_res=res)
        tele.make_turb(fp,ref_wl,dt,seed=seed)

    return tele

def setup_tele(fp=0.169, diam = 10, fnum = 9.5, num_acts = 30, ref_wl=1.e-6, dt=0.001, seed=123456789, res=512, fast_boot=False, remove_TT=False):

    tele = AOtele(diam,fnum,ref_wl,num_acts,hi_pupil_res=res,remove_TT=remove_TT)
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
    ### configurations ###

    fileout = "SR70_noPIAA_paper" 

    ## telescope physical specs ##
    tele_diam = 10.0 #4.2 # meters
    tele_focal_ratio = 60/16  # =focal grid radius/16, for given setup #3 * 56.29976221549987/48 #this mainly sets the sampling at focal plane, dx = lambda F # I have set this to the optical f # for
    num_acts_across_pupil = 30
    ref_wavelength = 1.0e-6 # meters

    ## random number generation seed ##
    seed = 123456789#

    ## atmosphere specs ##
    coherence_length = 0.169#0.166#0.295 #0.245 for 60#0.215 for 50 #0.196 for 40 #0.185 for 30 #0.1765 for 20 #0.169 for 10% # meters
    remove_TT = False
    rms_TT = 0

    ## simulation runtime specs ##
    freq = 1000 # Hz
    run_for = 1 # seconds
    save_every = 100 # save every 100th DM shape

    ## leaky int controller params
    leak = 0.01
    gain = 0.5

    ## desired psf wavelength
    wl = 1.0e-6

    ## PIAA params
    beam_radius = 0.013/2 # meters
    lens_sep = 2*0.13 # meters
    cutoff = 3
    inner = 0
    apo_mode = "FR"
    res = 512

    ## fiber params
    rcore = 6.21e-6 #m
    ncore = 1.4504
    nclad = 1.4504 - 5.5e-3
    m = 2

    #r0s = [0.169,0.1765,0.185,0.196,0.215,0.245,0.295]
    #SRs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7]

    #plt.plot(SRs,r0s)
    #np.save("SRvsr0",np.array([SRs,r0s]))
    #plt.show()

    ### run the sim ###

    tele = setup_tele(coherence_length,tele_diam,tele_focal_ratio,num_acts_across_pupil,ref_wavelength,1/freq,seed=seed,res=res,remove_TT=remove_TT)
    

    #tele.setup_TT_injector(rms_TT)

    #tele.calibrate_DM()
    #tele.init_collimator(beam_radius)
    #tele.init_PIAA(beam_radius,lens_sep,inner=inner,outer=cutoff,mode=apo_mode)
    #tele.init_PIAA_LP0m(m,beam_radius,lens_sep,rcore,ncore,nclad,use_RT=True,inner_trunc=0)

    #optics now set up, we can compute the perfect wfs
    #tele.get_perfect_wfs()

    #print('hi')
    #tele.make_animation(leak,gain,'test')
    #rint('bye')


    rms_s = [0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2]

    """
    for rms in rms_s:

        name = fileout+str(rms) 

        tele.setup_TT_injector(rms)
        u,avg,s = tele.run_closed_loop(leak,gain,name,run_for,freq,save_every,wl=wl)
        print("average SR: ", s)
    """



    u,avg,s = tele.run_closed_loop(leak,gain,fileout,run_for,freq,save_every,wl=wl)
    print("average SR: ", s)
    #plt.imshow(np.abs(u))
    #plt.show()


    #tips,tilts,strehl,us = tele.get_TT_stats(leak,gain)


    #planar wf masked by aperture
    wf = tele.wf_pupil_hires

    #hc.imshow_field(wf.electric_field)
    #plt.show()

    #turbulence induced abberation
    wf = tele.propagate_through_turb(wf)

    #hc.imshow_field(wf.electric_field)
    #plt.show()

    #partial AO correction
    wf = tele.DM_hires.forward(wf)
    #hc.imshow_field(wf.electric_field)
    #plt.show()

    #beam shrinking
    wf = tele.collimate(wf)
    #hc.imshow_field(wf.electric_field)
    #plt.show()

    print(wf.total_power)
    if tele.apodize is not None:
        #apodization
        wf = tele.apodize(wf)
        #hc.imshow_field(wf.electric_field,vmax=np.max(np.abs(wf.electric_field))/2.5)
        #plt.show()
    print(wf.total_power)
    phase = wf.phase.reshape((res,res))
    #plt.imshow(phase)
    #plt.show()

    u = get_u(wf)
    #plt.imshow(np.sqrt(np.abs(u*np.conj(u))))
    #plt.show()

    #propagation to focal plane
    wf = tele.propagator.forward(wf)

    hc.imshow_field(wf.electric_field)
    plt.show()

    u = get_u(wf)
    plt.imshow(np.abs(u),cmap='inferno')
    plt.colorbar()
    plt.show()
