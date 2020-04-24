import numpy as np
import hcipy as hc
import matplotlib.pyplot as plt
from matplotlib import animation, rc

class AOtele: 
    
    #imaged star parameters
    zero_magnitude_flux = 3.9e10 #phot/s
    star_mag = 0

    def __init__(self, diameter, fnum, wavelength, num_DM_acts = 30):
        self.diameter = diameter
        self.fnum = fnum
        self.wavelength = wavelength
        
        num_pupil_pixels = 60*2
        pupil_pixel_samps = 56*2
        pupil_grid_diam = diameter * num_pupil_pixels / pupil_pixel_samps
        
        self.pupil_grid = hc.make_pupil_grid(num_pupil_pixels,diameter = pupil_grid_diam)
        ap = hc.make_obstructed_circular_aperture(diameter,np.sqrt(0.074),4,0.06)
        self.ap = hc.evaluate_supersampled(ap,self.pupil_grid,6)

        self.ap2 = hc.evaluate_supersampled(hc.circular_aperture(diameter),self.pupil_grid,6)

        act_spacing = diameter / num_DM_acts
        influence_funcs = hc.make_gaussian_influence_functions(self.pupil_grid,num_DM_acts,act_spacing)
        self.DM = hc.DeformableMirror(influence_funcs)
        
        self.pwfs = hc.PyramidWavefrontSensorOptics(self.pupil_grid,wavelength_0=wavelength)
        
        self.detector = hc.NoiselessDetector()
        self.dt = 1 #integration time

        self.atmos = None

        #self.focal_grid = hc.make_focal_grid(q = 8, num_airy = 20, pupil_diameter=diameter, focal_length=fnum*diameter ,reference_wavelength=wavelength)
        self.focal_grid = hc.make_focal_grid(q = 32, num_airy = 220/(1.55*fnum), f_number=fnum,reference_wavelength=wavelength)
        #self.focal_grid = hc.make_focal_grid(q = 8, num_airy = 20, spatial_resolution=wavelength/diameter)

        self.ref_image = None
        self.rmat = None

        self.propagator = hc.FraunhoferPropagator(self.pupil_grid,self.focal_grid,focal_length=diameter*fnum)

    def read_out(self, wf, poisson = False):
        self.detector.integrate(wf,self.dt)

        if poisson:
            image = hc.large_poisson(self.detector.read_out()).astype(np.float)
        else:
            image = self.detector.read_out()
        image /= image.sum()
        return image

    def calibrate_DM(self, rcond = 0.3, fname_append=None):
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

    def make_turb(self,fp0=0.2,wl0=500e-9,outer_scale_length=20,vel=10):
        """create a single atmospheric layer according to params (SI units)"""

        Cn2 = hc.Cn_squared_from_fried_parameter(fp0,wl0)
        layer = hc.InfiniteAtmosphericLayer(self.pupil_grid,Cn2,outer_scale_length,vel)

        self.atmos = layer
        return layer
    
    def run_AO_step(self,wf,leakage,gain,t,dt):
        #norm = self.propagator(wf).power.max()

        self.atmos.t = t #seconds

        wf_turb = self.atmos.forward(wf)
        wf_dm = self.DM.forward(wf_turb)            #prop through dm
        wf_pyr = self.pwfs.forward(wf_dm)           #prop through pwfs

        wfs_image = self.read_out(wf_pyr,poisson=True)

        diff_image = wfs_image-self.ref_image

        #leaky integrator correction algorithm
        self.DM.actuators = (1-leakage) * self.DM.actuators - gain * self.rmat.dot(diff_image)

        phase = self.ap * self.DM.surface
        phase -= np.mean(phase[self.ap>0])

        _wf = self.propagator.forward(wf_dm)    #prop to science plane

        return _wf

    def run_closed_loop(self,leakage,gain,run_for = 0.2,freq = 1000,save_ani=False,strehl=False):
        reals = []
        imags = []
        dt = 1/freq
        t_arr = np.linspace(0,run_for, int(run_for*freq) + 1)

        wf = hc.Wavefront(self.ap,wavelength = self.wavelength)

        #wf2 = hc.Wavefront(self.ap2,wavelength = self.wavelength)
        #wf2 = self.propagator.forward(wf2)
        #wf3 = self.propagator.forward(wf)

        #hc.imshow_field(wf2.power/wf2.power.max(),grid=self.focal_grid)
        #plt.show()
        
        #hc.imshow_field(np.sqrt(wf3.power/wf3.power.max()),grid=self.focal_grid)
        #plt.show()

        wf.total_power = self.zero_magnitude_flux * 10**(- self.star_mag/2.5) * dt
        norm = self.propagator.forward(wf).power.max()

        for t in t_arr:
            _wf = self.run_AO_step(wf,leakage,gain,t,dt)

        reals = _wf.real
        imags = _wf.imag

        if not strehl:
            return _wf, reals, imags
        
        strehl = _wf.power.max()/norm
        #print(leakage,gain,strehl)
        return _wf, strehl, reals, imags
    def save_ani(self,leakage,gain,fname,run_for = 1,freq = 1000):
        frames = run_for*freq
        frame_arr = np.arange(frames)
        dt = 1/freq
        
        wf = hc.Wavefront(self.ap,wavelength = self.wavelength)
        wf.total_power = self.zero_magnitude_flux * 10**(- self.star_mag/2.5) * dt
        
        PSF = self.run_AO_step(wf,leakage,gain,0,dt)
        PSF = PSF/PSF.max()

        fig, ax = plt.subplots(1,1,figsize=(8,8))
        ax.set_title('Science image plane')

        im = hc.imshow_field(np.log10(PSF), vmin=-5, vmax=0)

        plt.close(fig)

        def animate(frame):
            if frame%10 == 0:
                print("rendering frame %d" % frame)
            t = frame/freq
            output_psf = self.run_AO_step(wf,leakage,gain,t,dt)
            im.set_data(*self.focal_grid.separated_coords, np.log10(output_psf.shaped/output_psf.max()))
            return [im]

        anim = animation.FuncAnimation(fig, animate, frame_arr, interval=160, blit=True)


        Writer = animation.writers['imagemagick']
        writer = Writer(fps=60, metadata=dict(artist='Me'), bitrate=1800)
        anim.save(fname,writer = writer)

wl = 1.55e-6 #m

tele = AOtele(4.2,9.5,wl,30)
tele.calibrate_DM(rcond = 0.01)

tele.make_turb()

leak = 2e-1
gain = 0.5

'''
wf, strehl, reals, imags = tele.run_closed_loop(leak,gain,1,strehl=True)
print(leak,gain,strehl)

hc.imshow_field(wf.power/wf.power.max(),vmax=1,grid=tele.focal_grid)
plt.show()

field = wf.electric_field

reals,imags = np.real(field),np.imag(field)
u = reals + 1.j*imags
u = u.reshape((973,973))

plt.imshow(np.abs(u)/np.abs(u).max(),extent=(-224,224,-224,224),origin = "lower")

plt.show()
'''
wls = np.linspace(1,2.5,31)

for i in range(len(wls)):
    tele.wavelength = wls[0]*1e-6

    wf, strehl, reals, imags = tele.run_closed_loop(leak,gain,1,strehl=True)
    print(leak,gain,strehl)

    field = wf.electric_field

    reals = np.real(field)
    imags = np.imag(field)

    u = reals + 1.j*imags
    u = u.reshape((956,956))
    np.save("psf"+str(i)+".npy",u)
