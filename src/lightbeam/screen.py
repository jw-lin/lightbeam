# Michael Fitzgerald (mpfitz@ucla.edu)
import numpy as np
from numpy.fft import fftfreq, fft2, ifft2,fftshift
import matplotlib.pyplot as plt
from matplotlib import animation

def boiling_freq(k,t1):
    return np.power(k,2/3)/t1

# from Srinath et al. (2015) 28 Dec 2015 | Vol. 23, No. 26 | DOI:10.1364/OE.23.033335 | OPTICS EXPRESS 33335
class PhaseScreenGenerator(object):
    def __init__(self, D, p, vy, vx, T, r0, wl0, wl, rs=None, seed=None, alpha_mag=1.,filter_func = None,filter_scale=None):
        # set up random number generator
        if rs is None:
            rs = np.random.RandomState(seed=seed)
            self.seed = seed

        if filter_scale is None:
            filter_scale = D/2
        
        if filter_func is None:
            filter_func = lambda x,y: np.ones_like(x)

        self.rs = rs

        # set up array dimensions
        a = 2.**np.ceil(np.log2(D/p)) / (D/p)
        self.N = int(a*D/p) # number of pixels on a side of a screen
        S = self.N*p # [m] screen size

        # frequency array
        fy = fftfreq(self.N, d=p)
        fx = fftfreq(self.N, d=p)

        fy = np.dot(fy[:,np.newaxis], np.ones(self.N)[np.newaxis,:])
        fx = np.dot(np.ones(self.N)[:,np.newaxis], fx[np.newaxis,:])

        ff = np.sqrt(fx*fx+fy*fy)
        
        # turbulent power spectrum
        with np.errstate(divide='ignore'):
            self.P =   2.*np.pi/S * self.N * r0**(-5./6.) * (fy*fy + fx*fx)**(-11./12.) * np.sqrt(0.00058) * (wl0/wl)
            self.P[0,0] = 0. # no infinite power

            if filter_func is not None:
                self.P *= np.sqrt(filter_func(filter_scale*fx,filter_scale*fy))
            """
            plt.loglog(fftshift(fx[0]), np.sqrt(fftshift(filter_func(filter_scale*fx[0],0)))*np.max(self.P[0]) ,color='white')
            
            for _fx,_P in zip(fx[0],self.P[0]):
                plt.plot((_fx,_fx),(0,_P),color='0.5',lw=1)
            plt.loglog(fx[0],self.P[0],marker='.',markerfacecolor='steelblue',markeredgecolor='white',markersize=10,ls='None',markeredgewidth=0.5)
            

            
            plt.loglog(fx[0],self.P[0],marker='.',markerfacecolor='indianred',markeredgecolor='white',markersize=10,ls='None',markeredgewidth=0.5)
            plt.xlabel(r"wavenumber ($m^{-1}$) ")
            plt.ylabel("power")
            plt.show()
            """
           

        # set phase scale
        theta = -2.*np.pi*T*(fy*vy+fx*vx)
        self.alpha = alpha_mag*np.exp(1j*theta)#*np.exp(-T*boiling_freq(ff,0.27)) # |alpha|=1 is pure frozen flow 

        # white noise scale factor
        self.beta = np.sqrt(1.-np.abs(self.alpha)**2)*self.P

        self.last_phip = None

        self.t = 0

    def generate(self):
        # generate white noise

        w = self.rs.randn(self.N,self.N)
        wp = fft2(w)

        # get FT of phase
        if self.last_phip is None:
            # first timestep, generate power
            phip = self.P*wp
        else:
            phip = self.alpha*self.last_phip + self.beta*wp
        self.last_phip = phip
    
        # get phase
        phi = ifft2(phip).real

        return phi
    
    def reset(self):
        self.last_phip = None 
        self.rs = np.random.RandomState(self.seed)

    def get_layer(self,t):
        # need to generate from self.t up until t in steps of T
        pass

def make_ani(out,t,dt,D=10,p=0.1,wl0=1,wl=1,vx=4,vy=0,r0=0.25,alpha=1,seed=345698,delay=20):
    psgen = PhaseScreenGenerator(D, p, vy, vx, dt, r0, wl0, wl, seed=seed,alpha_mag=alpha)

    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure()
    ax = plt.axes(xlim=(0, 128), ylim=(0,128))

    im=plt.imshow(psgen.generate())

    # initialization function: plot the background of each frame
    def init():
        im.set_data(np.zeros((128,128)))
        return [im]

    # animation function.  This is called sequentially
    def animate(i):
        im.set_data(psgen.generate())
        return [im]

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                frames=int(t/dt), interval=20, blit=True)

    anim.save(out+'.mp4', fps=60, extra_args=['-vcodec', 'libx264'])

    plt.show()

if __name__=='__main__':

    import zernike as zk
    #make_ani("turb_ani_test_alpha0pt999",5,0.01,vx=0,alpha=0.99)

    r0s = [0.169,0.1765,0.185,0.196,0.215,0.245,0.295]
    SRs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7]

    rms2s=[]
    rms3s=[]

    for sr,r0 in zip(SRs,r0s):

        D = 10. # [m]  telescope diameter
        p = 10/256 # [m/pix] sampling scale

        # set wind parameters
        vy, vx = 0., 10. # [m/s] wind velocity vector
        T = 0.01 # [s]  sampling interval

        # set turbulence parameters
        #r0 = 0.185 # [m]
        wl0 = 1 #[um]
        wl = 1 #[um]

        seed = 123456
        psgen = PhaseScreenGenerator(D, p, vy, vx, T, r0, wl0, wl, seed=seed)

        xa = ya = np.linspace(-1,1,256)
        xg , yg = np.meshgrid(xa,ya)

        dA = (D/256)**2

        z2 = zk.Zj_cart(2)(xg,yg)
        z3 = zk.Zj_cart(3)(xg,yg)

        #plt.imshow(z2)
        #plt.show()

        print(np.sum(z2*z2) * dA / np.pi/25)
        s = psgen.generate()


        PV2s = []
        PV3s = []
        c2s = [] 
        c3s = []
        pe2s = []
        pe3s = []

        for i in range(1000):
            s = psgen.generate()
            #plt.imshow(s)
            #plt.show()

            c2 = np.sum(s*z2)*dA/np.pi/25
            c3 = np.sum(s*z3)*dA/np.pi/25

            c2s.append(c2)
            c3s.append(c3)

            PV2s.append(4*c2/(2*np.pi))
            PV3s.append(4*c3/(2*np.pi))

            pe2 = np.std(c2*z2)/(2*np.pi)
            pe3 = np.std(c3*z3)/(2*np.pi) 

            pe2s.append(pe2)
            pe3s.append(pe3)

        print(np.std(np.array(c2s)))
        plt.plot(np.arange(1000)*0.01,c2s)
        plt.show()

        rms2 = np.sqrt(np.mean(np.power(np.array(PV2s),2)))
        rms3 = np.sqrt(np.mean(np.power(np.array(PV3s),2)))

        plt.plot(np.arange(100),pe2s,color='steelblue',label='x tilt')
        plt.plot(np.arange(100),pe3s,color='indianred',label='y tilt')
        #plt.axhline(y=rms2,color='steelblue',ls='dashed')
        #plt.axhline(y=rms3,color='indianred',ls='dashed')
        plt.xlabel("timestep")
        plt.ylabel("RMS wf error of TT modes")

        plt.legend(frameon=False)
        plt.show()

        rms2s.append(rms2)
        rms3s.append(rms3)

    plt.plot(SRs,rms2s,label="x tilt",color='steelblue')
    plt.plot(SRs,rms3s,label="y tilt",color='indianred')

    plt.xlabel("Strehl ratio")
    plt.ylabel("rms mode amplitude")
    plt.legend(frameon=False)
    plt.show()

    """
    plt.plot(np.arange(len(coeffs2)),coeffs2,color='steelblue',label="x tilt")
    plt.plot(np.arange(len(coeffs3)),coeffs3,color='indianred',label="y tilt")

    rms2 = np.sqrt(np.mean(np.power(np.array(coeffs2),2)))
    rms3 = np.sqrt(np.mean(np.power(np.array(coeffs3),2)))

    plt.axhline(y=rms2,color='steelblue',ls='dashed')
    plt.axhline(y=rms3,color='indianred',ls='dashed')
    plt.xlabel("timestep")
    plt.ylabel("zernike mode amplitude")
    plt.ylim(-8,8)

    plt.legend(frameon=False)

    plt.show()
    """

    """
    last = psgen.generate()
    out = np.zeros_like(last)

    for i in range(8000):
        cur = psgen.generate()
        out += np.power(cur-last,2)
        last = cur

    out/=8000
    plt.imshow(out)
    plt.show()
    print(np.mean(out))
    """



    #fits.writeto('screens_test.fits', screens, overwrite=True)




    ## show results
    #import pylab
    #fig = pylab.figure(0)
    #fig.clear()
    #ax = fig.add_subplot(111)
    #for screen in screens:
    #    ax.cla()
    #    ax.imshow(screen)
    #    pylab.draw()
    #    pylab.show()
