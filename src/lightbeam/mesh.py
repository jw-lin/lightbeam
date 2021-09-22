import numpy as np
from numpy import s_,arange,sqrt,power,complex64 as c64
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt
from itertools import chain
from bisect import bisect_left
import numexpr as ne
import math

## to do

# figure out how to normalize ucrit
# combine the different remeshing options into a single func with a switch argument
# remeshing process is a little inefficient. not sure how to speed up though

TOL=1e-12

class RectMesh2D:
    ''' transverse adapative mesh class '''

    def __init__(self,xw,yw,dx,dy,Nbc=4):

        self.max_iters = 6
        self.Nbc = Nbc
        self.dx0,self.dy0 = dx,dy

        self.xa = None
        self.ya = None

        self.xg = None
        self.yg = None

        self.ccel_ix = np.s_[Nbc+2:-Nbc-2]

        ###??? idk why these have to overlap but the pml doesn't work any other way
        self.cvert_ix = np.s_[Nbc:-Nbc]
        self.pvert_ix = np.hstack((arange(Nbc+1),arange(-Nbc-1,0)))

        self.reinit(xw,yw)
        self.update(self.rfacxa,self.rfacya)

    def reinit(self,xw,yw):
        self.xa_last,self.ya_last = self.xa,self.ya

        dx,dy = self.dx0,self.dy0
        Nbc = self.Nbc

        self.shape0_comp = (int(round(xw/dx)+1),int(round(yw/dy)+1))
        xres,yres = self.shape0_comp[0] + 2*Nbc , self.shape0_comp[1] + 2*Nbc

        self.shape0 = (xres,yres)
        self.shape = (xres,yres)

        self.xw,self.yw = xw,yw

        #anchor point of the adaptive grid
        self.xm = -xw/2-Nbc*dx
        self.ym = -yw/2-Nbc*dy
        self.xM = xw/2+Nbc*dx
        self.yM = yw/2+Nbc*dy

        self.xa0 = np.linspace(-xw/2-Nbc*dx,xw/2+Nbc*dx,xres)
        self.ya0 = np.linspace(-yw/2-Nbc*dy,yw/2+Nbc*dy,yres)

        self.xix_base = np.arange(xres)
        self.yix_base = np.arange(yres)

        self.dxa = np.full(xres,dx)
        self.dya = np.full(yres,dy)

        self.xa = self.xa0 = np.linspace(-xw/2-Nbc*dx,xw/2+Nbc*dx,xres)
        self.ya = self.ya0 = np.linspace(-yw/2-Nbc*dy,yw/2+Nbc*dy,yres)

        self.pvert_xa = self.xa0[self.pvert_ix]
        self.pvert_ya = self.ya0[self.pvert_ix]

        self.rfacxa = self.rfacxa0 = np.full(xres-1,1)
        self.rfacya = self.rfacya0 = np.full(yres-1,1)

    def snapto(self,xw,yw):
        xwr = 2*math.ceil(xw/2/self.dx0)
        ywr = 2*math.ceil(yw/2/self.dy0)

        #round
        xw = xwr*self.dx0
        yw = ywr*self.dy0
        return xw, yw

    def dxa2xa(self,dxa):
        N = len(dxa)
        out = np.zeros(N+1)
        np.cumsum(dxa,out=out[1:])
        return out + self.xm

    def update(self,rfacxa,rfacya):
        self.rfacxa = rfacxa
        self.rfacya = rfacya

        xix_base = np.empty(len(rfacxa)+1,dtype=int)
        yix_base = np.empty(len(rfacya)+1,dtype=int)

        xix_base[0] = 0
        yix_base[0] = 0

        xix_base[1:] = np.cumsum(rfacxa)
        yix_base[1:] = np.cumsum(rfacya)

        self.xix_base = xix_base[self.xix_base]
        self.yix_base = yix_base[self.yix_base]

        new_dxa = np.repeat(self.dxa[1:]/rfacxa,rfacxa)
        new_dya = np.repeat(self.dya[1:]/rfacya,rfacya)

        new_xa = self.dxa2xa(new_dxa)
        new_ya = self.dxa2xa(new_dya)

        self.xa = new_xa
        self.ya = new_ya

        rxa = np.empty_like(self.xa,dtype=float)
        rxa[1:-1] = new_dxa[1:]/new_dxa[:-1]
        rxa[0] = 1
        rxa[-1] = 1
        self.rxa = rxa

        rya = np.empty_like(self.ya,dtype=float)
        rya[1:-1] = new_dya[1:]/new_dya[:-1]
        rya[0] = 1
        rya[-1] = 1
        self.rya = rya

        self.dxa = np.empty_like(self.xa)
        self.dxa[1:] = new_dxa
        self.dxa[0] = self.dxa[1]

        self.dya = np.empty_like(self.ya)
        self.dya[1:] = new_dya
        self.dya[0] = self.dya[1]

        self.xres,self.yres = len(self.xa),len(self.ya)

        self.xg,self.yg = np.meshgrid(new_xa,new_ya,indexing='ij')

        #offset grids
        xhg = np.empty(( self.xg.shape[0] + 1 , self.xg.shape[1] ))
        yhg = np.empty(( self.yg.shape[0] , self.yg.shape[1] + 1 ))

        ne.evaluate("(a+b)/2",local_dict={"a":self.xg[1:],"b":self.xg[:-1]},out=xhg[1:-1])
        ne.evaluate("(a+b)/2",local_dict={"a":self.yg[:,1:],"b":self.yg[:,:-1]},out=yhg[:,1:-1])

        xhg[0] = self.xg[0] - self.dxa[0]*0.5
        xhg[-1] = self.xg[-1] + self.dxa[-1]*rxa[-1]*0.5

        yhg[:,0] = self.yg[:,0] - self.dya[0]*0.5
        yhg[:,-1] = self.yg[:,-1] + self.dya[-1]*self.rya[-1]*0.5

        self.xhg,self.yhg = xhg,yhg

        self.shape = (len(new_xa),len(new_ya))

    def get_weights(self):
        xhg,yhg = self.xhg,self.yhg
        weights = ne.evaluate("(a-b)*(c-d)",local_dict={"a":xhg[1:],"b":xhg[:-1],"c":yhg[:,1:],"d":yhg[:,:-1]})
        return weights

    def resample(self,u,xa=None,ya=None,newxa=None,newya=None):
        if xa is None or ya is None:
            out = RectBivariateSpline(self.xa_last,self.ya_last,u)(self.xa,self.ya)
        else:
            out = RectBivariateSpline(xa,ya,u)(newxa,newya)
        return out
    
    def resample_complex(self,u,xa=None,ya=None,newxa=None,newya=None):
        reals = np.real(u)
        imags = np.imag(u)
        reals = self.resample(reals,xa,ya,newxa,newya)
        imags = self.resample(imags,xa,ya,newxa,newya)
        return reals+1.j*imags

    def plot_mesh(self,reduce_by = 1,show=True):
        i=0
        for x in self.xa:
            if i%reduce_by == 0:
                plt.axhline(y=x,color='k',lw=0.5,alpha=0.5)
            i+=1
        i=0
        for y in self.ya:
            if i%reduce_by == 0:
                plt.axvline(x=y,color='k',lw=0.5,alpha=0.5)
            i+=1
        if show:
            plt.axis('equal')
            plt.show()
    
    def get_base_field(self,u):
        return u[self.xix_base].T[self.yix_base].T

    def _compute_refinement_factor(self,u0,crit_val):
        ''' given some electric field u0, compute values that corresponds to the degree of refinement required
            in the x and y subdivisions in the field. larger refinement factors should imply more grid refinement. 
            crit_val determines the normalization of this refinement factor - this is left as an argument
            to allow the degree of mesh subdivision to be controlled. note that the output refinement factors,
            _rx and _ry, essentially will act as booleans. wherever these factors are larger than 1, the grid will
            marked for subdivision.
        '''
        
        # the default scheme presented here sets the refinement factor to the geometric mean of field amplitude and 
        # field second derivative magnitude. convergence testing shows that this metric leads to faster convergence 
        # over using just field amplitude or second derivative magnitude alone. 
        # evidence for this is entirely empirical and comes from testing that is not 100% comprehensive.

        ix = self.ccel_ix

        # x second derivative estimation
        xdif2 = np.empty_like(u0,dtype=np.complex128)
        xdif2[1:-1] = u0[2:]+u0[:-2] - 2*u0[1:-1]
        xdif2[0] = xdif2[-1] = 0

        # y second derivative estimation
        ydif2 = np.empty_like(u0,dtype=np.complex128)
        ydif2[:,1:-1] = u0[:,2:]+u0[:,:-2] - 2*u0[:,1:-1]

        ydif2[:,0] = ydif2[:,-1] = 0

        # field amps
        umaxx = np.sqrt(np.max(np.abs(u0),axis=1) * np.max(np.abs(xdif2),axis=1))
        umaxx = 0.5*(umaxx[1:]+umaxx[:-1])

        umaxy = np.sqrt(np.max(np.abs(u0),axis=0) * np.max(np.abs(ydif2),axis=0))
        umaxy = 0.5*(umaxy[1:]+umaxy[:-1])

        _rx = umaxx[ix]*self.dxa[1:][ix]/crit_val
        _ry = umaxy[ix]*self.dya[1:][ix]/crit_val

        return _rx,_ry


    def refine_by_two(self,u0,crit_val):
        ''' uses a hybrid approach where cells tagged are based on the product of field amplitude
            and second derivative magnitude '''

        ix = self.ccel_ix

        _rx,_ry = self._compute_refinement_factor(u0,crit_val)

        rfacxa = np.full(u0.shape[0]-1,1,dtype=int)
        rfacya = np.full(u0.shape[1]-1,1,dtype=int)

        mask = (_rx>1)
        rfacxa[ix][mask] = 2

        mask = (_ry>1)
        rfacya[ix][mask] = 2

        xa_old = self.xa
        ya_old = self.ya

        self.update(rfacxa,rfacya)

        return self.resample_complex(u0,xa_old,ya_old,self.xa,self.ya)

    def refine_base(self,u0,ucrit):
        ''' iteratively apply refine_by_two to fully subdivide the simulation grid. '''
        for i in range(self.max_iters):
            u0 = self.refine_by_two(u0,ucrit)

class RectMesh3D:
    def __init__(self,xw,yw,zw,ds,dz,PML=4,xwfunc=None,ywfunc=None):
        '''base is a uniform mesh. can be refined'''
        self.xw,self.yw,self.zw = xw,yw,zw
        self.ds,self.dz = ds,dz
        self.xres,self.yres,self.zres = round(xw/ds)+1+2*PML, round(yw/ds)+1+2*PML, round(zw/dz)+1

        self.xa = np.linspace(-xw/2-PML*ds,xw/2+PML*ds,self.xres)
        self.ya = np.linspace(-xw/2-PML*ds,xw/2+PML*ds,self.yres)

        self.xg,self.yg = np.meshgrid(self.xa,self.ya,indexing='ij')

        self.shape=(self.zres,self.xres,self.yres)

        if xwfunc is None:
            xy_xw = xw
        else:
            xy_xw = 2*math.ceil(xwfunc(0)/2/ds)*ds
        
        if ywfunc is None:
            xy_yw = yw
        else:
            xy_yw = 2*math.ceil(ywfunc(0)/2/ds)*ds
            
        self.xy = RectMesh2D(xy_xw,xy_yw,ds,ds,PML)

        self.za = np.linspace(0,zw,self.zres)

        self.sigma_max = 5.+0.j #max (dimensionless) conductivity in PML layers
        self.PML = PML
        self.half_dz = dz/2.

        self.xwfunc = xwfunc
        self.ywfunc = ywfunc
    
    def get_loc(self ):

        xy = self.xy
        ix0 = bisect_left(self.xa,xy.xm-TOL)
        ix1 = bisect_left(self.xa,xy.xM-TOL)
        ix2 = bisect_left(self.ya,xy.ym-TOL)
        ix3 = bisect_left(self.ya,xy.yM-TOL)
        return ix0,ix1,ix2,ix3

    def sigmax(self,x):
        '''dimensionless, divided by e0 omega'''
        return np.where(np.abs(x)>self.xy.xw/2.,power((np.abs(x) - self.xy.xw/2)/(self.PML*self.xy.dx0),2.)*self.sigma_max,0.+0.j)
    
    def sigmay(self,y):
        '''dimensionless, divided by e0 omega'''
        return np.where(np.abs(y)>self.xy.yw/2.,power((np.abs(y) - self.xy.yw/2)/(self.PML*self.xy.dy0),2.)*self.sigma_max,0.+0.j)

