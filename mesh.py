import numpy as np
from numpy import s_,arange,sqrt,power,complex64 as c64
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt
from itertools import chain
from bisect import bisect_left
import numexpr as ne
import math

TOL=1e-12

class RectMesh2D:
    ''' transverse adapative mesh class '''

    def __init__(self,xw,yw,dx,dy,Nbc=4,u0=None,ucrit=None):

        self.Nbc = Nbc
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

        self.xa = None
        self.ya = None

        #self.xa0_comp = np.linspace(-xw/2,xw/2,self.shape0_comp[0])
        #self.ya0_comp = np.linspace(-yw/2,yw/2,self.shape0_comp[1])

        self.ccel_ix = np.s_[Nbc+1:-Nbc-1]

        #self.xres0,self.yres0 = xres,yres
        self.dx0,self.dy0 = dx,dy

        self.max_ratio = 64

        self.cvert_ix = np.s_[Nbc+1:-Nbc]
        self.pvert_ix = np.hstack((arange(Nbc+1),arange(-Nbc-1,0)))

        self.pvert_xa = self.xa0[self.pvert_ix]
        self.pvert_ya = self.ya0[self.pvert_ix]

        self.rfacxa,self.rfacya = np.full(xres-1,1),np.full(yres-1,1)

        self.updatecoords2(self.rfacxa,self.rfacya)

        self.max_iters = 8

        #if u0 is not None:
        #    self.refine_base(u0,ucrit)
    
    def dxa2xa(self,dxa):
        N = len(dxa)
        out = np.zeros(N+1)
        np.cumsum(dxa,out=out[1:])
        return out + self.xm

    def updatecoords2(self,rfacxa,rfacya):

        self.rfacxa = rfacxa
        self.rfacya = rfacya

        self.xix_base = np.empty(len(rfacxa)+1,dtype=int)
        self.yix_base = np.empty(len(rfacya)+1,dtype=int)

        self.xix_base[1:] = np.cumsum(rfacxa)
        self.yix_base[1:] = np.cumsum(rfacya)
        self.xix_base[0] = 0
        self.yix_base[0] = 0

        new_dxa = np.repeat(self.dx0/rfacxa,rfacxa)
        new_dya = np.repeat(self.dy0/rfacya,rfacya)

        new_xa = self.dxa2xa(new_dxa)
        new_ya = self.dxa2xa(new_dya)

        #self.xa_last = self.xa
        #self.ya_last = self.ya
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

        #xhg[1:-1] = (self.xg[1:] + self.xg[:-1])/2
        #yhg[:,1:-1] = (self.yg[:,1:]+self.yg[:,:-1])/2

        xhg[0] = self.xg[0] - self.dxa[0]*0.5
        xhg[-1] = self.xg[-1] + self.dxa[-1]*rxa[-1]*0.5

        yhg[:,0] = self.yg[:,0] - self.dya[0]*0.5
        yhg[:,-1] = self.yg[:,-1] + self.dya[-1]*self.rya[-1]*0.5

        self.xhg,self.yhg = xhg,yhg

        #self.weights = (xhg[1:]-xhg[:-1]) * (yhg[:,1:]-yhg[:,:-1])
        self.weights = self.get_weights()
        
        #ix = self.cvert_ix

        #get views of computational region of arrs
        '''
        self.cxa = self.xa[ix]
        self.cya = self.ya[ix]
        self.cdxa = self.dxa[ix]
        self.cdya = self.dya[ix]
        self.crxa = self.rxa[ix]
        self.crya = self.rya[ix]
        '''

        self.shape = (len(new_xa),len(new_ya))
    
    def updatecoords3(self,rfacxa,rfacya):
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

        #xhg[1:-1] = (self.xg[1:] + self.xg[:-1])/2
        #yhg[:,1:-1] = (self.yg[:,1:]+self.yg[:,:-1])/2

        xhg[0] = self.xg[0] - self.dxa[0]*0.5
        xhg[-1] = self.xg[-1] + self.dxa[-1]*rxa[-1]*0.5

        yhg[:,0] = self.yg[:,0] - self.dya[0]*0.5
        yhg[:,-1] = self.yg[:,-1] + self.dya[-1]*self.rya[-1]*0.5

        self.xhg,self.yhg = xhg,yhg

        #self.weights = (xhg[1:]-xhg[:-1]) * (yhg[:,1:]-yhg[:,:-1])
        self.weights = self.get_weights()
        
        #ix = self.cvert_ix

        #get views of computational region of arrs
        '''
        self.cxa = self.xa[ix]
        self.cya = self.ya[ix]
        self.cdxa = self.dxa[ix]
        self.cdya = self.dya[ix]
        self.crxa = self.rxa[ix]
        self.crya = self.rya[ix]
        '''

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

    def refine_base(self,u0,ucrit,maxr=None):
        '''split the mesh until a max of du is stored in each cell'''

        #umid = ne.evaluate("0.25*(u1+u2+u3+u4)",local_dict={"u1":u0[1:,1:],"u2":u0[:-1,1:],"u3":u0[1:,:-1],"u4":u0[:-1,:-1]})

        #umid = 0.25 * (u0[1:,1:]+u0[:-1,1:]+u0[1:,:-1]+u0[:-1,:-1])
        #umaxx = np.max(umid,axis=1)
        #umaxy = np.max(umid,axis=0)

        if maxr is None:
            maxr = self.max_ratio

        ix = self.ccel_ix

        umaxx = np.max(u0,axis=1)
        umaxx = 0.5*(umaxx[1:]+umaxx[:-1])

        umaxy = np.max(u0,axis=0)
        umaxy = 0.5*(umaxy[1:]+umaxy[:-1])

        rfacxa = np.full_like(umaxx,1,dtype=int)
        _rx = umaxx[ix]/ucrit

        mask = (_rx>1)
        rfacxa[ix][mask] = np.clip(np.power(2,np.ceil(np.log2(_rx[mask]))),1, maxr )

        rfacya = np.full_like(umaxy,1,dtype=int)
        _ry = umaxy[ix]/ucrit
        mask = (_ry>1)
        rfacya[ix][mask] = np.clip(np.power(2,np.ceil(np.log2(_ry[mask]))),1, maxr)

        self.updatecoords2(rfacxa,rfacya)
    
    def refine_by_two(self,u0,ucrit):

        ix = self.ccel_ix
        umaxx = np.max(u0,axis=1)
        umaxx = 0.5*(umaxx[1:]+umaxx[:-1])

        umaxy = np.max(u0,axis=0)
        umaxy = 0.5*(umaxy[1:]+umaxy[:-1])

        rfacxa = np.full_like(umaxx,1,dtype=int)
        _rx = umaxx[ix]*self.dxa[1:][ix]/ucrit

        mask = (_rx>1)
        rfacxa[ix][mask] = 2

        rfacya = np.full_like(umaxy,1,dtype=int)
        _ry = umaxy[ix]*self.dya[1:][ix]/ucrit
        mask = (_ry>1)
        rfacya[ix][mask] = 2

        xa_old = np.copy(self.xa)
        ya_old = np.copy(self.ya)

        self.updatecoords3(rfacxa,rfacya)

        return self.resample(u0,xa_old,ya_old,self.xa,self.ya)
    
    def reset(self):
        self.xa_last = self.xa
        self.ya_last = self.ya
        xres,yres = self.shape0
        self.rfacxa,self.rfacya = np.full(xres-1,1),np.full(yres-1,1)
        self.updatecoords2(self.rfacxa,self.rfacya)

    def refine_base2(self,u0,ucrit):
        for i in range(self.max_iters):
            u0 = self.refine_by_two(u0,ucrit)

    def plot_mesh(self,reduce_by = 1,show=True):
        i = 0
        for x in self.xa:
            if i%reduce_by == 0:
                plt.axhline(y=x,color='white',lw=0.5,alpha=0.5)
            i+=1
        i=0
        for y in self.ya:
            if i%reduce_by == 0:
                plt.axvline(x=y,color='white',lw=0.5,alpha=0.5)
            i+=1
        if show:
            plt.axes().set_aspect('equal')
            plt.show()
    
    def get_base_field(self,u):
        return u[self.xix_base].T[self.yix_base].T
    
    def expand(self,new_xw,new_yw):
        '''expand grid to encompass new size by tiling new dx0 x dy0 cells around the grid boudary. grid stays 0-centered '''
        Nbc = self.Nbc 

        xwr = 2*math.ceil(new_xw/2/self.dx0)
        ywr = 2*math.ceil(new_yw/2/self.dy0)

        #round
        new_xw = xwr*self.dx0
        new_yw = ywr*self.dy0

        if (new_xw == self.xw) and (new_yw == self.yw):
            return False

        #for base mesh!
        new_xres = xwr + 1 + 2 * Nbc
        new_yres = ywr + 1 + 2 * Nbc
        new_shape = ( new_xres , new_yres )

        #snap new grid dims to the dx0 dy0 grid

        dx,dy = self.dx0,self.dy0

        xex = int((new_xres - self.shape0[0])/2)
        yex = int((new_yres - self.shape0[1])/2)

        self.xa_last = np.hstack( (np.linspace(-new_xw/2-Nbc*dx,-self.xw/2-dx-Nbc*dx,xex),self.xa_last,np.linspace(self.xw/2+Nbc*dx+dx,new_xw/2+Nbc*dx,xex) ) )
        self.ya_last = np.hstack( (np.linspace(-new_yw/2-Nbc*dy,-self.yw/2-dy-Nbc*dy,yex),self.ya_last,np.linspace(self.yw/2+Nbc*dy+dy,new_yw/2+Nbc*dy,yex) ) )
    
        #rewrite some mesh parameters ...
        self.xa0 = np.linspace(-new_xw/2-Nbc*dx,new_xw/2+Nbc*dx,new_shape[0])
        self.ya0 = np.linspace(-new_yw/2-Nbc*dy,new_yw/2+Nbc*dy,new_shape[1])

        self.pvert_xa = self.xa0[self.pvert_ix]
        self.pvert_ya = self.ya0[self.pvert_ix]

        self.xM = (new_xres-1)/2*self.dx0
        self.xm = -self.xM
        self.yM = (new_yres-1)/2*self.dy0
        self.ym = -self.yM

        #now we need to set the rfactors so they match
        xix_boundary = int((new_xres-self.shape0[0])/2)
        yix_boundary = int((new_yres-self.shape0[1])/2)

        rfacxa = np.full(new_shape[0]-1,1,dtype=int)  
        rfacya = np.full(new_shape[1]-1,1,dtype=int)  

        rfacxa[xix_boundary:-xix_boundary] = self.rfacxa
        rfacya[yix_boundary:-yix_boundary] = self.rfacya

        self.xw,self.yw = new_xw,new_yw

        self.shape0 = new_shape
        self.updatecoords2(rfacxa,rfacya)

        return True

    def refine_by_two_alt(self,u0,crit_val):

        ix = self.ccel_ix

        xdif2 = np.empty_like(u0,dtype=np.complex128)
        xdif2[1:-1] = u0[2:]+u0[:-2] - 2 * u0[1:-1]

        xdif2[0] = xdif2[-1] = 0

        ydif2 = np.empty_like(u0,dtype=np.complex128)
        ydif2[:,1:-1] = u0[:,2:]+u0[:,:-2] - 2 * u0[:,1:-1]

        ydif2[:,0] = ydif2[:,-1] = 0

        umaxx = np.max(np.abs(xdif2),axis=1)*np.max(np.abs(u0),axis=1)
        umaxx = 0.5*(umaxx[1:]+umaxx[:-1])

        umaxy = np.max(np.abs(ydif2),axis=0)*np.max(np.abs(u0),axis=0)
        umaxy = 0.5*(umaxy[1:]+umaxy[:-1])

        rfacxa = np.full_like(umaxx,1,dtype=int)
        _rx = umaxx[ix]*self.dxa[1:][ix]/crit_val

        mask = (_rx>1)
        rfacxa[ix][mask] = 2

        rfacya = np.full_like(umaxy,1,dtype=int)
        _ry = umaxy[ix]*self.dya[1:][ix]/crit_val
        mask = (_ry>1)
        rfacya[ix][mask] = 2

        xa_old = np.copy(self.xa)
        ya_old = np.copy(self.ya)

        self.updatecoords3(rfacxa,rfacya)

        return self.resample_complex(u0,xa_old,ya_old,self.xa,self.ya)

    def refine_base_alt(self,u0,ucrit):
        for i in range(self.max_iters):
            u0 = self.refine_by_two_alt(u0,ucrit)

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

'''
plt.style.use('dark_background')
u0 = np.load("PSF0lo.npy")
u0/=np.max(np.abs(u0))
xdif2 = np.empty_like(u0,dtype=np.complex128)
xdif2[1:-1] = u0[2:]+u0[:-2] - 2 * u0[1:-1]

xdif2[0] = xdif2[-1] = 0

ydif2 = np.empty_like(u0,dtype=np.complex128)
ydif2[:,1:-1] = u0[:,2:]+u0[:,:-2] - 2 * u0[:,1:-1]

ydif2[:,0] = ydif2[:,-1] = 0

#plt.imshow(np.abs(u0))
#plt.show()
_xdif2 = np.abs(xdif2)
_xdif2 /= np.max(_xdif2)

#plt.imshow(np.abs(u0))
#plt.show()

#plt.imshow(_xdif2)
#plt.show()


#plt.imshow(np.abs(ydif2))
#plt.show()
import cv2
from misc import resize
u0 = resize(u0,(237,237))

c = 2e-4

xy = RectMesh2D(440,440,2,2,8)
u0 = xy.refine_by_two_dif(u0,c)
print(xy.shape)
#plt.imshow(np.abs(u0),extent=(-224,224,-224,224))
xy.plot_mesh(8)

u0 = xy.refine_by_two_dif(u0,c)
print(xy.shape)
#plt.imshow(np.abs(u0),extent=(-224,224,-224,224))
xy.plot_mesh(8)

u0 = xy.refine_by_two_dif(u0,c)
print(xy.shape)
#plt.imshow(np.abs(u0),extent=(-224,224,-224,224))
xy.plot_mesh(8)

u0 = xy.refine_by_two_dif(u0,c)
print(xy.shape)
#plt.imshow(np.abs(u0),extent=(-224,224,-224,224))
xy.plot_mesh(8)

u0 = xy.refine_by_two_dif(u0,c)
print(xy.shape)
#plt.imshow(np.abs(u0),extent=(-224,224,-224,224))
xy.plot_mesh(8)

u0 = xy.refine_by_two_dif(u0,c)
print(xy.shape)
#plt.imshow(np.abs(u0),extent=(-224,224,-224,224))
xy.plot_mesh(8)

u0 = xy.refine_by_two_dif(u0,c)
print(xy.shape)
#plt.imshow(np.abs(u0),extent=(-224,224,-224,224))
xy.plot_mesh(8)

u0 = xy.refine_by_two_dif(u0,c)
print(xy.shape)
#plt.imshow(np.abs(u0),extent=(-224,224,-224,224))
xy.plot_mesh()

print(xy.shape)

'''
'''
c = 0.0012

u0 = np.load("PSF0lo.npy")

xy = RectMesh2D(440,440,0.5,0.5,8)

plt.imshow(np.abs(u0),extent=(-224,224,-224,224))
xy.plot_mesh(16)

xy.refine_base2(np.abs(u0),c)
plt.imshow(np.abs(u0),extent=(-224,224,-224,224))
xy.plot_mesh(16)

print(xy.shape)

xy.reset()
xy.plot_mesh(16)

print(xy.shape)
'''
'''
print(xy.shape)

avgu = np.mean(u0)

du = avgu * xy.dx0*xy.dy0 * 4


xy.refine_base(u0,0.005)

print(xy.shape)

plt.imshow(u0,origin="lower",extent=(-224,224,-224,224))
xy.plot_mesh(reduce_by=8)
plt.show()
'''