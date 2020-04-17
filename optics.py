import numpy as np
from numpy import logical_and as AND, logical_not as NOT
from bisect import bisect_left,bisect_right
import geom
from typing import List
from mesh import RectMesh2D

class OpticPrim:
    '''base class for optical primitives (simple 3D shapes with a single IOR value)'''
    
    z_invariant = False
    
    def __init__(self,n):

        ## these parameters allow you to assign a sampling grid to the primitive
        ## so that it can automatically compute IOR dists.
        
        self.xg = None
        self.yg = None
        self.xhg = None
        self.yhg = None
        self.ds = None

        self.n = n
        self.n2 = n*n
        
        self.mask_saved = None

        self.xymesh = None
    
    def _bbox(self,z):
        '''calculate the 2D bounding box of the primitive at given z. allows for faster IOR computation. Should be overwritten.'''
        return (-np.inf,np.inf,-np.inf,np.inf)

    def _contains(self,x,y,z):
        '''given coords, return whether or not those coords are inside the element. Should be overwritten.'''
        return np.full_like(x,False)

    def bbox_idx(self,z):
        '''get index slice corresponding to the primitives bbox, given an xg,yg coord grid'''
        m = self.xymesh
        xa,ya,xg,yg = m.xa,m.ya,m.xg,m.yg

        xmin,xmax,ymin,ymax = self._bbox(z)
        imin = max(bisect_left(xa,xmin)-1,0)
        imax = min(bisect_left(xa,xmax)+1,len(xa))
        jmin = max(bisect_left(ya,ymin)-1,0)
        jmax = min(bisect_left(ya,ymax)+1,len(ya))
        return np.s_[imin:imax,jmin:jmax], np.s_[imin:imax+1,jmin:jmax+1]
    
    def set_sampling(self,xymesh:RectMesh2D):
        self.xymesh = xymesh
    
    def IOR_sq(self,out,z,xg,yg,min_ds):
        pass

    def set_IORsq(self,out,z,coeff=1):
        ''' replace values of out with IOR^2, given coordinate grids xg, yg, and z location. 
            assumed primitive already has a set sampling grid'''

        if self.z_invariant and self.mask_saved is not None:
            mask = self.mask_saved
        else:    
            bbox,bboxh = self.bbox_idx(z)
            mask = self._contains(self.xymesh.xg[bbox],self.xymesh.yg[bbox],z)
        
        if self.z_invariant and self.mask_saved is None:
            self.mask_saved = mask
        
        out[bbox][mask] = self.n2*coeff

    def get_guided_mode(self,xg,yg,wl0):
        pass
    
    def get_boundary(self,z):
        ''' given z, get mask which will select pixels that lie on top of the primitive boundary
            you must set the sampling first before you call this!'''
        
        xhg,yhg = self.xymesh.xhg,self.xymesh.yhg
        maskh = self._contains(xhg,yhg,z)
        mask_r = (NOT ( AND(AND(maskh[1:,1:] == maskh[:-1,1:], maskh[1:,1:] == maskh[1:,:-1]),maskh[:-1,:-1]==maskh[:-1,1:])))
        return mask_r

class scaled_cyl(OpticPrim):
    ''' cylinder whose offset from origin and radius scale in the same way'''
    def __init__(self,xy,r,z_ex,n,nb,scale_func=None,final_scale=1):
        super().__init__(n)
        self.p1 = p1 = [xy[0],xy[1],0]
        self.p2 = [p1[0]*final_scale,p1[1]*final_scale,z_ex]
        self.r = r
        self.rsq = r*r
        self.nb2 = nb*nb
        self.n2 = n*n
        self.z_ex = z_ex

        def linear_func(_min,_max):
            slope =  (_max - _min)/self.z_ex
            def _inner_(z):
                return slope*z + _min
            return _inner_

        if scale_func is None:
            scale_func = linear_func(1,final_scale)
        self.scale_func = scale_func
        self.xoffset_func = linear_func(p1[0],self.p2[0])
        self.yoffset_func = linear_func(p1[1],self.p2[1])

    def _contains(self,x,y,z):
        xdist = x - self.xoffset_func(z)
        ydist = y - self.yoffset_func(z)
        scale = self.scale_func(z)
        return (xdist*xdist + ydist*ydist <= scale*scale*self.rsq)

    def _bbox(self,z):
        xc = self.xoffset_func(z)
        yc = self.yoffset_func(z)
        scale = self.scale_func(z)
        xmax = xc+scale*self.r
        xmin = xc-scale*self.r
        ymax = yc+scale*self.r
        ymin = yc-scale*self.r
        return (xmin,xmax,ymin,ymax)
    
    def set_IORsq(self,out,z,coeff=1):
        '''anti-aliased to improve convergence'''
        center = (self.xoffset_func(z),self.yoffset_func(z))
        scale = self.scale_func(z)
        bbox,bboxh = self.bbox_idx(z)  
        xg = self.xymesh.xg[bbox]
        yg = self.xymesh.yg[bbox]
        xhg = self.xymesh.xhg[bboxh]
        yhg = self.xymesh.yhg[bboxh]

        m = self.xymesh
        rxg,ryg = np.meshgrid(m.rxa,m.rya,indexing='ij')
        dxg,dyg = np.meshgrid(m.dxa,m.dya,indexing='ij')

        geom.AA_circle_nonu(out,xg,yg,xhg,yhg,center,self.r*scale,self.nb2*coeff,self.n2*coeff,bbox,rxg,ryg,dxg,dyg)
    
class OpticSys(OpticPrim):
    '''base class for optical systems, collections of primitives immersed in some background medium'''
    def __init__(self,elmnts:List[OpticPrim],nb):
        self.elmnts = elmnts
        self.nb = nb
        self.nb2 = nb*nb
    
    def _bbox(self,z):
        '''default behavior. won't work if the system has disjoint pieces'''
        if len(self.elmnts)==0:
            return super()._bbox(z)
        return self.elmnts[0]._bbox(z)
    
    def _contains(self,x,y,z):
        return self.elmnts[0]._contains(x,y,z)

    def set_sampling(self,xymesh:RectMesh2D):
        '''this function sets the spatial sampling for IOR computaitons'''
        super().set_sampling(xymesh)
        for elmnt in self.elmnts:
            elmnt.set_sampling(xymesh)

    def set_IORsq(self,out,z,xg=None,yg=None,coeff=1):
        '''replace values of out with IOR^2, given coordinate grids xg, yg, and z location'''
        xg = self.xymesh.xg if xg is None else xg
        yg = self.xymesh.yg if yg is None else yg
        bbox,bboxh = self.bbox_idx(z)
        out[bbox] = self.nb2*coeff
        for elmnt in self.elmnts:
            elmnt.set_IORsq(out,z,coeff)
        
class lant5(OpticSys):
    '''corrigan et al. 2018 style photonic lantern'''
    def __init__(self,rcore,rclad,rjack,ncore,nclad,njack,offset0,z_ex,scale_func=None,final_scale=1,nb=1):
        core0 = scaled_cyl([0,0],rcore,z_ex,ncore,nclad,scale_func=scale_func,final_scale=final_scale)
        core1 = scaled_cyl([offset0,0],rcore,z_ex,ncore,nclad,scale_func=scale_func,final_scale=final_scale)
        core2 = scaled_cyl([0,offset0],rcore,z_ex,ncore,nclad,scale_func=scale_func,final_scale=final_scale)
        core3 = scaled_cyl([-offset0,0],rcore,z_ex,ncore,nclad,scale_func=scale_func,final_scale=final_scale)
        core4 = scaled_cyl([0,-offset0],rcore,z_ex,ncore,nclad,scale_func=scale_func,final_scale=final_scale)
        clad = scaled_cyl([0,0],rclad,z_ex,nclad,njack,scale_func=scale_func,final_scale=final_scale)
        jack = scaled_cyl([0,0],rjack,z_ex,njack,nb,scale_func=scale_func,final_scale=final_scale)
        elmnts = [jack,clad,core4,core3,core2,core1,core0]
        
        super().__init__(elmnts,nb)


'''
core0.set_sampling(mesh)

out = np.full_like(mesh.xg,1.4*1.4)
core0.set_IORsq(out,0)

plt.imshow(out)
plt.show()

xg,yg = np.meshgrid(mesh.xa,mesh.ya,indexing='ij')
rg2 = yg*yg+xg*xg

gauss = np.exp(-rg2/100)
avg = np.sum(gauss)/(108*108)
m = avg*10

mesh.remesh(gauss,m,mode="max")

core0.set_sampling(mesh)
out = np.full_like(mesh.xg,1.4*1.4)
core0.set_IORsq(out,0)

plt.imshow(out)
plt.show()
'''


'''
yg,xg = np.meshgrid(mesh.xa,mesh.ya)

rg2 = (yg-50)*(yg-50)+xg*xg

gauss2 = np.exp(-rg2/100)


mesh.remesh(gauss2,.1)

gauss3 = mesh.resample(gauss2)

gauss3 = mesh.get_base_field(gauss3)

plt.imshow(gauss3,extent=(-58,58,-58,58),origin="lower")
mesh.plot_mesh()
'''
