import numpy as np
from numpy import logical_and as AND, logical_not as NOT
from bisect import bisect_left,bisect_right
import lightbeam.geom as geom
from typing import List
from lightbeam.mesh import RectMesh2D

### to do

# some ideas
#       -currently have outer bbox to speed up raster. maybe an inner bbox will also speed things up? this would force fancy indexing though...
#           -but the boundary region between inner and outer bbox could also be split into four contiguous blocks
#       -extension to primitives with elliptical cross sections
#           -I don't think this is that hard. An ellipse is a stretched circle. So antialiasing the ellipse on a rectangular grid is the same
#            as antialiasing a circle on another differently stretched rectangular grid.
#           -but as of now, this is unnecessary

class OpticPrim:
    '''base class for optical primitives (simple 3D shapes with a single IOR value)'''
    
    z_invariant = False
    
    def __init__(self,n):

        self.n = n
        self.n2 = n*n
        
        self.mask_saved = None

        # optionally, give prims a mesh to set the samplinig for IOR computations
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
        xa,ya = m.xa,m.ya

        xmin,xmax,ymin,ymax = self._bbox(z)
        imin = max(bisect_left(xa,xmin)-1,0)
        imax = min(bisect_left(xa,xmax)+1,len(xa))
        jmin = max(bisect_left(ya,ymin)-1,0)
        jmax = min(bisect_left(ya,ymax)+1,len(ya))
        return np.s_[imin:imax,jmin:jmax], np.s_[imin:imax+1,jmin:jmax+1]
    
    def set_sampling(self,xymesh:RectMesh2D):
        self.xymesh = xymesh

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
    
    def get_boundary(self,z):
        ''' given z, get mask which will select pixels that lie on top of the primitive boundary
            you must set the sampling first before you call this!'''
        
        xhg,yhg = self.xymesh.xhg,self.xymesh.yhg
        maskh = self._contains(xhg,yhg,z)
        mask_r = (NOT ( AND(AND(maskh[1:,1:] == maskh[:-1,1:], maskh[1:,1:] == maskh[1:,:-1]),maskh[:-1,:-1]==maskh[:-1,1:])))
        return mask_r

class scaled_cyl(OpticPrim):
    ''' cylinder whose offset from origin and radius scale in the same way'''
    def __init__(self,xy,r,z_ex,n,nb,z_offset=0,scale_func=None,final_scale=1):
        ''' Initialize a scaled cylinder, where the cross-sectional geometry along the object's 
            length is a scaled version of the initial geometry. 

            Args:
            xy -- Initial location of the center of the cylinder at z=0.
            r -- initial cylinder radius
            z_ex -- cylinder length
            n -- refractive index of cylinder
            nb -- background index (required for anti-aliasing)

            z_offset -- offset that sets the z-coord for the cylinder's front
            scale_func -- optional custom function. Should take in z and return a scale value. 
                          set to None to use a linear scale function, where the scale factor 
                          of the back end is set by ...
            final_scale -- the scale of the final cross-section geometry, 
                           relative to the initial geoemtry.
        '''

        super().__init__(n)
        self.p1 = p1 = [xy[0],xy[1],z_offset]
        self.p2 = [p1[0]*final_scale,p1[1]*final_scale,z_ex+z_offset]
        self.r = r
        self.rsq = r*r
        self.nb2 = nb*nb
        self.n2 = n*n
        self.z_ex = z_ex
        self.z_offset = z_offset

        def linear_func(_min,_max):
            slope =  (_max - _min)/self.z_ex
            def _inner_(z):
                return slope*(z-self.z_offset) + _min
            return _inner_

        if scale_func is None:
            scale_func = linear_func(1,final_scale)
        self.scale_func = scale_func
        self.xoffset_func = linear_func(p1[0],self.p2[0])
        self.yoffset_func = linear_func(p1[1],self.p2[1])

    def _contains(self,x,y,z):
        if not (self.z_offset <= z <= self.z_offset+self.z_ex):
            return False

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
        '''overwrite base function to incorporate anti-aliasing and improve convergence'''
        if not (self.z_offset <= z <= self.z_offset+self.z_ex):
            return

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
        '''replace values of out with IOR^2, given coordinate grids xg, yg, and z location.'''
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

class lant5big(OpticSys):
    '''corrigan et al. 2018 style photonic lantern except the jacket is infinite'''
    def __init__(self,rcore,rclad,ncore,nclad,njack,offset0,z_ex,scale_func=None,final_scale=1):
        core0 = scaled_cyl([0,0],rcore,z_ex,ncore,nclad,scale_func=scale_func,final_scale=final_scale)
        core1 = scaled_cyl([offset0,0],rcore,z_ex,ncore,nclad,scale_func=scale_func,final_scale=final_scale)
        core2 = scaled_cyl([0,offset0],rcore,z_ex,ncore,nclad,scale_func=scale_func,final_scale=final_scale)
        core3 = scaled_cyl([-offset0,0],rcore,z_ex,ncore,nclad,scale_func=scale_func,final_scale=final_scale)
        core4 = scaled_cyl([0,-offset0],rcore,z_ex,ncore,nclad,scale_func=scale_func,final_scale=final_scale)
        clad = scaled_cyl([0,0],rclad,z_ex,nclad,njack,scale_func=scale_func,final_scale=final_scale)
        elmnts = [clad,core4,core3,core2,core1,core0]
        
        super().__init__(elmnts,njack)

class lant3big(OpticSys):
    '''3 port lantern, infinite jacket'''
    def __init__(self,rcore,rclad,ncore,nclad,njack,offset0,z_ex,z_offset=0,scale_func=None,final_scale=1):
        core0 = scaled_cyl([0,offset0],rcore,z_ex,ncore,nclad,z_offset,scale_func=scale_func,final_scale=final_scale)
        core1 = scaled_cyl([-np.sqrt(3)/2*offset0,-offset0/2],rcore,z_ex,ncore,nclad,z_offset,scale_func=scale_func,final_scale=final_scale)
        core2 = scaled_cyl([np.sqrt(3)/2*offset0,-offset0/2],rcore,z_ex,ncore,nclad,z_offset,scale_func=scale_func,final_scale=final_scale)
        clad = scaled_cyl([0,0],rclad,z_ex,nclad,njack,z_offset,scale_func=scale_func,final_scale=final_scale)
        elmnts = [clad,core2,core1,core0]
        
        super().__init__(elmnts,njack)

        self.init_core_locs = np.array([[0,offset0],[-np.sqrt(3)/2*offset0,-offset0/2],[np.sqrt(3)/2*offset0,-offset0/2]])
        self.final_core_locs = self.init_core_locs*final_scale

class lant3_ms(OpticSys):
    '''3 port lantern, infinite jacket, one core is bigger than the rest to accept LP01 mode.'''
    def __init__(self,rcore1,rcore2,rclad,ncore,nclad,njack,offset0,z_ex,z_offset=0,scale_func=None,final_scale=1):
        core0 = scaled_cyl([0,offset0],rcore1,z_ex,ncore,nclad,z_offset,scale_func=scale_func,final_scale=final_scale)
        core1 = scaled_cyl([-np.sqrt(3)/2*offset0,-offset0/2],rcore2,z_ex,ncore,nclad,z_offset,scale_func=scale_func,final_scale=final_scale)
        core2 = scaled_cyl([np.sqrt(3)/2*offset0,-offset0/2],rcore2,z_ex,ncore,nclad,z_offset,scale_func=scale_func,final_scale=final_scale)
        clad = scaled_cyl([0,0],rclad,z_ex,nclad,njack,z_offset,scale_func=scale_func,final_scale=final_scale)
        elmnts = [clad,core2,core1,core0]
        super().__init__(elmnts,njack)

class lant6_saval(OpticSys):
    '''6 port lantern, mode-selective, based off sergio leon-saval's paper'''
    def __init__(self,rcore0,rcore1,rcore2,rcore3,rclad,ncore,nclad,njack,offset0,z_ex,z_offset=0,scale_func=None,final_scale=1):
        
        t = 2*np.pi/5
        core_locs = [[0,0]]
        for i in range(5):
            core_locs.append([offset0*np.cos(i*t),offset0*np.sin(i*t)])
        self.core_locs = np.array(core_locs)
        core0 = scaled_cyl(core_locs[0],rcore0,z_ex,ncore,nclad,z_offset,scale_func=scale_func,final_scale=final_scale)
        core1 = scaled_cyl(core_locs[1],rcore1,z_ex,ncore,nclad,z_offset,scale_func=scale_func,final_scale=final_scale)
        core2 = scaled_cyl(core_locs[2],rcore1,z_ex,ncore,nclad,z_offset,scale_func=scale_func,final_scale=final_scale)
        core3 = scaled_cyl(core_locs[3],rcore2,z_ex,ncore,nclad,z_offset,scale_func=scale_func,final_scale=final_scale)
        core4 = scaled_cyl(core_locs[4],rcore2,z_ex,ncore,nclad,z_offset,scale_func=scale_func,final_scale=final_scale)
        core5 = scaled_cyl(core_locs[5],rcore3,z_ex,ncore,nclad,z_offset,scale_func=scale_func,final_scale=final_scale)
        
        clad = scaled_cyl([0,0],rclad,z_ex,nclad,njack,z_offset,scale_func=scale_func,final_scale=final_scale)
        elmnts = [clad,core5,core4,core3,core2,core1,core0]
        
        super().__init__(elmnts,njack)

class lant19(OpticSys):
    '''19 port lantern, with cores hexagonally packed'''

    def __init__(self,rcore,rclad,ncore,nclad,njack,core_spacing,z_ex,z_offset=0,scale_func=None,final_scale=1):
        
        core_locs = self.get_19port_positions(core_spacing)
        self.core_locs = core_locs
        clad = scaled_cyl([0,0],rclad,z_ex,nclad,njack,z_offset,scale_func=scale_func,final_scale=final_scale)
        elmnts = [clad]

        for loc in core_locs:
            core = scaled_cyl(loc,rcore,z_ex,ncore,nclad,z_offset,scale_func=scale_func,final_scale=final_scale)
            elmnts.append(core)
        super().__init__(elmnts,njack)
        
    @staticmethod
    def get_19port_positions(core_spacing,plot=False):
        pos= [[0,0]]

        for i in range(6):
            xpos = core_spacing*np.cos(i*np.pi/3)
            ypos = core_spacing*np.sin(i*np.pi/3)
            pos.append([xpos,ypos])
        
        startpos = np.array([2*core_spacing,0])
        startang = 2*np.pi/3
        pos.append(startpos)
        for i in range(11):
            if i%2==0 and i!=0:
                startang += np.pi/3
            nextpos = startpos + np.array([core_spacing*np.cos(startang),core_spacing*np.sin(startang)])
            pos.append(nextpos)
            startpos = nextpos

        pos = np.array(pos)
        if not plot:
            return pos
        
        import matplotlib.pyplot as plt

        for p in pos:
            plt.plot(*p,marker='.',ms=10,color='k')
        
        plt.axis('equal')
        plt.show()
