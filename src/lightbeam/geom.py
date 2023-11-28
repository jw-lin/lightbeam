import numpy as np
from numpy import logical_not as NOT, logical_and as AND, logical_or as OR
from numba import njit

'''a collection of functions for antialiasing circles'''

## calculate circle-square overlap.

# original code in pixwt.c by Marc Buie
# ported to pixwt.pro (IDL) by Doug Loucks, Lowell Observatory, 1992 Sep
# subsequently ported to python by Michael Fitzgerald,
# LLNL, fitzgerald15@llnl.gov, 2007-10-16

### Marc Buie, you are my hero

def _arc(x, y0, y1, r):
    """
    Compute the area within an arc of a circle.  The arc is defined by
    the two points (x,y0) and (x,y1) in the following manner: The
    circle is of radius r and is positioned at the originp.  The origin
    and each individual point define a line which intersects the
    circle at some point.  The angle between these two points on the
    circle measured from y0 to y1 defines the sides of a wedge of the
    circle.  The area returned is the area of this wedge.  If the area
    is traversed clockwise then the area is negative, otherwise it is
    positive.
    """
    thetas = np.empty_like(x)
    mask = (x==0)
    thetas[mask] = np.pi/2*(np.sign(y1[mask])-np.sign(y0[mask]))
    thetas[~mask] = np.arctan(y1[~mask]/x[~mask])-np.arctan(y0[~mask]/x[~mask])
    return 0.5*r**2*thetas
    #thetas = np.where(x==0, np.pi/2*(np.sign(y1)-np.sign(y0)),np.arctan(y1/x)-np.arctan(y0/x))
    #return 0.5 * r**2 * (np.arctan2(y1,x) - np.arctan2(y0,x))

def _chord(x, y0, y1):
    """
    Compute the area of a triangle defined by the origin and two
    points, (x,y0) and (x,y1).  This is a signed area.  If y1 > y0
    then the area will be positive, otherwise it will be negative.
    """
    return 0.5 * x * (y1 - y0)

def _oneside(x, y0, y1, r):
    """
    Compute the area of intersection between a triangle and a circle.
    The circle is centered at the origin and has a radius of r.  The
    triangle has verticies at the origin and at (x,y0) and (x,y1).
    This is a signed area.  The path is traversed from y0 to y1.  If
    this path takes you clockwise the area will be negative.
    """

    if np.all((x==0)): return x

    sx = x.shape
    ans = np.zeros(sx, dtype=np.float64)
    yh = np.zeros(sx, dtype=np.float64)
    to = (abs(x) >= r)
    ti = (abs(x) < r)
    if np.any(to):
        ans[to] = _arc(x[to], y0[to], y1[to], r)
    if not np.any(ti):
        return ans

    yh[ti] = np.sqrt(r**2 - x[ti]**2)

    i = ((y0 <= -yh) & ti)
    if np.any(i):

        j = ((y1 <= -yh) & i)
        if np.any(j):
            ans[j] = _arc(x[j], y0[j], y1[j], r)

        j = ((y1 > -yh) & (y1 <= yh) & i)
        if np.any(j):
            ans[j] = _arc(x[j], y0[j], -yh[j], r) + \
                     _chord(x[j], -yh[j], y1[j])

        j = ((y1 > yh) & i)
        if np.any(j):
            ans[j] = _arc(x[j], y0[j], -yh[j], r) + \
                     _chord(x[j], -yh[j], yh[j]) + \
                     _arc(x[j], yh[j], y1[j], r)

    i = ((y0 > -yh) & (y0 < yh) & ti)
    if np.any(i):

        j = ((y1 <= -yh) & i)
        if np.any(j):
            ans[j] = _chord(x[j], y0[j], -yh[j]) + \
                     _arc(x[j], -yh[j], y1[j], r)

        j = ((y1 > -yh) & (y1 <= yh) & i)
        if np.any(j):
            ans[j] = _chord(x[j], y0[j], y1[j])

        j = ((y1 > yh) & i)
        if np.any(j):
            ans[j] = _chord(x[j], y0[j], yh[j]) + \
                     _arc(x[j], yh[j], y1[j], r)
        
    i = ((y0 >= yh) & ti)
    if np.any(i):

        j = ((y1 <= -yh) & i)
        if np.any(j):
            ans[j] = _arc(x[j], y0[j], yh[j], r) + \
                     _chord(x[j], yh[j], -yh[j]) + \
                     _arc(x[j], -yh[j], y1[j], r)

        j = ((y1 > -yh) & (y1 <= yh) & i)
        if np.any(j):
            ans[j] = _arc(x[j], y0[j], yh[j], r) + \
                     _chord(x[j], yh[j], y1[j])

        j = ((y1 > yh) & i)
        if np.any(j):
            ans[j] = _arc(x[j], y0[j], y1[j], r)
        
    return ans

def _intarea(xc, yc, r, x0, x1, y0, y1):
    """
    Compute the area of overlap of a circle and a rectangle.
      xc, yc  :  Center of the circle.
      r       :  Radius of the circle.
      x0, y0  :  Corner of the rectangle.
      x1, y1  :  Opposite corner of the rectangle.
    """
    x0 = x0 - xc
    y0 = y0 - yc
    x1 = x1 - xc
    y1 = y1 - yc
    return _oneside(x1, y0, y1, r) + _oneside(y1, -x1, -x0, r) + \
           _oneside(-x0, -y1, -y0, r) + _oneside(-y0, x0, x1, r)

def nonu_pixwt(xc,yc,r,x,y,rx,ry,dx,dy):

    area = dx*dy/4*(rx+1)*(ry+1)
    return _intarea(xc,yc,r,x-0.5*dx,x+0.5*rx*dx,y-0.5*dy,y+0.5*ry*dy)/area

def AA_circle_nonu(out,xg,yg,xgh,ygh,center,R,n0,n1,where,rxg,ryg,dxg,dyg):

    xdif = xgh-center[0]
    ydif = ygh-center[1]
    rsqh = xdif*xdif+ydif*ydif
    mask_in,mask_b = get_masks(rsqh,R*R)
    out[where][mask_in] = n1
    x0,y0 = xg[mask_b],yg[mask_b]

    rx,ry = rxg[where][mask_b], ryg[where][mask_b]
    dx,dy = dxg[where][mask_b], dyg[where][mask_b]

    area = nonu_pixwt(center[0],center[1],R,x0,y0,rx,ry,dx,dy)
    newvals = n1*area+n0*(1-area)
    out[where][mask_b] = newvals

def pixwt(xc, yc, r, x, y):
    """
    Compute the fraction of a unit pixel that is interior to a circle.
    The circle has a radius r and is centered at (xc, yc).  The center
    of the unit pixel (length of sides = 1) is at (x, y).

    Divides the circle and rectangle into a series of sectors and
    triangles.  Determines which of nine possible cases for the
    overlap applies and sums the areas of the corresponding sectors
    and triangles.

    area = pixwt( xc, yc, r, x, y )

    xc, yc : Center of the circle, numeric scalars
    r      : Radius of the circle, numeric scalars
    x, y   : Center of the unit pixel, numeric scalar or vector
    """
    return _intarea(xc, yc, r, x-0.5, x+0.5, y-0.5, y+0.5)

def get_masks(rsqh,R2):
    maskh = np.full(rsqh.shape,False)
    maskh[rsqh<=R2] = True

    mask_in = AND(maskh[1:,1:], AND(maskh[1:,:-1], AND(maskh[:-1,1:],maskh[:-1,:-1]))) 
    mask_out = AND(NOT(maskh[1:,1:]), AND(NOT(maskh[1:,:-1]), AND(NOT(maskh[:-1,1:]),NOT(maskh[:-1,:-1])))) 
    mask_b = NOT(OR(mask_in,mask_out))
    
    return mask_in,mask_b

def AA_circle(out,xg,yg,xgh,ygh,center,R,n0,n1,ds,where):
    xdif = xgh-center[0]
    ydif = ygh-center[1]
    rsqh = xdif*xdif+ydif*ydif
    mask_in,mask_b = get_masks(rsqh,R*R)
    out[where][mask_in] = n1
    x0,y0 = xg[mask_b],yg[mask_b]
    area = pixwt(center[0]/ds,center[1]/ds,R/ds,x0/ds,y0/ds)
    newvals = n1*area+n0*(1-area)
    out[where][mask_b] = newvals













    