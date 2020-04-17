import numpy as np
from scipy.special import jn_zeros, jn, kn
from scipy.optimize import brentq

def get_V(k0,coreradius,ncore,nclad):
    return k0 * coreradius * np.sqrt(ncore**2-nclad**2)

def get_modes(V):
    '''frequency cutoff occurs when b(V) = 0.  solve eqn 4.19.
    checks out w/ the function in the fiber ipynb'''

    l = 0
    m = 1
    modes = []
    while True:

        if l == 0:
            #solving dispersion relation leads us to the zeros of J_1
            #1st root of J_1 is 0. 

            modes.append((0,1))

            while jn_zeros(1,m)[m-1]< V:

                modes.append((l,m+1))
                m+=1
        else:
            #solving dispersion relation leads us to the zeros of J_l-1, neglecting 0
            if jn_zeros(l-1,1)[0]>V:
                break
            
            while jn_zeros(l-1,m)[m-1]<V:
                modes.append((l,m))
                m+=1
        m = 1
        l += 1
    return modes

def get_mode_cutoffs(l, mmax):
    from scipy.special import jn_zeros
    if l > 0:
        return jn_zeros(l-1, mmax)
    else:
        if mmax > 1:
            return np.concatenate(((0.,),jn_zeros(l-1, mmax-1)))
        else:
            return np.array((0.,))

def findBetween(solve_fn, lowbound, highbound, args=(), maxj=15):
    v = [lowbound, highbound]
    s = [solve_fn(lowbound, *args), solve_fn(highbound, *args)]
    
    if s[0] == 0.: return lowbound
    if s[1] == 0.: return highbound

    from itertools import count
    for j in count():  # probably not needed...
        if j == maxj:
            print("findBetween: max iter reached")
            return v[np.argmin(np.abs(s))]
            #return np.nan
        for i in range(len(s)-1):
            a, b = v[i], v[i+1]
            fa, fb = s[i], s[i+1]

            if (fa > 0 and fb < 0) or (fa < 0 and fb > 0):
                z = brentq(solve_fn, a, b, args=args)
                fz = solve_fn(z, *args)
                if abs(fa) > abs(fz) < abs(fb):  # Skip discontinuities
                    return z

        ls = len(s)
        for i in range(ls-1):
            a, b = v[2*i], v[2*i+1]
            c = (a + b) / 2
            v.insert(2*i+1, c)
            s.insert(2*i+1, solve_fn(c, *args))

def get_b(l, m, V):
    if l == 0:
        def solve_fn(b, V):
            v = V*np.sqrt(b)
            u = V*np.sqrt(1.-b)
            return (u * jn(1, u) * kn(0, v) - v * jn(0, u) * kn(1, v))
    else:
        def solve_fn(b, V):
            v = V*np.sqrt(b)
            u = V*np.sqrt(1.-b)
            return (u * jn(l - 1, u) * kn(l, v) + v * jn(l, u) * kn(l - 1, v))

    epsilon = 1.e-12
    
    Vc = get_mode_cutoffs(l+1, m)[-1]
    if V < Vc:
        lowbound = 0.
    else:
        lowbound = 1.-(Vc/V)**2    
    Vcl = get_mode_cutoffs(l, m)[-1]
    if V < Vcl: 
        return np.nan

    highbound = 1.-(Vcl/V)**2

    if np.isnan(lowbound): lowbound = 0.
    if np.isnan(highbound): highbound = 1.

    lowbound = np.max((lowbound-epsilon,0.+epsilon))
    highbound = np.min((highbound+epsilon,1.))
    b_opt = findBetween(solve_fn, lowbound, highbound, maxj=10, args=(V,))

    return b_opt

def lpfield(xg,yg,l,m,a,wl0,ncore,nclad,which = "cos"):
    '''calculate transverse field distribution of lp mode'''

    assert which in ("cos","sin"), "lp mode azimuthal component is either a cosine or sine, choose either 'cos' or 'sin'"

    V = get_V(2*np.pi/wl0,a,ncore,nclad)
    rs = np.sqrt(np.power(xg,2) + np.power(yg,2))
    b = get_b(l,m,V)

    #print(np.sqrt(nclad**2+b*(ncore**2-nclad**2)))

    u = V*np.sqrt(1-b)
    v = V*np.sqrt(b)

    fieldout = np.zeros_like(rs,dtype = np.complex128)
    
    inmask = np.nonzero(rs<=a)
    outmask = np.nonzero(rs>a)

    fieldout[inmask] = jn(l,u*rs[inmask]/a)
    fieldout[outmask] = jn(l,u)/kn(l,v) * kn(l, v*rs[outmask]/a)

    #cosine/sine modulation
    phis = np.arctan2(yg,xg)
    if which == 'cos':
        fieldout *= np.cos(l*phis)
    else:
        fieldout *= np.sin(l*phis)

    return fieldout
