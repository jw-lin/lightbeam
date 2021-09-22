''' bunch of miscellaneous functions that I didn't know where to put'''

import numpy as np
from bisect import bisect_left
import time
from scipy.interpolate import RectBivariateSpline

def getslices(bounds,arr):
    '''given a range, get the idxs corresponding to that range in the sorted array arr '''
    if len(bounds)==0:
        return np.s_[0:0]
    elif len(bounds)==1:
        return np.s_[bisect_left(arr,bounds[0])]
    elif len(bounds)==2:
        return np.s_[bisect_left(arr,bounds[0]):bisect_left(arr,bounds[1])+1]
    else:
        raise Exception("malformed bounds input in getslices(); check savex,savey,savez in config.py")

def resize2(image,newshape):
    '''another resampling function that uses scipy, not cv2'''
    xpix = np.arange(image.shape[0])
    ypix = np.arange(image.shape[1])

    xpix_new = np.linspace(xpix[0],xpix[-1],newshape[0])
    ypix_new = np.linspace(ypix[0],ypix[-1],newshape[1])

    return RectBivariateSpline(xpix,ypix,image)(xpix_new,ypix_new)

def overlap(u1,u2,weight=1):
    return weight*np.abs(np.sum(np.conj(u2)*u1))

def overlap_nonu(u1,u2,weights):
    return np.abs(np.sum(weights*np.conj(u2)*u1))

def normalize(u0,weight=1,normval = 1):
    norm = np.sqrt(normval/overlap(u0,u0,weight))
    u0 *= norm
    return u0

def norm_nonu(u0,weights,normval = 1):
    norm = np.sqrt(normval/overlap_nonu(u0,u0,weights))
    u0 *= norm
    return u0

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Pulled from https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console

    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

def timeit(method):
    '''pulled from someone's github or something. can't find it anymore'''
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts))
        else:
            print('%r  %2.4f s' % \
                  (method.__name__, (te - ts)))
        return result
    return timed

def gauss(xg,yg,theta,phi,sigu,sigv,k0,x0=0,y0=0.):
    '''tilted gaussian beam'''
    u = np.cos(theta)*np.cos(phi)*(xg-x0) + np.cos(theta)*np.sin(phi)*(yg-y0)
    v = -np.sin(phi)*(xg-x0) + np.cos(phi)*(yg-y0)
    w = np.sin(theta)*np.cos(phi)*(xg-x0)  + np.sin(theta)*np.sin(phi)*(yg-y0)
    out = ( np.exp(1.j*k0*w)*np.exp(-0.5*np.power(u/sigu,2.)-0.5*np.power(v/sigv,2.)) ).astype(np.complex128)
    return out/np.sqrt(overlap(out,out))

def read_rsoft(fname):
    arr = np.loadtxt(fname,skiprows = 4).T
    reals = arr[::2]
    imags = arr[1::2]
    field = (reals+1.j*imags).T
    return field.astype(np.complex128)

def write_rsoft(fname,u0,xw,yw):
    '''save field to a file format useable by rsoft'''
    out = np.empty((u0.shape[0]*2,u0.shape[1]))
    reals = np.real(u0)
    imags = np.imag(u0)

    for j in range(out.shape[0]):
        if j%2==0:
            out[j] = reals[:,int(j/2)]
        else:
            out[j] = imags[:,int(j/2)]
    
    header = "/rn,a,b/nx0/ls1\n/r,qa,qb\n{} {} {} 0 OUTPUT_REAL_IMAG_3D\n{} {} {}".format(u0.shape[0],-xw/2,xw/2,u0.shape[1],-yw/2,yw/2)
    np.savetxt(fname+".dat", out.T, header = header, fmt = "%f",comments="",newline="\n")