import numpy as np
from misc import write_rsoft,read_rsoft,overlap,resize,normalize
import matplotlib.pyplot as plt
import LPmodes
from bisect import bisect_left
from scipy.optimize import curve_fit
from scipy.special import j0,k0
import LPmodes

np.random.seed(1)
x = np.random.randint(0,10)
print(x)
print(x+3)
print(np.random.randint(0,10))
print(np.random.randint(0,10))
print(np.random.randint(0,10))


'''
core_rs = np.linspace(1,10,10)
wl0 = 1
k0 = 2*np.pi/wl0
ncore = 1.4504
nclad = 1.4504 - 5.5e-3
i=1

for r in core_rs:
    V = LPmodes.get_V(k0,r,ncore,nclad)
    print(r,V)
    xa = ya = np.linspace(-20,20,201)
    xg,yg = np.meshgrid(xa,ya)
    field = LPmodes.lpfield(xg,yg,0,1,r,wl0,ncore,nclad)
    field = normalize(field)
    cross = field[int((len(field)+1)/2)]
    plt.plot(xa,np.abs(cross)-i/100+1/20,alpha = i/10,color='k')
    #plt.plot(xa,np.abs(cross),color='k',label="lp01 mode")
    #popt,pcov = curve_fit(g,xa,np.abs(cross))
    #plt.plot(xa,g(xa,popt[0],popt[1]),color='steelblue',ls='dashed',label="gaussian")

    i+=1


#plt.legend(loc='best',frameon=False)

plt.xlabel("position (um)")

plt.show()


'''

def plot_tp(fname):
    outs = np.loadtxt(fname)
    p0 = outs[:,0]
    p1 = outs[:,1]
    p2 = outs[:,2]
    p3 = outs[:,3]
    p4 = outs[:,4]
    pt = outs[:,5]

    wl0s = np.linspace(1,2.5,31)

    plt.plot(wl0s,p0,label="central core")
    plt.plot(wl0s,p1+p2+p3+p4,label="outer cores")
    plt.plot(wl0s,p0+p1+p2+p3+p4,label="all cores")
    plt.plot(wl0s,pt,label='total field')

    plt.xlabel("wavelength (microns)")
    plt.ylabel("power")

    plt.legend(loc='best')
    plt.show()

#plot_tp("houtput.txt")

'''
outs = np.loadtxt("houtput.txt")
outs2 = np.loadtxt("houtput2.txt")
outs3 = np.loadtxt("houtput3.txt")

p0 = outs2[:,0]
p1 = outs2[:,1]
p2 = outs2[:,2]
p3 = outs2[:,3]
p4 = outs2[:,4]
pt = outs2[:,5]

_p0 = outs3[:,0]
_p1 = outs3[:,1]
_p2 = outs3[:,2]
_p3 = outs3[:,3]
_p4 = outs3[:,4]
_pt = outs3[:,5]

wl0s = np.linspace(1,2.5,31)

plt.plot(wl0s,p0,label="central")
plt.plot(wl0s,p1+p2+p3+p4,label="outer")
plt.plot(wl0s,p0+p1+p2+p3+p4,label="all")
plt.plot(wl0s,pt,label='total')

plt.legend(loc='best')
plt.show()

plt.plot(wl0s,_p0,label="central")
plt.plot(wl0s,_p1+_p2+_p3+_p4,label="outer")
plt.plot(wl0s,_p0+_p1+_p2+_p3+_p4,label="all")
plt.plot(wl0s,_pt,label="total")

plt.legend(loc='best')
plt.show()
'''

'''
u = np.load("PSF0hi.npy")
fr = read_rsoft("r10_ex.M00")
fr = resize(fr,u.shape)


xaf,yaf = np.linspace(-250,250,u.shape[0]),np.linspace(-250,250,u.shape[1])

idx = bisect_left(xaf,-100)

plt.imshow(np.abs(u))
plt.show()

u = u[idx:-idx,idx:-idx]
fr = fr[idx:-idx,idx:-idx]

print("overlap0",np.abs(np.sum(u*np.conj(fr))))
print("overlap1",np.abs(np.sum(fr*np.conj(fr))))
print("overlap2",np.abs(np.sum(u*np.conj(u))))



plt.imshow(np.abs(u))
plt.show()

#xa,ya = xaf,yaf
xa,ya =  xaf[idx:-idx],yaf[idx:-idx] #np.linspace(-100,100,1001),np.linspace(-100,100,1001)

xg,yg = np.meshgrid(xa,ya)

#lp01 = LPmodes.lpfield(xg,yg,0,1,10,1.55,1.444,1.4431)
lp01 = fr

plt.imshow(np.abs(lp01))
plt.show()
#u = resize(u,(1001,1001))
#print(np.power(overlap(u,fr),2)/overlap(u,u)/overlap(fr,fr))
print(np.power(overlap(u,lp01),2)/)

'''
