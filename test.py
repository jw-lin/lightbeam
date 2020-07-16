import numpy as np
import h5py

f = h5py.File("mytestfile.hdf5", "w")
f.create_dataset("value",data = 1)
f.close()

f = h5py.File("mytestfile.hdf5",'r')
a = f["value"][()]
print(a)
f.close()

