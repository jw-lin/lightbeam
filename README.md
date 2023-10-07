# lightbeam
Simulate light through weakly guiding waveguides using the finite-differences beam propagation method on an adaptive mesh.

## installation
Use pip: `pip install git+https://github.com/jw-lin/lightbeam.git`

Python dependencies: `numpy`,`scipy`,`matplotlib`,`numba`,`numexpr`,`jupyter`

## getting started
Check out the Python notebook in the `tutorial` folder for a quickstart guide. <a href="tutorial/Lightbeam.ipynb">Direct link.</a>

## references
J. Shibayama, K. Matsubara, M. Sekiguchi, J. Yamauchi and H. Nakano, "Efficient nonuniform schemes for paraxial and wide-angle finite-difference beam propagation methods," in Journal of Lightwave Technology, vol. 17, no. 4, pp. 677-683, April 1999, <a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=754799">doi: 10.1109/50.754799.</a> 

The Beam-Propagation Method. (2015). In Beam Propagation Method for Design of Optical Waveguide Devices (pp. 22–70). <a href="https://onlinelibrary.wiley.com/doi/book/10.1002/9781119083405">  doi:10.1002/9781119083405.ch2</a>

(I'm also working on a faster Julia version, stay tuned ...)
