# *This* Looks Like *That There*
***
Interpretable prototype convolutional networks inspired by Chen et al. (2019) for when absolute location matters. 
* author: Elizabeth A. Barnes and Randal J. Barnes
***


## References
***
[1] Chen, Chaofan, Oscar Li, Daniel Tao, Alina Barnett, Cynthia Rudin, and Jonathan K. Su. 2019. “This Looks Like That: Deep Learning for Interpretable Image Recognition.” In Advances in Neural Information Processing Systems, edited by H. Wallach, H. Larochelle, A. Beygelzimer, F. d\textquotesingle Alché-Buc, E. Fox, and R. Garnett. Vol. 32. Curran Associates, Inc. https://proceedings.neurips.cc/paper/2019/file/adf7ee2dcf142b0e11888e72b43fcb75-Paper.pdf


## Python Environment
***
The following python environment was used to implement this code.
```
- conda create --name env-tf2.5-cartopy
- conda activate env-tf2.5-cartopy
- conda install anaconda
- pip install tensorflow==2.5 silence-tensorflow memory_profiler  
- conda install -c conda-forge cartopy
- pip uninstall shapely
- pip install --no-binary :all: shapely
- conda install -c conda-forge matplotlib cmocean xarray netCDF4 
- conda install -c cmasher cmocean icecream palettable seaborn
- pip install keras-tuner --upgrade
```
