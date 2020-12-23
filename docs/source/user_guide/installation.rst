.. _installation:

Install
=======

You can install myGym by standard procedure:

``git clone https://github.com/incognite/mygym.git``

``cd myGym``

We reccomend to create conda environment:

``conda env create -f environment.yml``

``conda activate mygym``

Install myGym:

``python setup.py develop``

Pretrained modules
------------------

If you want to use pretrained visual modules, please download them
first:

``cd myGym`` ``sh download_vision.sh``

If you want to use pretrained baselines models, download them here:

``cd myGym`` ``sh download_baselines.sh``

Supported systems
-----------------

Ubuntu 18.04, 20.04 Python 3.5, 3.6.,3.7,3.8 GPU acceleration strongly
supported