Author: Arndt Kaps (arndt.kaps@dlr.de)  
Adaptation Mateo-Garcia et al (2022) code (https://github.com/IPL-UV/pvl8dagans) for ESACCI/ICON domain adaptation with Wasserstein Cyclegans  
The results produced with this code are part of the doctoral dissertation of Arndt Kaps (citation will go here)  
The provided scripts train neural networks to transfer between the domains of climate models and observations and produce plots to quantify the quality of the results.  

To run the scripts, install the required packages with the following command:  
```
conda create -n DA python=3.10 matplotlib=3.5 pytorch netcdf4 h5py scikit-learn=1.1 dask distributed -y
conda activate DA
pip install --extra-index-url https://pypi.nvidia.com tensorrt-bindings==8.6.1 tensorrt-libs==8.6.1
pip install -U tensorflow[and-cuda]==2.11
pip install tqdm cartopy seaborn prefetch-generator luigi ray==2.3 pyarrow pot tensorflow_addons
pip install ray[tune]
```

## Purpose of scripts  
`train_cycle_gans_da.py`:  
main script. manages hyperparameters, defines models and datasets calls the training and plotting functions. The figure below shows the results of the application of generator on a single sample (Fig. 6.4 in the dissertation).  
<img src="https://github.com/EyringMLClimateGroup/kaps23_domain_adaptation/blob/main/figures/fig4.png" width="500" />  

`tune_dagan.py`:  
does the HPO using ray-tune  

`distance.py`:  
computes metrics for the final models. requires joint dataframes produced by the main function in `dagans/apply_to_icon.py`. The figure below shows the improvement of the distance metrics for all 2D joint distributions(Fig 6.3 in the dissertation)    
<img src="https://github.com/EyringMLClimateGroup/kaps23_domain_adaptation/blob/main/figures/fig3.png" width="500" />

`quantidiff.py`:  
compares the changes in all variables induced by the WGAN-DA to the statistical difference between ICON and ESACCI  

`toa_model.py`:  
trains the NN used to predict CREs and cloud cover. uses `dagans/models_toa.py` for definition of models and forward passes  

`mk_fakeICON_df.py` converts the synthetic observations returned by `apply_to_icon.py` as npz into a `.parquet` dataframe  

`diffESA_fake.py` compares the cloud distributions obtained from synthetic observations to CCClim  

`dagans/dataloader_pt.py`:
creates the hdf5 datasets used here from npz files

`cosp_vs_native.py`:  
compares our DA to that of COSP. Below is shown the comparison of the cloud cover changes caused by both methods (Fig. 6.5)  
<img src="https://github.com/EyringMLClimateGroup/kaps23_domain_adaptation/blob/main/figures/fig5.png" width="500" />  

`ESACCI_regional.py` does similar plots but in the style of the plots from Chapter 4.