# RU Gravitational Lensing Data Repository 

This repository contains key software and data used in our analysis of line-of-sight effects of cosmological large-scale structures on cluster lens models

---

## Required Packages/Libraries: 
* illustris_python (Downloading snapshot data)
* numpy
* scipy
* astropy
* matplotlib
* pygravlens (Download here: https://github.com/chuckkeeton/pygravlens)
* shapely

## Available Products:
*   **data_ext.py:** Enables users to download snapshot data in the IllustrisTNG workspace. This code assigns the random rotations to each snapshot, and places upper and lower limits on the particles' $z$-coordinates. To use this code, run the following commands in the
     IllustrisTNG Workspace:

        import numpy as np
        import data_ext as dext

        rotmat = dext.rotator() # Rotates each snapshot. It is important that this code is only run once so that snapshots of the same number get the same rotation matrix applied.

        d = dext.dEx(snap, snapclo, snapfar, sbox_size, particle_type, rotmat) # Make sure to fill in the individual inputs as needed. 
        d.z_restrictor_bounds() # Compute upper and lower limits to place on particles
        d.data_extractor() # Extract the particles
        d.data_reducer1() # Apply the limits to the particles
        d.hist() # Compute the 2D histogram weighted by mass
    After running this code, we will have three mass density files per redshift (one each for stars, gas, and dark matter). We add these files to compute the total mass density map for a redshift. Download these onto a loczl machine.
* **conv_comp.py:** Computes lensing properties (convergence, shear, magnification) for provided mass density maps. This file requires the pygravlens package to run succesfully. Use the following code to run this file:

      import conv_comp as cc
            
      mtoconv = cc.m_to_conv(snap)  #  snap: array of snapshot numbers
      mtoconv.dosnap2()
* **clensmod.py:** Computes deflection statistics for each cluster (in our analysis, we used this package for Abell 2744 and MACS J0416.1-2403). This file contains two classes: (i) lens, and (ii) clusterclass. The first class sets up the multiplane lensmodel containing all snapshots. The next class computes deflection statistics for a galaxy cluster. To set up the lensmodel, use the following code: 

      import clensmod as cl
            
      lensmodel = cl.lens(data_dictionary_file, kappa_file_directory, box_size)
      model, mybox = lensmodel.lensmodel(start_snap, end_snap, fov)

    data_dictionary_file indicates the location where the data_dictionary (output from conv_comp) is stored. kappa_file_directory is the directory to where the kappa files (output from conv_comp) is stored. box_size is the simulation box size, expressed in Mpc/h. start_snap is the starting snapshot number, and end_snap is the ending snapshot number. fov is the field-of-view in degrees. If we want the code to also output the deflection array and kappa arrays, we pass an additional argument, "output = True". The default output is the multiplane lensmodel, and lensmodel box. Once we run these code, we can compute deflection statistics for a cluster as follows:

      cluster = cl.clusterclass(cluster_image_file, z_clus, clusname)
      info = cluster.DefStats(model,Nsamp= 10000, box = mybox)
      cluster.plot_DefStats()

    The cluster_image_file is the .txt file containing image positions, z_clus is the cluster redshift, and clusname is the name of the cluster (e.g., A2744, M0416). info is a dictionary containing the covariance matrices:

      info = {'kapgam': [kapgam], 'LOS': [defavg_los, defcov_los], 'FG': [defavg_fg, defcov_fg], 'LOSFG': [defcov_los_fg]}

    cluster.plot_DefStats() plots the actual deflection statistics.
* **data_dictionary.npy:** Dictionary containing information (i.e., angular scale (asec/ckpc), angular scale (ckpc/asec), redshift , critical density (solMass/asec^2), pixel scale) for each snapshot.
* **lens_stats_2744.npy:** numpy file containing shears ($\gamma_{c}$, $\gamma_{s}$) and convergence ($\kappa$) distributions for each snapshot, for Abell 2744.
* **lens_stats_0416.npy:** numpy file containing shears ($\gamma_{c}$, $\gamma_{s}$) and convergence ($\kappa$) distributions for each snapshot, for MACS J0416.1-2403.
* **fits_2744.npy:** numpy file containing distribution fit information (normal fit for shears; log-normal for convergence) for Abell 2744.
* **fits_0416.npy:** numpy file containing distribution fit information (normal fit for shears; log-normal for convergence) for MACS J0416.1-2403.

## Available Tutorials: 
* **Example.ipynb:** Tutorial on downloading simulation snapshots from the IllustrisTNG workspace. 
