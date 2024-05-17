#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import illustris_python as il
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binned_statistic_2d
from astropy.cosmology import Planck15 as cosmo
import astropy.units as u
from astropy.constants import c, G
from numpy.random import randint
import scipy
from numpy.fft import fftn, ifftn, fftfreq
import Pygravlens as g

def rotator():
    """
    Line 5 loads the .npy file that contains the rotation matrices. Then, each snapshot (50 to 99) is randomly assigned
    a rotation matrix. This function should only be run once so that all three data slices for each snapshot have the 
    same rotation matrix applied to them.
    """
    allrot = np.load('rotation.npy', allow_pickle = True)
    rotmat = {}
    for i in range(50, 100):
        rotmat[i] = allrot[randint(0, 23)]
    return rotmat

    class dEx:
        def __init__(self, snap, snapclo, snapfar, Sbox_size, particle_type, rotmat):
            """
            snap: Snapshot number
            snapclo: Snapshot number with z = 0.0
            snapfar: Snapshot number with z = 1.0
            particle_type: Three options: (1) Dark Matter, (2) Gas, and (3) Stars
            rotmat: Rotation matrix output from rotator() function.
            This function needs to be run each time the particle type and/or snapshot number is changed.
            """
            self.snap = snap
            self.snapclo = snapclo
            self.snapfar = snapfar
            self.Sbox_size = Sbox_size
            self.particle_type = particle_type
            self.rotmat = rotmat
        def z_restrictor_bounds(self):
            """
            (1) Creates an array of redshifts (i.e., reds) (going from snapshot 49 - 99). This array is specific to the Illustris TNG 100-3 simulation.
            (2) Creates a dictionary (i.e., zsnap), linking each snapshot number with its respective redshift.
            (3) Creates a dictionary (i.e., Lsnap), linking each spapshot number to its associated distance in comoving kpc
            (4) Finds the upper and lower bound for the z-values (will be used to restrict the data)
            """
            reds = [1.04, 1.0,0.95,0.92,0.89,0.85,0.82,0.79,0.76,0.73,0.70,0.68,0.64,0.62,0.60,0.58,0.55,0.52,0.50,0.48,0.46,0.44,0.42,0.40,0.38,0.36,0.35,0.33,0.31,0.30,0.27,0.26,0.24,0.23,0.21,0.20,0.18,0.17,0.15,0.14,0.13,0.11,0.10,0.08,0.07,0.06,0.05,0.03,0.02,0.01,0.00]
            zsnap = {}
            for i in np.arange(49, 100,1):
                zsnap[i] = reds[i - 49]
            Lsnap = {}
            for i in zsnap.keys():
                Lsnap[i] = cosmo.comoving_distance(zsnap[i]).to('kpc').value*cosmo.h
            if self.snap < self.snapclo and self.snap > self.snapfar:
                zplo = 0.5 * (Lsnap[self.snap + 1] - Lsnap[self.snap])
                zphi = 0.5 * (Lsnap[self.snap -1] - Lsnap[self.snap])
            elif self.snap == self.snapclo:
                zplo = 0.5* Lsnap[self.snap]
                zphi = 0.5 * (Lsnap[self.snap - 1] -Lsnap[self.snap])
            elif self.snap == self.snapfar:
                zplo = 0.5 * (Lsnap[self.snap + 1] - Lsnap[self.snap])
                zphi = 0.5 * Lsnap[self.snap]
            zplo += self.Sbox_size/2
            zphi += self.Sbox_size/2
            self.zplo = zplo
            self.zphi = zphi
        def data_extractor(self):
            """
            Extracts the positions and masses of particles for each particle type and snapshot number. This section is
            specific to the Illustris TNG 100-3 simulation.
            """
            header = il.groupcat.loadHeader('../sims.TNG/TNG100-3/output/', self.snap)
            r = self.rotmat[self.snap]
            if self.particle_type == 'dm':
                data = il.snapshot.loadSubset('../sims.TNG/TNG100-3/output/', self.snap, 'dm', fields = ['Coordinates'])
                data = data@r.T
                mass = np.full(len(data),0.0323567549719664) 
                self.coords = data
                self.mass = mass
                self.header = header
            elif self.particle_type == 'stars':
                data = il.snapshot.loadSubset('../sims.TNG/TNG100-3/output/', self.snap, 'stars', fields = ['Coordinates', 'Masses'])
                data['Coordinates'] = data['Coordinates']@r.T
                self.coords = data['Coordinates']
                self.mass = data['Masses']
                self.header = header
            else:
                data = il.snapshot.loadSubset('../sims.TNG/TNG100-3/output/', self.snap, 'gas', fields = ['Coordinates', 'Masses'])
                data['Coordinates'] = data['Coordinates']@r.T
                self.coords = data['Coordinates']
                self.mass = data['Masses']
                self.header = header
        def data_reducer1(self):
            """
            Uses the upper and lower bounds for the z-values obtained from the z_restrictor_bounds() function and chooses
            points in each simulation whose z-values lie within this range. This is to avoid double-counting the same region
            when calculating the mass densities.
            """
            if self.particle_type == 'dm':
                if np.mean(self.coords[:, 2]) < 0:
                    self.flag1 = (self.coords[:,2]+self.Sbox_size>=self.zplo)
                    self.flag2 = (self.coords[:,2]+self.Sbox_size<=self.zphi)
                    self.restricted_data = self.coords[self.flag1*self.flag2]
                    self.mass_rest = np.full(len(self.restricted_data),0.0323567549719664) 
                else:
                    self.flag1 = (self.coords[:,2]>=self.zplo)
                    self.flag2 = (self.coords[:,2]<=self.zphi)
                    self.restricted_data = self.coords[self.flag1*self.flag2]
                    self.mass_rest = np.full(len(self.restricted_data),0.0323567549719664)
                self.x_rest =self.restricted_data[:, 0]
                self.y_rest =self.restricted_data[:, 1] 
            else:
                if np.mean(self.coords[:, 2]) < 0:
                    self.flag1 = (self.coords[:, 2] + self.Sbox_size >= self.zplo)
                    self.flag2 = (self.coords[:, 2] + self.Sbox_size <= self.zphi)
                    self.restricted_data = self.coords[self.flag1 * self.flag2]
                    self.mass_rest = self.mass[:len(self.restricted_data):]
                    self.x_rest =self.restricted_data[:, 0]
                    self.y_rest =self.restricted_data[:, 1]
                else:
                    self.flag1 = (self.coords[:, 2]>= self.zplo)
                    self.flag2 = (self.coords[:, 2]<= self.zphi)
                    self.restricted_data = self.coords[self.flag1 * self.flag2]
                    self.mass_rest = self.mass[:len(self.restricted_data):]
                    self.x_rest =self.restricted_data[:, 0]
                    self.y_rest =self.restricted_data[:, 1] 
        def hist(self):
            """
            Calculates the mass density for each snapshot, saves the density and the x-, y-centers to be used in the 
            Fourier Analysis. 
            """
            centers = {}
            grid, xedge, yedge, _ = binned_statistic_2d(self.x_rest, self.y_rest,self.mass_rest, 'sum', bins=[600, 600], range=[[np.min(self.x_rest),np.max(self.x_rest)],[np.min(self.y_rest),np.max(self.y_rest)]])
            xcen = (xedge[:-1]+xedge[1:])/2
            ycen = (yedge[:-1]+yedge[1:])/2
    
            pxSize = self.header['BoxSize'] / 600 # code units
            pxSize_kpc = pxSize * self.header['Time'] / self.header['HubbleParam']
            pxArea = pxSize_kpc**2
    
            grid_log_msun_per_kpc2 = grid * 1e10 / self.header['HubbleParam'] / pxArea
            if self.particle_type == 'dm':
                np.save(f'density_{self.snap}_dm.npy', grid_log_msun_per_kpc2)
            elif self.particle_type == 'stars':
                np.save(f'density_{self.snap}_stars.npy', grid_log_msun_per_kpc2)
            else:
                np.save(f'density_{self.snap}_gas.npy', grid_log_msun_per_kpc2)
            del grid_log_msun_per_kpc2
            if self.particle_type == 'dm':
                centers['x'] = xcen
                centers['y'] = ycen
                np.save(f'center_{self.snap}.npy', centers)
            del xcen, ycen 
class m_to_conv:
    def __init__(self, snap):
        """
        snap: snapshot number array
        reds: array of redshifts (specific to IllustrisTNG 100-3 simulation)
        """
        self.snap = snap
        self.reds = [1.00, 0.95, 0.92, 0.89, 0.85, 0.82, 0.79, 0.76, 0.73, 0.70, 0.68, 0.64, 0.62, 0.60, 0.58, 0.55, 0.52, 0.50, 0.48, 0.46, 0.44, 0.42, 0.40, 0.38, 0.36, 0.35, 0.33, 0.31, 0.30, 0.27, 0.26, 0.24, 0.23, 0.21, 0.20, 0.18, 0.17,0.15, 0.14, 0.13, 0.11, 0.10, 0.08, 0.07, 0.06, 0.05, 0.03, 0.02, 0.01, 0.0]
        sr = {}
        for i in np.arange(50, 100, 1):
            sr[i] = self.reds[i - 50]
        self.sr = sr
    def mtok(density_map, zlens, zsrc = 2.0):
        """
        User inputs mass density map (for dark matter, gas, or stars), the redshift of the lens, and the redshift of the source. 
        The default setting on the source redshift is 2.0. dL is the distance to the lens, dS is the distance to the source, dLS is the
        distance between the source and the lens. crit is the critical mass density for the lens. 
        """
        dL = cosmo.angular_diameter_distance(zlens)
        dS = cosmo.angular_diameter_distance(zsrc)
        dLS = cosmo.angular_diameter_distance_z1z2(zlens,zsrc)
        #compute critical density and convert to Msun per kpc^2
        crit = ((c)**2 /(4*np.pi*G) *dS/(dL*dLS)).to(u.Msun/(u.kpc)**2)
        #and convert to Msun per arcsec^2
        dist = dL.to(u.kiloparsec)
        crit = (crit * dist**2 /u.rad**2).to(u.Msun/u.arcsec**2)
        print(f'Critical density: {crit:.2e}')

        kappa = density_map/crit.value
        Crit = crit.value
        return kappa, Crit
    def dosnap2(self):
        """
        Performs the lensing calculations (calculates and saves the density, deflection angle, single- and second-order (including mixed)
        derivatives of the lensing potential, and kappa maps. This version also saves additional information about the angular scale of the 
        snapshot along with the redshift, critical density, and pixel scale in a dictionary for use with pygravlens multi-plane lensing modelling.
        """
        dens_maps = {}
        def_maps = {}
        phix_maps = {}
        phiy_maps = {}
        phi_maps = {}
        phixy_maps = {}
        phixx_maps = {}
        phiyy_maps = {}
        kappa_maps = {}
        data = {}
        ang_scale_per_ckpc = {}
        ang_scale_per_arcsec = {}
        redshift = {} 
        critical_density = {}
        PixSc = {}

        for i in range(len(self.snap)):
            angle_per_ckpc = cosmo.arcsec_per_kpc_comoving(self.sr[self.snap[i]]).value
            print(f'Snapshot {self.snap[i]}, z = {self.sr[self.snap[i]]}, angular scale = {angle_per_ckpc:.3f} arcsec/ckpc or {1/angle_per_ckpc:.2f} ckpc/arcsec')
            Redshift = self.sr[self.snap[i]]
            ang_scale_per_arcsec[self.snap[i]] = 1/angle_per_ckpc
            ang_scale_per_ckpc[self.snap[i]] = angle_per_ckpc
            redshift[self.snap[i]] = Redshift


            # box size, in arcsec
            Lbox_arcsec = 75000/cosmo.h * angle_per_ckpc

            # density maps - dm
            dens_map_dm = np.load('density_'+str(self.snap[i])+'_dm.npy', allow_pickle = 'TRUE')
            dens_map_dm /= angle_per_ckpc**2
            # stars
            dens_map_stars = np.load('density_'+str(self.snap[i])+'_stars.npy', allow_pickle = 'TRUE')
            dens_map_stars /=angle_per_ckpc**2
            # gas
            dens_map_gas = np.load('density_'+str(self.snap[i])+'_gas.npy', allow_pickle = 'TRUE')
            dens_map_gas /=angle_per_ckpc**2
            # total
            global comb_dens_map
            comb_dens_map = dens_map_dm + dens_map_stars + dens_map_gas

            dens_maps[self.snap[i]] = comb_dens_map

            # number of pixels
            npix = comb_dens_map.shape[0]
            # pixel scale
            pixel_scale = Lbox_arcsec/npix
            print(f'Pixel scale = {pixel_scale:.2f} arcsec/pix')
            # x and y arrays (1d)

            # calculate the kappa map
            kappa_comb, crit = m_to_conv.mtok(comb_dens_map, self.sr[self.snap[i]])
            # difference relative to mean
            kappa_mean = np.mean(kappa_comb)
            dkappa = kappa_comb - kappa_mean
            print('Mean kappa =',kappa_mean)
            # Load the x-, and y-centers
            xcen = np.load(f'center_{self.snap[i]}.npy', allow_pickle = True).item()['x']
            ycen = np.load(f'center_{self.snap[i]}.npy', allow_pickle = True).item()['y']
            # calculate lensing quantities
            phi,phix,phiy,phixx,phiyy,phixy = g.kappa2lens([i/cosmo.h * angle_per_ckpc for i in xcen],[i/cosmo.h * angle_per_ckpc for i in ycen],dkappa)
            defnorm = np.sqrt(phix**2+phiy**2)

            def_maps[self.snap[i]] = defnorm
            phi_maps[self.snap[i]] = phi
            phix_maps[self.snap[i]] = phix
            phiy_maps[self.snap[i]] = phiy
            phixy_maps[self.snap[i]] = phixy
            phiyy_maps[self.snap[i]] = phiyy
            phixx_maps[self.snap[i]] = phixx
            kappa_maps[self.snap[i]] = dkappa
            critical_density[self.snap[i]] = crit
            PixSc[self.snap[i]] = pixel_scale
        
        data = {"Angular Scale (asec/ckpc)": ang_scale_per_ckpc, "Angular Scale (ckpc/asec)": ang_scale_per_arcsec, "Redshift": redshift, "Critical Density (solMass/asec^2)": critical_density, "Pixel Scale": PixSc}
        if np.size(self.snap) > 1:
            np.save(f'dens_maps_dictionary.npy', dens_maps)
            np.save(f'def_maps_dictionary.npy', def_maps)
            np.save(f'phi_maps_dictionary.npy', phi_maps)
            np.save(f'phix_maps_dictionary.npy', phix_maps)
            np.save(f'phiy_maps_dictionary.npy', phiy_maps)
            np.save(f'phixy_maps_dictionary.npy', phixy_maps)
            np.save(f'kappa_maps_dictionary.npy', kappa_maps)
            np.save(f'phiyy_maps_dictionary.npy', phiyy_maps)
            np.save(f'phixx_maps_dictionary.npy', phixx_maps)
            np.save(f'data_dictionary.npy', data)
        else:
            np.save(f'dens_maps_dictionary_{self.snap}.npy', dens_maps)
            np.save(f'def_maps_dictionary_{self.snap}.npy', def_maps)
            np.save(f'phi_maps_dictionary_{self.snap}.npy', phi_maps)
            np.save(f'phix_maps_dictionary_{self.snap}.npy', phix_maps)
            np.save(f'phiy_maps_dictionary_{self.snap}.npy', phiy_maps)
            np.save(f'phixy_maps_dictionary_{self.snap}.npy', phixy_maps)
            np.save(f'kappa_maps_dictionary_{self.snap}.npy', kappa_maps)
            np.save(f'phiyy_maps_dictionary_{self.snap}.npy', phiyy_maps)
            np.save(f'phixx_maps_dictionary_{self.snap}.npy', phixx_maps)
            np.save(f'data_dictionary_{self.snap}.npy', data)