import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import corner
import pygravlens as gl
import pygravlens_shifts as gls
from astropy.cosmology import Planck15 as cosmo
from scipy.interpolate import RectBivariateSpline
from astropy import units as u
import matplotlib
import pickle

plt.rcParams.update({'font.size': 16, 'text.usetex': True, "font.family": "Times New Roman"})

class lens:
    '''
    Sets up multiplane lensmodel environment.
    '''
    def __init__(self,DAT_filename, kap_fidir, boxsize):
        '''
        DAT_filename: File Directory for data_dictionary
        kap_fidir: File Directory for kappa maps
        boxsize: Simulation Boxsize (Mpc/h)
        '''
        self.DAT_filename = DAT_filename
        self.kap_fidir = kap_fidir
        self.boxsize = boxsize
        self.SIMDATA = np.load(self.DAT_filename, allow_pickle = True).item()
    def lensmodel(self, sni, snf, fov, output = False, info = False, Npix = 200):
        '''
        sni: Initial Snapshot #
        snf: Final Snapshot #
        fov: Field of View (Degrees)
        output: Determines output. If false, only model and mybox are returned. Else, deflection array and
                kappa array are also returned. 
        info: Reports information about each snapshot. If info = True, all info is printed. Otherwise, the
              info is retained.
        Npix: Number of pixels for box.
        '''
        allplanes = []
        if info == True:
            for i in range(sni,snf + 1):
                # put them in the gl structure
                zl = self.SIMDATA['Redshift'][i]
                Dl = cosmo.comoving_distance(zl)
                boxsize = cosmo.arcsec_per_kpc_comoving(zl) * self.boxsize*1000*u.kpc*(1/cosmo.h)
                print(i,zl,Dl,'boxsize',boxsize)
                plane = gls.lensplane('kapmap', self.kap_fidir+str(i), map_bound='periodic', Dl=Dl)
                allplanes.append(plane)
        else:
            for i in range(sni,snf + 1):
                # put them in the gl structure
                zl = self.SIMDATA['Redshift'][i]
                Dl = cosmo.comoving_distance(zl)
                boxsize = cosmo.arcsec_per_kpc_comoving(zl) * self.boxsize*1000*u.kpc*(1/cosmo.h)
                plane = gls.lensplane('kapmap', self.kap_fidir+str(i), map_bound='periodic', Dl=Dl)
                allplanes.append(plane)
                
        # we use a dummy source distance, since Ds will be overridden with actual values in calculations
        Ds = cosmo.comoving_distance(2.0)

        # combine the planes into a model; here we need to specify the comoving distance to the source;
        # also, position_mode indicates that the planes are fixed in space
        model = gls.lensmodel(allplanes, Ds=Ds, Dref=np.inf, Ddecimals=8, position_mode='fix')
        model.info()
        
        Lbox = fov * 3600
        xtmp = np.arange(Npix)*Lbox/Npix - Lbox/2
        mybox = [xtmp[0],xtmp[-1],xtmp[0],xtmp[-1]]
        xarr = gls.mygrid(xtmp,xtmp)
        defarr,Aarr = model.defmag(xarr)
        defnorm = np.sqrt(defarr[:,:,0]**2+defarr[:,:,1]**2)
        kaparr = 1 - 0.5*(Aarr[:,:,0,0]+Aarr[:,:,1,1])
        print('means: kappa',np.mean(kaparr),'deflection',np.mean(defarr[:,:,0]),np.mean(defarr[:,:,1]))
        
        if output == True:
            return model, mybox, defarr, kaparr
        else:
            return model, mybox


class clusterclass:

    ############################################################
    # load image data and do a lot of setup
    ############################################################

    def __init__(self,filename, zclus,clusname=''):

        # cluster redshift and distance
        self.zclus = zclus
        self.Dclus = cosmo.comoving_distance(self.zclus)

        # cluster name, which will be used for plotting if set
        self.clusname = clusname

        # image data
        self.imgdat = np.loadtxt(filename)
        self.ximg = self.imgdat[:,:2]
        self.zimg = self.imgdat[:,2]
        self.Dimg = cosmo.comoving_distance(self.zimg)
        self.nimg = len(self.ximg)

        # use the midpoint as the reference point
        xlo = np.min(self.ximg,axis=0)
        xhi = np.max(self.ximg,axis=0)
        self.xmid = 0.5*(xlo+xhi)
        # prepend the midpoint in arrays used for calculations; put it at Dclus
        self.xcalc = np.insert(self.ximg,0,self.xmid,axis=0)
        self.Dcalc = np.insert(self.Dimg,0,self.Dclus,axis=0)

        # store separations (from midpoint) and redshifts in arrays that are
        # aligned with structure of deflection statistics
        dtmp = np.linalg.norm(self.ximg-self.xmid,axis=1)
        self.dplot = np.column_stack((dtmp,dtmp+0.25)).flatten()
        self.zplot = np.column_stack((self.zimg,self.zimg)).flatten()
        # color points by redshift
        self.color_norm = matplotlib.colors.Normalize(vmin=np.amin(self.zplot),vmax=np.amax(self.zplot))
        self.color_map = matplotlib.cm.get_cmap('viridis')
        self.color_array = self.color_map(self.color_norm(self.zplot))

        # labels for plotting, again aligned with structure of deflection statistics
        self.labels = []
        for i in range(len(self.ximg)):
            self.labels.append('x'+str(i))
            self.labels.append('y'+str(i))
            

        # information about all pairwise separations
        # (currently this isn't used, but it might be interesting at some point)
        self.allpairs = []
        for i in range(self.nimg):
            for j in range(i+1,self.nimg):
                dr = np.linalg.norm(self.ximg[i]-self.ximg[j])
                dz = np.abs(self.zimg[i]-self.zimg[j])
                self.allpairs.append([dr,dz])
        self.allpairs = np.array(self.allpairs)
        
        # plot image configuration
        plt.figure(figsize=(6,6))
        plt.scatter(self.ximg[:,0],self.ximg[:,1],marker='x',c=self.color_map(self.color_norm(self.zimg)))
        plt.plot(self.xmid[0],self.xmid[1],'+',color='red')
        plt.colorbar(matplotlib.cm.ScalarMappable(norm=self.color_norm,cmap=self.color_map),label='source redshift',shrink=0.6)
        plt.gca().set_aspect('equal')
        plt.xlabel('x (arcsec)')
        plt.ylabel('y (arcsec)')
        plt.title(self.clusname)
        plt.savefig(f'{self.clusname}_image_pos.png', bbox_inches = 'tight')
        plt.show()
        
    ############################################################
    # do deflection statistics calculation, and process results
    ############################################################
    
    def DefStats(self,model,Nsamp=10000, box = []):

        if len(box)<4:
            print('Error: box must be specified in DefStats')
            return

        # compute the full arrays; recall that this uses the midpoint as the
        # reference, but it is omitted from the output arrays
        print('Computing deflection statistics',self.clusname)
        self.defavg_full,self.defcov_full,self.kapgam = model.DefStats(self.xcalc,
            extent=box,
            Nsamp=Nsamp,
            refimg=0,
            fitshear=True,
            Dlnew=self.Dclus,
            Dsnew=self.Dcalc)
        
        # separate into los and fg pieces
        half = 2*self.nimg
        self.defavg_los = self.defavg_full[:half]
        self.defavg_fg  = self.defavg_full[half:]
        self.defcov_los    = self.defcov_full[:half,:half]
        self.defcov_fg     = self.defcov_full[half:,half:]
        self.defcov_los_fg = self.defcov_full[:half,half:]

        # standard deviations
        self.defsig_los = np.sqrt(np.diag(self.defcov_los))
        self.defsig_fg  = np.sqrt(np.diag(self.defcov_fg ))

        # statistics for all pairs
        # (currently this isn't used, but it might be interesting at some point)
        self.allpairs_los = []
        for i in range(self.nimg):
            for j in range(i+1,self.nimg):
                xmu = self.defavg_los[2*i  ] - self.defavg_los[2*j  ]
                ymu = self.defavg_los[2*i+1] - self.defavg_los[2*j+1]
                xsg = np.sqrt( self.defcov_los[2*i  ,2*i  ] + self.defcov_los[2*j  ,2*j  ] - 2.0*self.defcov_los[2*i  ,2*j  ] )
                ysg = np.sqrt( self.defcov_los[2*i+1,2*i+1] + self.defcov_los[2*j+1,2*j+1] - 2.0*self.defcov_los[2*i+1,2*j+1] )
                self.allpairs_los.append([xmu,xsg,ymu,ysg])
        self.allpairs_los = np.array(self.allpairs_los)
        
        self.rms_los = np.sqrt(np.sum(np.diag(self.defcov_los))/self.nimg)
        
        # report
        print('los: $\langle \mu \\rangle = {:.3e}$ ; $\langle \sigma \\rangle = {:.3e}$ ; RMS = {:.3e}'.format(np.mean(self.defavg_los), np.mean(self.defsig_los), self.rms_los))
        print('fg:  $\langle \mu \\rangle$ = {:.3e}  $\langle \sigma \\rangle$ = {:.3e}'.format(np.mean(self.defavg_fg ),np.mean(self.defsig_fg)))
        
        info = {'LOS': [self.defavg_los, self.defcov_los], 'FG': [self.defavg_fg, self.defcov_fg], 'LOSFG': [self.defcov_los_fg]}
        return info 

    ############################################################
    # plot deflection statistics
    ############################################################

    def plot_DefStats(self):
        f,ax = plt.subplots(2,1,figsize=(12,10))
        # los
        ax[0].axhline(0,color='gray',linestyle='dotted')
        ax[0].errorbar(self.dplot,self.defavg_los,yerr=self.defsig_los,ecolor=self.color_array,alpha=0.7,fmt=',')
        ax[0].scatter(self.dplot[0::2],self.defavg_los[0::2],marker='x',c=self.color_array[0::2])
        ax[0].scatter(self.dplot[1::2],self.defavg_los[1::2],marker='1',c=self.color_array[1::2])
        # ax[0].set_xlabel('image separation (arcsec)')  # omit because panels share x-axis
        ax[0].set_ylabel('Deflection $\mu, \sigma$ (arcsec)')
        if len(self.clusname)>0:
            ax[0].set_title(self.clusname+': Full Line of Sight')
        else:
            ax[0].set_title('Full Line of Sight')
        # fg
        ax[1].axhline(0,color='gray',linestyle='dotted')
        ax[1].errorbar(self.dplot,self.defavg_fg,yerr=self.defsig_fg,ecolor=self.color_array,alpha=0.7,fmt=',')
        ax[1].scatter(self.dplot[0::2],self.defavg_fg[0::2],marker='x',c=self.color_array[0::2])
        ax[1].scatter(self.dplot[1::2],self.defavg_fg[1::2],marker='1',c=self.color_array[1::2])
        ax[1].set_xlabel('Distance from Midpoint (arcsec)')
        ax[1].set_ylabel('Deflection $\mu,\sigma$ (arcsec)')
        ax[1].set_title('Foreground')
        #
        f.colorbar(matplotlib.cm.ScalarMappable(norm=self.color_norm,cmap=self.color_map),label='source redshift',ax=ax,shrink=0.6)
        plt.savefig(f'{self.clusname}_defstats.png', bbox_inches = 'tight')
        f.show()