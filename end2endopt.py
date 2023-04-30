import sigproc as sgp
import meepadj as mpj
import autograd.numpy as np
from autograd import grad
import scipy.ndimage as nd
import n2f
import nlopt
import meep as mp
from mpi4py import MPI
comm = MPI.COMM_WORLD

nfreqs=2
lamda_um_min=1.53
lamda_um_max=1.61
freqs=np.linspace(lamda_um_min/lamda_um_max,1.0,nfreqs)

lamda=1./np.max(freqs)
du=0.05
npix=int(lamda/du)
npml=int(npix/2)
nx=200
ny=200
nrep=10

nsens = int(40)
nintg = int(3*npix)
nmask=int(npix/2)

alpha = 10.

nlay=int(1)
pml2struct=npix
mz=np.array([40],dtype=int)
gap=[]
substrate=[]
struct2pml=npix
nz=npml+pml2struct+np.sum(mz)+int(np.sum(gap))+int(np.sum(substrate))+struct2pml+npml

z0=np.array([npml+pml2struct],dtype=int)
for i in range(1,nlay):
     z0.append(z0[i-1]+mz[i-1]+gap[i-1]+substrate[i-1])

eps_ipdip=2.341
eps_vac=1.
epsbkg=np.ones((nx,ny,nz))
epsbkg[:,:,0:z0[0]]=eps_ipdip
for i in range(1,nlay):
     epsbkg[:,:,z0[i]-substrate[i-1]:z0[i]]=eps_ipdip
epsdiff=(eps_ipdip-eps_vac) * np.ones(nlay)

fgs=[]
Dz=30.
for ifreq in range(nfreqs):
     fgs.append(n2f.greens(nsens*nintg+nx*nrep,nsens*nintg+ny*nrep,du,du, freqs[ifreq], 1.,1., Dz))

print("computational domain: {} x {} x {} pixels or {} x {} x {} wavlength^3 AND du: {}".format(nx,ny,nz,nx*du,ny*du,nz*du,du))
print("pml size: {} pixels or {} wavelengths".format(npml,npml*du))
print("device size {} x {} pixels OR {} x {} wavelengths, with {} layers with thicknesses {} pixels".format(nx*nrep,ny*nrep,(nx*nrep)*du,(ny*nrep)*du,nlay,mz))
print("the device layer(s) starting at {}-th pixel".format(z0))
print("screen nsens: {} pixels, nintg: {} pixels, total size: {} wavelengths, pixel width: {} wavelengths".format(nsens,nintg,nsens*nintg*du,nintg*du))
print("padding to the nearfield: {} pixels".format(nsens*nintg/2.))
print("{} freqs: {}, wavelengths (um): {}".format(nfreqs,freqs,lamda_um_min/freqs))
print("Distance between nearfield monitor and screen: {} wavelengths".format(Dz))

#make sure nx, ny and nz are even
if (nx%2) != 0:
     print('WARNING: nx is not even.\n')
if (ny%2) != 0:
     print('WARNING: ny is not even.\n')
if (nz%2) != 0:
     print('WARNING: nz is not even.\n')
     
srcz=npml+1
mtrz=nz-npml-1

sig=1.
df=0.1
beta=60.
courant=0.5

########

dofsize=nx*ny*nlay
objfunc = lambda dof, gdat: mpj.mse(dof,gdat,
                                    nx,ny,nz,du,npml,
                                    nlay,mz,z0, sig, beta,
                                    epsbkg, epsdiff,
                                    srcz, mtrz,
                                    freqs, df, courant,
                                    nrep,fgs,nsens,nintg,alpha)
Job=1

if Job==0:

     np.random.seed(1234)

     ndat=100
     dp=0.0001

     tmp=np.zeros(dofsize)
     gdat=np.zeros(dofsize)
     for i in range(ndat):
          if comm.rank==0:
               hdof=[]
               for ilay in range(nlay):
                    hdof.append( np.random.uniform(low=0.01,high=0.99,size=(nx,ny)) + 0.01 )
               dof=mpj.hdof2dof(hdof,nlay)
               chk_ilay=np.random.randint(low=0,high=nlay)
               chk_ix=np.random.randint(low=0,high=nx)
               chk_iy=np.random.randint(low=0,high=ny)
               chk = chk_ix + nx * chk_iy + nx*ny * chk_ilay
          else:
               dof=None
               chk=None
          comm.Barrier()
          dof=comm.bcast(dof,root=0)
          chk=comm.bcast(chk,root=0)

          obj=objfunc(dof,gdat)
          adj=gdat[chk]
     
          tmp[:]=dof[:]
          tmp[chk]-=dp
          obj0=objfunc(tmp,np.array([]))

          tmp[:]=dof[:]
          tmp[chk]+=dp
          obj1=objfunc(tmp,np.array([]))

          cendiff=(obj1-obj0)/(2.*dp)
          
          print("check: {} {} {} {}".format(adj,cendiff,(adj-cendiff)/adj,obj))
     
if Job==1:

     print('Optimization Job [1] is chosen.')

     maxeval=1000
     init=2
     
     dof0=[]
     hdof0 = []
     for i in range(nlay):
          if init==0:
               hdof0.append( 0.5*np.ones((nx,ny)) )
               #hdof0.append( np.random.uniform(low=0.01,high=0.99,size=(nx,ny)) + 0.01 )
          if init==1:
               np.random.seed(1234)
               tmp = nd.gaussian_filter(np.random.uniform(low=0.0,high=1.0,size=(nx,ny)),sigma=2.0)
               nd.grey_closing(tmp, size=(5,5))
               nd.grey_opening(tmp, size=(5,5))
               tmp[tmp>0.5]=int(1)
               tmp[tmp<0.5]=int(0)
               #tmp = nd.gaussian_filter(tmp,sigma=5.0)
               hdof0.append( 0.98*tmp + 0.01 )
               if comm.rank==0:
                    from matplotlib.image import imsave
                    imsave('initrand_layer{}.png'.format(i),hdof0[i])
               comm.Barrier()

     if init==0 or init==1:
          dof0=mpj.hdof2dof(hdof0, nlay)
     if init==2:
          dof0=np.loadtxt('opt4.txt')

     hub, hlb = [], []
     for i in range(nlay):
          hub.append( 1.00*np.ones((nx,ny)) )
          hlb.append( 0.01*np.ones((nx,ny)) )
     ub=mpj.hdof2dof(hub, nlay)
     lb=mpj.hdof2dof(hlb, nlay)

     opt = nlopt.opt(nlopt.LD_MMA, dofsize)
     opt.set_min_objective(objfunc)
     opt.set_lower_bounds(lb)
     opt.set_upper_bounds(ub)
     opt.set_ftol_rel(1e-8)
     opt.set_maxeval(maxeval)
     xopt = opt.optimize(dof0)
     
     print('Optimization returns {}'.format(xopt))
     print('Optimization Done!')
                                                                                     

if Job==2:

     import filters as flt
     import solver3d as s3
     
     prefix='opt1'
     dof = np.loadtxt('{}.txt'.format(prefix))
     hdof = mpj.dof2hdof(dof,nlay,nx,ny)
     eps, bdof = flt.hdof2eps(hdof,sig,nx,ny,nlay,mz,z0,beta,epsbkg,epsdiff)

     if comm.rank==0:
          import h5py as hp
          fid = hp.File('{}_geom.h5'.format(prefix),'w')
          for i in range(nlay):
               fid.create_dataset('layer{}'.format(i),data=bdof[i])
          fid.create_dataset('eps',data=eps)
          fid.close()
     comm.Barrier()
     
if Job==3:

     import h5py as hp
     
     prefix='opt1far'

     nstart=20
     nsens,nintg=10,60
     nspan=nsens*nintg

     Dzarr=np.array([20.0])
     for Dz in Dzarr:
          U = np.array([]).reshape((nsens*nsens,0))
          for ifreq in range(nfreqs):
               fid = hp.File('{}_ipol0ifreq{}.h5'.format(prefix,ifreq),'r')
               e0x = (np.array(fid["farex.r.Dz{}".format(Dz)]) + 1j * np.array(fid["farex.i.Dz{}".format(Dz)]))[nstart:nstart+nspan,nstart:nstart+nspan].copy() 
               e0y = (np.array(fid["farey.r.Dz{}".format(Dz)]) + 1j * np.array(fid["farey.i.Dz{}".format(Dz)]))[nstart:nstart+nspan,nstart:nstart+nspan].copy() 
               e0z = (np.array(fid["farez.r.Dz{}".format(Dz)]) + 1j * np.array(fid["farez.i.Dz{}".format(Dz)]))[nstart:nstart+nspan,nstart:nstart+nspan].copy() 
               fid.close()
               fid = hp.File('{}_ipol1ifreq{}.h5'.format(prefix,ifreq),'r')
               e1x = (np.array(fid["farex.r.Dz{}".format(Dz)]) + 1j * np.array(fid["farex.i.Dz{}".format(Dz)]))[nstart:nstart+nspan,nstart:nstart+nspan].copy() 
               e1y = (np.array(fid["farey.r.Dz{}".format(Dz)]) + 1j * np.array(fid["farey.i.Dz{}".format(Dz)]))[nstart:nstart+nspan,nstart:nstart+nspan].copy() 
               e1z = (np.array(fid["farez.r.Dz{}".format(Dz)]) + 1j * np.array(fid["farez.i.Dz{}".format(Dz)]))[nstart:nstart+nspan,nstart:nstart+nspan].copy() 
               fid.close()
               
               u0 = np.abs(e0x)**2 + np.abs(e0y)**2 + np.abs(e0z)**2
               u1 = np.abs(e1x)**2 + np.abs(e1y)**2 + np.abs(e1z)**2
               u2 = np.real(e0x*np.conj(e1x) + e0y*np.conj(e1y) + e0z*np.conj(e1z))
               u3 = np.imag(e0x*np.conj(e1x) + e0y*np.conj(e1y) + e0z*np.conj(e1z))

               u0 = np.reshape(u0.reshape((nsens,nintg,nsens,nintg)).sum(axis=(1,3)),(nsens*nsens,1)) * du**2
               u1 = np.reshape(u1.reshape((nsens,nintg,nsens,nintg)).sum(axis=(1,3)),(nsens*nsens,1)) * du**2
               u2 = np.reshape(u2.reshape((nsens,nintg,nsens,nintg)).sum(axis=(1,3)),(nsens*nsens,1)) * du**2
               u3 = np.reshape(u3.reshape((nsens,nintg,nsens,nintg)).sum(axis=(1,3)),(nsens*nsens,1)) * du**2
               U = np.concatenate( (U,u0,u1,u2,u3), axis=1 )

          s0=np.linalg.norm(U, ord=-2)
          s1=np.linalg.norm(U, ord=2)
          cond = s1/s0
          print("For Dz {}, cond: {}".format(Dz,cond))

               
