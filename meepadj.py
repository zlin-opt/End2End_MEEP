import solver3d as s3
import filters as flt
import sigproc as sgp
import numpy as np
import meep as mp
import gc
from mpi4py import MPI
comm = MPI.COMM_WORLD

def dof2hdof(dof,nlay,nx,ny):
    hdof=[]
    for i in range(nlay):
        hdof.append( dof[i*nx*ny:(i+1)*nx*ny].reshape((nx,ny),order='F') )
    return hdof

def hdof2dof(hdof,nlay):
    dof=hdof[0].flatten(order='F')
    for i in range(1,nlay):
        dof=np.concatenate((dof,hdof[i].flatten(order='F')))
    return dof

count = [0]
def mse(dof, gdat,
        nx,ny,nz,du,npml,
        nlay,mz,z0, sig, beta,
        epsbkg, epsdiff,
        srcz, mtrz,
        freqs, df, courant,
        nrep,fgs,nsens,nintg,alpha):

    if comm.rank==0:
        print('Applying filters')
        eps, bdof = flt.hdof2eps(dof2hdof(dof,nlay,nx,ny),
                                 sig,nx,ny,nlay,mz,z0,beta,epsbkg,epsdiff)
    else:
        eps, bdof = None, None
    comm.Barrier()
    eps = comm.bcast(eps,root=0)
    bdof = comm.bcast(bdof,root=0)

    nfreqs=freqs.size

    ######
    isrc = mp.divide_parallel_processes(2*nfreqs)
    ipol = int(isrc%2)
    ifreq = int(isrc//2)

    pwsrc = np.zeros((nx,ny,nz),dtype=complex)
    pwsrc[:,:,srcz]=1./du
    print("Starting the forward simulation.")
    fcen=freqs[ifreq]
    omega=2.*np.pi*fcen
    ex,ey,ez,dt = s3.fdtd(eps,eps,eps,
                          float(1-ipol)*pwsrc,
                          float(  ipol)*pwsrc,
                          np.zeros((nx,ny,nz),dtype=complex),
                          nx,ny,nz,du,
                          0,0,npml,
                          fcen,df,courant,
                          src_intg=True,
                          mtr_tol=1e-8)
    iw = (1.0 - np.exp(-1j*omega*dt)) * (1.0/dt)
    w2 = omega*omega
    t0 = 5.0/df
    expfac = np.exp(1j*omega*t0)
    
    mex=ex[:,:,mtrz-1:mtrz+1].copy()
    mey=ey[:,:,mtrz-1:mtrz+1].copy()
    mez=ez[:,:,mtrz-1       ].copy()
    
    print("Gather the fields at the monitor plane.")
    tmpx = s3.merge_subgroup_data(mex)
    tmpy = s3.merge_subgroup_data(mey)
    tmpz = s3.merge_subgroup_data(mez)

    print('Signal Processing')
    if comm.rank==0:
        np.savetxt('dofstep{}.txt'.format(count[0]),dof)
        errnorm, gx,gy,gz, cond = sgp.postproc(tmpx,tmpy,tmpz,nrep,
                                               du,freqs,fgs,
                                               nsens,nintg,alpha)
    else:
        errnorm, gx,gy,gz, cond = None, None,None,None, None
    comm.Barrier()
    errnorm = comm.bcast(errnorm,root=0)
    gx = comm.bcast(gx,root=0)
    gy = comm.bcast(gy,root=0)
    gz = comm.bcast(gz,root=0)
    cond = comm.bcast(cond,root=0)
    print('At step {}, errnorm: {}, cond: {}'.format(count[0],errnorm,cond))
    
    if gdat.size>0:
        print("Starting the adjoint simulation.")
        adjx=np.zeros((nx,ny,nz),dtype=complex)
        adjy=np.zeros((nx,ny,nz),dtype=complex)
        adjz=np.zeros((nx,ny,nz),dtype=complex)
        if ipol==0:
            adjx[:,:,mtrz-1:mtrz+1] = np.conj(gx[:,:,:,0 + 2*ifreq]) * (df/(expfac*iw))
            adjy[:,:,mtrz-1:mtrz+1] = np.conj(gy[:,:,:,0 + 2*ifreq]) * (df/(expfac*iw))
            adjz[:,:,mtrz-1       ] = np.conj(gz[:,:,  0 + 2*ifreq]) * (df/(expfac*iw))
        else:
            adjx[:,:,mtrz-1:mtrz+1] = np.conj(gx[:,:,:,1 + 2*ifreq]) * (df/(expfac*iw))
            adjy[:,:,mtrz-1:mtrz+1] = np.conj(gy[:,:,:,1 + 2*ifreq]) * (df/(expfac*iw))
            adjz[:,:,mtrz-1       ] = np.conj(gz[:,:,  1 + 2*ifreq]) * (df/(expfac*iw))

            
        ux,uy,uz, dt = s3.fdtd(eps,eps,eps,
                               adjx,
                               adjy,
                               adjz,
                               nx,ny,nz,du,
                               0,0,npml,
                               fcen,df,courant,
                               src_intg=False,
                               mtr_tol=1e-8)

        grad = comm.reduce( np.real( w2 * (ux*ex + uy*ey + uz*ez) ) if mp.am_master() else np.zeros(ux.shape),
                            op=MPI.SUM,
                            root=0 )

        if comm.rank==0:
            tmp = hdof2dof(flt.grad2hgrad(grad,bdof,sig,nx,ny,nlay,mz,z0,beta,epsdiff),
                           nlay)
        else:
            tmp = None
        comm.Barrier()
        tmp = comm.bcast(tmp,root=0)

        gdat[:] = tmp[:]
        
        adjx,adjy,adjz = None,None,None
        ux,uy,uz = None,None,None
        grad,tmp = None,None

    ex,ey,ez = None,None,None
    mex,mey,mez = None,None,None
    tmpx,tmpy,tmpz = None,None,None
    gx,gy,gz=None,None,None
    eps,bdof = None,None
    gc.collect()
    
    count[0] = count[0] + 1
    return errnorm
