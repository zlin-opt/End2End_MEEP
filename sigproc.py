import autograd.numpy as np
from autograd import grad
import n2f

def e2dof(ex,ey,ez):
        
    return np.concatenate(( np.real(ex.reshape((ex.shape[0]*ex.shape[1]*ex.shape[2]*ex.shape[3],1))),
                            np.real(ey.reshape((ey.shape[0]*ey.shape[1]*ey.shape[2]*ey.shape[3],1))),
                            np.real(ez.reshape((ez.shape[0]*ez.shape[1]*ez.shape[2]            ,1))),
                            np.imag(ex.reshape((ex.shape[0]*ex.shape[1]*ex.shape[2]*ex.shape[3],1))),
                            np.imag(ey.reshape((ey.shape[0]*ey.shape[1]*ey.shape[2]*ey.shape[3],1))),
                            np.imag(ez.reshape((ez.shape[0]*ez.shape[1]*ez.shape[2]            ,1))) ),axis=0)

def dof2e(dof, nx,ny,nfreqs):

    n=int(dof.size/2)
    cdof = dof[:n] + 1j * dof[n:]
    
    exsize=nx*ny*2*2*nfreqs
    eysize=nx*ny*2*2*nfreqs
    ezsize=nx*ny  *2*nfreqs
    ex = np.reshape(cdof[0            :exsize              ],(nx,ny,2,2*nfreqs))
    ey = np.reshape(cdof[exsize       :exsize+eysize       ],(nx,ny,2,2*nfreqs))
    ez = np.reshape(cdof[exsize+eysize:exsize+eysize+ezsize],(nx,ny,  2*nfreqs))

    return ex,ey,ez

def mse(ex_in,ey_in,ez_in,nrep,
        du,freqs,fgs,
        nsens,nintg,alpha,
        chk=0):
    
    ex=np.tile(ex_in,(nrep,nrep,1,1))
    ey=np.tile(ey_in,(nrep,nrep,1,1))
    ez=np.tile(ez_in,(nrep,nrep,  1))

    nx,ny = int(ex.shape[0]),int(ex.shape[1])
    nfreqs = int(ex.shape[3]/2)
    nfar = int(nsens*nintg)
    ngreen = int(nfar+nx)
    pad = int((ngreen-nx)/2)

    U = np.array([]).reshape((nsens*nsens,0))
    for ifreq in range(nfreqs):
        ne0x,ne0y,nh0x,nh0y = n2f.geteh(ex[:,:,:,0+2*ifreq],ey[:,:,:,0+2*ifreq],ez[:,:,0+2*ifreq], du,du,du, freqs[ifreq])
        ne1x,ne1y,nh1x,nh1y = n2f.geteh(ex[:,:,:,1+2*ifreq],ey[:,:,:,1+2*ifreq],ez[:,:,1+2*ifreq], du,du,du, freqs[ifreq])
        ne0x = np.pad(ne0x, ((pad,pad),(pad,pad)), mode='constant')
        ne0y = np.pad(ne0y, ((pad,pad),(pad,pad)), mode='constant')
        nh0x = np.pad(nh0x, ((pad,pad),(pad,pad)), mode='constant')
        nh0y = np.pad(nh0y, ((pad,pad),(pad,pad)), mode='constant')
        ne1x = np.pad(ne1x, ((pad,pad),(pad,pad)), mode='constant')
        ne1y = np.pad(ne1y, ((pad,pad),(pad,pad)), mode='constant')
        nh1x = np.pad(nh1x, ((pad,pad),(pad,pad)), mode='constant')
        nh1y = np.pad(nh1y, ((pad,pad),(pad,pad)), mode='constant')
        
        ffe0x,ffe0y,ffe0z = n2f.n2f(ne0x,ne0y,
                                    nh0x,nh0y,
                                    freqs[ifreq], 1.,1., fgs[ifreq])
        ffe1x,ffe1y,ffe1z = n2f.n2f(ne1x,ne1y,
                                    nh1x,nh1y,
                                    freqs[ifreq], 1.,1., fgs[ifreq])

        fe0x = np.array( ffe0x[int((ngreen-nfar)/2):int((ngreen-nfar)/2)+nfar,int((ngreen-nfar)/2):int((ngreen-nfar)/2)+nfar] , copy=True)
        fe0y = np.array( ffe0y[int((ngreen-nfar)/2):int((ngreen-nfar)/2)+nfar,int((ngreen-nfar)/2):int((ngreen-nfar)/2)+nfar] , copy=True)
        fe0z = np.array( ffe0z[int((ngreen-nfar)/2):int((ngreen-nfar)/2)+nfar,int((ngreen-nfar)/2):int((ngreen-nfar)/2)+nfar] , copy=True)
        fe1x = np.array( ffe1x[int((ngreen-nfar)/2):int((ngreen-nfar)/2)+nfar,int((ngreen-nfar)/2):int((ngreen-nfar)/2)+nfar] , copy=True)
        fe1y = np.array( ffe1y[int((ngreen-nfar)/2):int((ngreen-nfar)/2)+nfar,int((ngreen-nfar)/2):int((ngreen-nfar)/2)+nfar] , copy=True)
        fe1z = np.array( ffe1z[int((ngreen-nfar)/2):int((ngreen-nfar)/2)+nfar,int((ngreen-nfar)/2):int((ngreen-nfar)/2)+nfar] , copy=True)

        u0 = np.abs(fe0x)**2 + np.abs(fe0y)**2 + np.abs(fe0z)**2
        u1 = np.abs(fe1x)**2 + np.abs(fe1y)**2 + np.abs(fe1z)**2
        u2 = np.real(fe0x*np.conj(fe1x) + fe0y*np.conj(fe1y) + fe0z*np.conj(fe1z))
        u3 = np.imag(fe0x*np.conj(fe1x) + fe0y*np.conj(fe1y) + fe0z*np.conj(fe1z))
        
        u0 = np.reshape(u0.reshape((nsens,nintg,nsens,nintg)).sum(axis=(1,3)),(nsens*nsens,1)) * du**2
        u1 = np.reshape(u1.reshape((nsens,nintg,nsens,nintg)).sum(axis=(1,3)),(nsens*nsens,1)) * du**2
        u2 = np.reshape(u2.reshape((nsens,nintg,nsens,nintg)).sum(axis=(1,3)),(nsens*nsens,1)) * du**2
        u3 = np.reshape(u3.reshape((nsens,nintg,nsens,nintg)).sum(axis=(1,3)),(nsens*nsens,1)) * du**2
        U = np.concatenate( (U,u0,u1,u2,u3), axis=1 )

        #print([np.mean(u0),np.mean(u1)])
        
    ncols = int(4*nfreqs)
    UTU = np.matmul(np.transpose(U),U)
    Lreg = np.add( UTU, alpha*np.identity(ncols) )
    Linv = np.linalg.inv(Lreg)

    UTUx = np.matmul(UTU, np.identity(ncols))
    err = np.add( np.matmul(Linv,UTUx), -np.identity(ncols) )
    errnorm = np.linalg.norm( err )

    if chk==0:
        return errnorm
    else:
        return U, errnorm

def postproc(ex,ey,ez,nrep,
             du,freqs,fgs,
             nsens,nintg,alpha):

    nx,ny = int(ex.shape[0]),int(ex.shape[1])
    nfreqs = int(ex.shape[3]/2)
    dof=e2dof(ex,ey,ez)
    
    def tmpfun(dof):
        eex, eey, eez = dof2e(dof, nx,ny,nfreqs)
        return mse(eex,eey,eez,nrep,
                   du,freqs,fgs,
                   nsens,nintg,alpha,
                   chk=0)

    print("sigproc grad construction in progress.")
    g = grad(tmpfun)
    gx, gy, gz = dof2e(g(dof), nx,ny,nfreqs)
    print("sigproc grad construction completed.")

    ###print the svds, condition number and individual errors
    print("computing mse in progress, including ffts")
    U, errnorm = mse(ex,ey,ez,nrep,
                     du,freqs,fgs,
                     nsens,nintg,alpha,
                     chk=1)
    print("computing mse completed, including ffts")
    
    s0=np.linalg.norm(U, ord=-2)
    s1=np.linalg.norm(U, ord=2)
    cond = s1/s0
    print("From sigproc, smallest svd of U: {}".format(s0))
    print("From sigproc, largest svd of U: {}".format(s1))
    print("From sigproc, condition number of U: {}".format(cond))
    print("From sigproc, errnorm: {}".format(errnorm))
    ###########

    return errnorm, gx, gy, gz, cond

    
test=0
if test==1:
    nx,ny,nfreqs=60,60,4
    nrep=10
    np.random.seed(5678)
    ex=np.random.uniform(low=-1.,high=1.,size=(nx,ny,2,2*nfreqs)) + 1j * np.random.uniform(low=-1.,high=1.,size=(nx,ny,2,2*nfreqs))
    ey=np.random.uniform(low=-1.,high=1.,size=(nx,ny,2,2*nfreqs)) + 1j * np.random.uniform(low=-1.,high=1.,size=(nx,ny,2,2*nfreqs))
    ez=np.random.uniform(low=-1.,high=1.,size=(nx,ny,  2*nfreqs)) + 1j * np.random.uniform(low=-1.,high=1.,size=(nx,ny,  2*nfreqs))

    eex,eey,eez = dof2e(e2dof(ex,ey,ez), nx,ny,nfreqs)
    print("exdiff: {}".format(np.linalg.norm(ex-eex)))
    print("eydiff: {}".format(np.linalg.norm(ey-eey)))
    print("ezdiff: {}".format(np.linalg.norm(ez-eez)))

    freqs=np.linspace(0.93,1.0,nfreqs)
    Dz=20.
    du=0.05
    nsens,nintg=10,20
    alpha=0.1
    fgs=[]
    for ifreq in range(nfreqs):
        fgs.append(n2f.greens(nsens*nintg+nx*nrep,nsens*nintg+nx*nrep,du,du, freqs[ifreq], 1.,1., Dz))

    def obj(dof):
        ex,ey,ez = dof2e(dof, nx,ny,nfreqs)
        return mse(ex,ey,ez,nrep, du,freqs,fgs, nsens,nintg,alpha, chk=0)

    g=grad(obj)

    dof0=e2dof(ex,ey,ez)
    print(np.linalg.norm(g(dof0)))

    n=int(dof0.size)
    dp=0.001
    tmp = np.zeros(n)
    ndat = 10
    for i in range(ndat):

        dof = np.random.uniform(low=-3.,high=3.,size=n)
        chk = np.random.randint(low=0, high=n)

        adj = g(dof)[chk]
        
        tmp[:] = dof[:]
        tmp[chk] -= dp
        mse0 = obj(tmp)
        tmp[:] = dof[:]
        tmp[chk] += dp
        mse1 = obj(tmp)
    
        cendiff = (mse1 - mse0)/(2.*dp)

        print("check: {} {} {}".format(adj,cendiff,(adj-cendiff)/adj))

    
