import numpy as np
import scipy.ndimage as nd

def stepfunc(zz,eta,beta):

    b = beta if beta>1e-2 else 1e-2

    r1 = np.tanh(b*eta) + np.tanh(b*(zz-eta))
    r2 = np.tanh(b*eta) + np.tanh(b*(1.-eta))

    return 1. - np.divide(r1,r2)

def stepgrad(zz,eta,beta):

    b = beta if beta>1e-2 else 1e-2

    sech = np.divide(1.,np.cosh(b*(zz-eta)))
        
    return b * (np.sinh(b*zz)*np.sinh(b-b*zz)/np.sinh(b)) * np.multiply(sech,sech)

def varh_expand(hdof, nx,ny, nlay, mz, beta):

    edof=[]
    
    for i in range(nlay):
        tmp=np.zeros((nx,ny,mz[i]))
        for iz in range(mz[i]):
            zz = float(iz)/float(mz[i])
            tmp[:,:,iz]=stepfunc(zz,hdof[i][:,:],beta)
        edof.append(tmp)

    return edof

def varh_contract(egrad, hdof, nx,ny, nlay, mz, beta):

    hgrad=[]

    for i in range(nlay):
        tmp=np.zeros((nx,ny))
        for iz in range(mz[i]):
            zz = float(iz)/float(mz[i])
            tmp[:,:] += stepgrad(zz,hdof[i][:,:],beta) * egrad[i][:,:,iz]
        hgrad.append(tmp)

    return hgrad

def hdof2eps(hdof,sig,nx,ny,nlay,mz,z0,beta,epsbkg,epsdiff):

    print("Inside hdof2eps")
    eps=np.array(epsbkg,copy=True)

    bdof=[]
    for jl in range(nlay):
        if sig<0.0001:
            bdof.append( hdof[jl] )
        else:
            bdof.append( nd.gaussian_filter(hdof[jl],sigma=sig) )

   
    edof=varh_expand(bdof, nx,ny, nlay, mz, beta)
    for jl in range(nlay):
        eps[:,:,z0[jl]:z0[jl]+mz[jl]] += epsdiff[jl] * edof[jl][:,:,:]

    print("Done hdof2eps")
    return eps, bdof

def grad2hgrad(grad,bdof,sig,nx,ny,nlay,mz,z0,beta,epsdiff):

    print("Inside grad2hgrad")

    egrad=[]
    for jl in range(nlay):
        egrad.append( epsdiff[jl]*grad[:,:,z0[jl]:z0[jl]+mz[jl]] )
    bgrad=varh_contract(egrad, bdof, nx,ny, nlay, mz, beta)

    hgrad=[]
    for jl in range(nlay):
        if sig<0.0001:
            hgrad.append( bgrad[jl] )
        else:
            hgrad.append( nd.gaussian_filter(bgrad[jl],sigma=sig) )

    print("Done grad2hgrad")
    
    return hgrad

        
