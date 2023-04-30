import autograd.numpy as np

def greens(nx,ny,dx,dy, freq, eps,mu, Dz):

    omega = 2.*np.pi*freq
    n = np.sqrt(eps*mu)
    k = n*omega
    
    Lx, Ly = nx*dx, ny*dy

    x, y= np.meshgrid( np.linspace(-Lx/2.,Lx/2.-dx,nx),
                       np.linspace(-Ly/2.,Ly/2.-dy,ny) )

    r = np.sqrt( x**2 + y**2 + Dz**2 )
    expfac = np.exp(1j * k * r)
    dxy = dx*dy
    
    g = expfac/(4*np.pi*r) *dxy

    gx = x  * (-1. + 1j * k * r) * expfac/(4.*np.pi*r**3) * dxy
    gy = y  * (-1. + 1j * k * r) * expfac/(4.*np.pi*r**3) * dxy
    gz = Dz * (-1. + 1j * k * r) * expfac/(4.*np.pi*r**3) * dxy

    gxx = (3.*x**2/r**3 - 3.*1j*k*x**2/r**2 - (1.+k**2*x**2)/r + 1j*k) * expfac/(4.*np.pi*r**2) * dxy
    gyy = (3.*y**2/r**3 - 3.*1j*k*y**2/r**2 - (1.+k**2*y**2)/r + 1j*k) * expfac/(4.*np.pi*r**2) * dxy

    gxy = (3./r**2 - 3.*1j*k/r - k**2) * expfac * x*y /(4.*np.pi*r**3) * dxy
    gzx = (3./r**2 - 3.*1j*k/r - k**2) * expfac * x*Dz/(4.*np.pi*r**3) * dxy
    gzy = (3./r**2 - 3.*1j*k/r - k**2) * expfac * y*Dz/(4.*np.pi*r**3) * dxy

    fg = np.fft.fft2( g )

    fgx = np.fft.fft2( gx )
    fgy = np.fft.fft2( gy )
    fgz = np.fft.fft2( gz )

    fgxx = np.fft.fft2( gxx ) 
    fgyy = np.fft.fft2( gyy ) 
    fgxy = np.fft.fft2( gxy ) 
    fgzx = np.fft.fft2( gzx ) 
    fgzy = np.fft.fft2( gzy ) 

    ret = {"fg" : fg,
           "fgx" : fgx, "fgy" : fgy, "fgz" : fgz,
           "fgxx" : fgxx, "fgyy" : fgyy, "fgxy" : fgxy, "fgzx" : fgzx, "fgzy" : fgzy}

    return ret

def n2f(ex_near,ey_near,
        hx_near,hy_near,
        freq, eps,mu, fgs,
        output_hfar=0):

    omega = 2.*np.pi*freq
    n = np.sqrt(eps*mu)
    k = n*omega

    g=fgs["fg"]
    gx,gy,gz = fgs["fgx"],fgs["fgy"],fgs["fgz"]
    gxx,gyy = fgs["fgxx"],fgs["fgyy"]
    gxy,gzx,gzy = fgs["fgxy"],fgs["fgzx"],fgs["fgzy"]

    ex = np.fft.fft2( ex_near )
    ey = np.fft.fft2( ey_near )
    hx = np.fft.fft2( hx_near )
    hy = np.fft.fft2( hy_near )

    fex = mu  * (-1j*omega*g*hy - 1j*omega/k**2 * (gxx*hy - gxy*hx) - 1./eps * gz*ex)
    fey = mu  * (+1j*omega*g*hx - 1j*omega/k**2 * (gxy*hy - gyy*hx) - 1./eps * gz*ey)
    fez = mu  * (-1j*omega/k**2 * (gzx*hy - gzy*hx) + 1./eps * (gx*ex + gy*ey))
    ex_far = np.fft.ifftshift(np.fft.ifft2( fex )) 
    ey_far = np.fft.ifftshift(np.fft.ifft2( fey )) 
    ez_far = np.fft.ifftshift(np.fft.ifft2( fez )) 

    if output_hfar == 0:
        return ex_far, ey_far, ez_far
    else:
        fhx = eps * (+1j*omega*g*ey + 1j*omega/k**2 * (gxx*ey - gxy*ex) - 1./mu  * gz*hx)
        fhy = eps * (-1j*omega*g*ex + 1j*omega/k**2 * (gxy*ey - gyy*ex) - 1./mu  * gz*hy)
        fhz = eps * (+1j*omega/k**2 * (gzx*ey - gzy*ex) + 1./mu  * (gx*hx + gy*hy))
        hx_far = np.fft.ifftshift(np.fft.ifft2( fhx )) 
        hy_far = np.fft.ifftshift(np.fft.ifft2( fhy )) 
        hz_far = np.fft.ifftshift(np.fft.ifft2( fhz )) 

        return ex_far, ey_far, ez_far, hx_far, hy_far, hz_far

def geteh(ex,ey,ez, dx,dy,dz, freq):

    omega = 2.*np.pi*freq
    iw = 1j*omega
    
    hx = ( ( np.roll(ez,-1,axis=1) - ez)/dy + (-ey[:,:,1] + ey[:,:,0])/dz )/iw
    hy = ( (-np.roll(ez,-1,axis=0) + ez)/dx + ( ex[:,:,1] - ex[:,:,0])/dz )/iw

    ex_sync = np.mean(0.5*ex + 0.5*np.roll(ex,-1,axis=1),axis=2)
    ey_sync = np.mean(0.5*ey + 0.5*np.roll(ey,-1,axis=0),axis=2)

    hx_sync = 0.5*hx + 0.5*np.roll(hx,-1,axis=0)
    hy_sync = 0.5*hy + 0.5*np.roll(hy,-1,axis=1)

    return ex_sync,ey_sync,hx_sync,hy_sync
    
