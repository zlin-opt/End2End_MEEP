import numpy as np
import meep as mp
import mpi4py.MPI as MPI
comm=MPI.COMM_WORLD

def dat2pos(r,datx,daty,datz,du,nx,ny,nz):

    decplace=5
    x0=np.around(-nx*du/2.,decplace)
    y0=np.around(-ny*du/2.,decplace)
    z0=np.around(-nz*du/2.,decplace)
    rx=np.around(r.x,decplace)
    ry=np.around(r.y,decplace)
    rz=np.around(r.z,decplace)
    ix0=np.around( (rx-x0)/du, decplace)
    iy0=np.around( (ry-y0)/du, decplace)
    iz0=np.around( (rz-z0)/du, decplace)
    ix = int(np.floor( ix0 ))
    iy = int(np.floor( iy0 ))
    iz = int(np.floor( iz0 ))
    dx = ix0-float(ix)
    dy = iy0-float(iy)
    dz = iz0-float(iz)

    if   dx>0.001 and dy<0.001 and dz<0.001:
        return datx[(ix+0)%nx,iy-1,iz-1]
    elif dx<0.001 and dy>0.001 and dz<0.001:
        return daty[ix-1,(iy+0)%ny,iz-1]
    elif dx<0.001 and dy<0.001 and dz>0.001:
        return datz[ix-1,iy-1,(iz+0)%nz]
    else:
        return 0.0
    
def fdtd(epsx,epsy,epsz,
         jx,jy,jz,
         nx,ny,nz,du,
         npmlx,npmly,npmlz,
         fcen,df,courant,
         src_intg=True,
         mtr_tol=1e-5):

    dpmlx=npmlx*du
    dpmly=npmly*du
    dpmlz=npmlz*du
    Lx=nx*du
    Ly=ny*du
    Lz=nz*du

    resolution=1./du

    mtr_dt=10./fcen
    mtr_c=[mp.Ex,mp.Ey,mp.Ez]
    mtr_r=mp.Vector3(0,0,Lz/2.-dpmlz-du)

    cell = mp.Vector3(Lx,Ly,Lz)
    pml_layers = [mp.PML(thickness=dpmlx,direction=mp.X),
                  mp.PML(thickness=dpmly,direction=mp.Y),
                  mp.PML(thickness=dpmlz,direction=mp.Z)]

    def epsfun(r):
        return dat2pos(r,epsx,epsy,epsz,du,nx,ny,nz)

    def jfun(r):
        return dat2pos(r,jx,jy,jz,du,nx,ny,nz)

    sources = [ mp.Source(mp.GaussianSource(fcen,fwidth=df,is_integrated=src_intg),
                          component=mp.Ex,
                          center=mp.Vector3(0,0,0),
                          size=cell,
                          amp_func=jfun),
                mp.Source(mp.GaussianSource(fcen,fwidth=df,is_integrated=src_intg),
                          component=mp.Ey,
                          center=mp.Vector3(0,0,0),
                          size=cell,
                          amp_func=jfun),
                mp.Source(mp.GaussianSource(fcen,fwidth=df,is_integrated=src_intg),
                          component=mp.Ez,
                          center=mp.Vector3(0,0,0),
                          size=cell,
                          amp_func=jfun) ]

    sim = mp.Simulation(cell_size=cell,
                         boundary_layers=pml_layers,
                         epsilon_func=epsfun,
                         eps_averaging=False,
                         sources=sources,
                         resolution=resolution,
                         force_complex_fields=True,
                         Courant=courant,
                         k_point=mp.Vector3())

    dft_vol = mp.Volume(center=mp.Vector3(0,0,0), size=cell)
    dft_obj = sim.add_dft_fields([mp.Ex,mp.Ey,mp.Ez], fcen, fcen,1, where=dft_vol, yee_grid=True)
    sim.run( until_after_sources=mp.stop_when_fields_decayed(mtr_dt, mtr_c, mtr_r, mtr_tol) )

    ex = sim.get_dft_array(dft_obj, mp.Ex, 0)[1:1+nx,1:1+ny,1:1+nz].copy()
    ey = sim.get_dft_array(dft_obj, mp.Ey, 0)[1:1+nx,1:1+ny,1:1+nz].copy()
    ez = sim.get_dft_array(dft_obj, mp.Ez, 0)[1:1+nx,1:1+ny,1:1+nz].copy()
    
    dt = sim.fields.dt

    sim.reset_meep()
    del sim

    return ex,ey,ez,dt




####Parallel routines below
def get_num_groups():
    # Lazy import
    from mpi4py import MPI
    comm = MPI.COMM_WORLD

    return comm.allreduce(int(mp.my_rank() == 0), op=MPI.SUM)

def get_group_masters():
    # Lazy import
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    num_workers = comm.Get_size()
    num_groups = get_num_groups

    # Check if current worker is a group master
    is_group_master = True if mp.my_rank() == 0 else False
    group_master_idx = np.zeros((num_workers,),dtype=np.bool)

    # Formulate send and receive packets
    smsg = [np.array([is_group_master]),([1]*num_workers, [0]*num_workers)]
    rmsg = [group_master_idx,([1]*num_workers, list(range(num_workers)))]

    # Send and receive
    comm.Alltoallv(smsg, rmsg)

    # get rank of each group master
    group_masters = np.arange(num_workers)[group_master_idx] # rank index of each group leader

    return group_masters

def merge_subgroup_data(data):
    # Lazy import
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    num_workers = comm.Get_size()
    num_groups = get_num_groups()
    
    # Initialize new input and output datasets
    input=np.array(data,copy=True,order='F')
    shape=input.shape
    size=input.size
    out_shape=shape + (num_groups,)
    output=np.zeros(out_shape,input.dtype,order='F')

    # Get group masters
    group_masters = get_group_masters()
    
    # Specify how much talking each proc will do. Only group masters send data.
    if mp.my_rank() == 0:
        scount = np.array([size] * num_workers)
    else:
        scount = np.array([0] * num_workers)
    rcount = np.array([0] * num_workers)
    rcount[group_masters] = size

    # Specify array mapping
    sdsp = [0] * num_workers
    rdsp = [0] * num_workers
    buf_idx = 0
    for grpidx in group_masters:
        rdsp[grpidx] = buf_idx # offset group leader worker by size of each count
        buf_idx += size

    # Formulate send and receive packets
    smsg = [input, (scount, sdsp)]
    rmsg = [output, (rcount, rdsp)]
    
    # Send and receive
    comm.Alltoallv(smsg, rmsg)
    
    return output

