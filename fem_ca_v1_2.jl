using LinearAlgebra
using SparseArrays
using LinearSolve

# open the mesh
input = open("mesh_64.inp","r")
# open the results
out   = open("result_1.out","w")
out2  = open("area_fract.out", "w")

nstep = 5000
nprnt = 50
dtime = 0.005
ttime = 0.0

# Material specific parameters
mobil = 5.0
grcof = 0.1

# input data
ncountm, ncounts, master, slave = periodic_boundary(npoin, coord)

posgp, weigp = gauss(ngaus, nnode)

dgdx, dvolum = cart_deriv(npoin, nelem, nnode, nstre, ndime, ndofn, ngaus, 
ntype, lnods, coord, posgp, weigp)

# initialize microstructure
iflag = 1

etas, ngrain = init_micro_grain_fem(npoin, coord, iflag)
ntotv = npoin * ngrain

#---------------------------
# form stiffness matrix
#---------------------------

gstif = gg_stif_1(ngrain, npoin, nelem, nnode, nstre, ndime, ndofn,
ngaus, ntype, lnods, coord, mobil, grcof, dtime, posgp, weigp, dgdx, dvolum)

# Rearrange gstif for PBC
gstif = apply_periodic_bc2a(ncountm, ncounts, master, slave, ndofn, npoin, gstif)

#---------------------------
# EVOLVE
#---------------------------
for istep = 1:nstep
    ttime = ttime + dtime
    for igrain = 1:ngrain
        gforce = gg_rhs_1(ngrain, npoin, )