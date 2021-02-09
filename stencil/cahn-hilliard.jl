using OffsetArrays

noise = 0.02
c0 = 0.4
gradC = 0.5
dtime = 0.01
nstep = 5000
mobility = 1.0
Nx = 65; Ny = 65
micro = reshape(c0 .+ noise .* (0.5 .- rand(Float64,Nx*Ny)),Nx,Ny)
I = CartesianIndices(micro)
# create padding
A = OffsetArray(zeros(Float64,Nx+2,Ny+2),0:Nx+1,0:Ny+1)
for i in I
    A[i] = micro[i]
end

# Create CartesianIndices for going up and down in matrix
neighborhood = CartesianIndices((-1:1,-1:1))
# Stencil for the laplace operator
stencil = [0  1  0
           1  -4 1
           0  1  0]

# enforce periodic boundary conditions
function periodic_bound(M)
    B = copy(M)
    B[0,:]    = B[Nx,:]
    B[Nx+1,:] = B[1,:]
    B[:,0]    = B[:,Ny]
    B[:,Ny+1] = B[:,1]
    return B
end

for istep = 1:nstep
    global A
    B = periodic_bound(A)
    # update the concentration using the laplacian stencil
    LB = copy(B)
    for i in I
        LB[i] = sum(B[i.+neighborhood].*stencil)
    end

    # do derivative of the free energy
    dF = 2.0 .* B .* (1 .- B).^2-2.0 * B.^2 .*(1.0 .- B)

    # create new matrix for doing laplacian again
    D  = dF - gradC .* LB

    # do laplacian again
    DL = periodic_bound(D)
    L = copy(D)
    for i in I
        L[i] = sum(DL[i.+neighborhood].*stencil)
    end
    # do the time integration
    A = A + (dtime * mobility) .* L
end

# plot solution of Cahn-Hilliard equation
using Plots
x = [1:Nx]
y = [1:Ny]
heatmap(x[:],y[:], A[1:Nx,1:Ny], clims=(0,1.1), title = "Cahn-Hilliard phase separation",
    color=:rainbow,linewidth = 0.,size=(540,540))
