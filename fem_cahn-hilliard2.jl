using LinearAlgebra
using SparseArrays
using LinearSolve

# open the mesh
input = open("mesh4n_40.inp","r")
# open the results
out   = open("result_1.out","w")

# Time integration parameters
nstep = 5000
nprnt = 25
dtime = 2.0e-2
toler = 5.0e-5
miter = 10

# Material specific parameters
conc0 = 0.40
mobil = 1.0
grcoef = 0.5

# -----------------------
# input data
# -----------------------
npoin, nelem, ntype, nnode, ndofn, ndime, ngaus, nstre, nmats,
 nprop, lnods, matno, coord = input_fem_pf( input )

# periodic boundary conditions
ncountm, ncounts, master, slave = periodic_boundary(npoin,coord)

# gaussian integration
posgp, weigp = gauss(ngaus, nnode)

dgdx, dvolum = cart_deriv(npoin,nelem,nnode,nstre,ndime,ndofn,ngaus,
    ntype, lnods, coord, posgp, weigp)

ntotv = npoin*ndofn
nevab = nnode*ndofn

# -----------------------------------
# Prepare microstructure
# -----------------------------------
con = init_micro_ch_fem(npoin, ndofn, conc0)
# -----------------------------------
#              EVOLVE
# -----------------------------------
@time begin
for istep = 1:nstep
    con_old = con
    # Newton iteration
    gstif = spzeros(ntotv, ntotv)
    LU    = spzeros(ntotv, ntotv)
    up    = spzeros(ntotv, ntotv)
    asdis = spzeros(ntotv)
    for iter = 1:miter

        @elapsed gforce = chem_stiff_v2!(npoin,nelem,nnode,nstre,ndime,
            ndofn, ngaus, ntype, lnods, coord, mobil, grcoef, con,
            con_old,dtime,posgp,weigp,istep, iter, gstif)

        # Rearrange gstif & gforce for PBC
        gforce = apply_periodic_bc!(ncountm, ncounts, master, slave,
        ndofn, npoin, gstif, gforce, iter)

        # set preconditioner to diagonal
        D = Diagonal(gstif)
        # ---------------------------------
        # solve equations and update
        # ---------------------------------
        if (iter == 1) 
            prob = LinearProblem(gstif,gforce)
            linsol = init(prob)
            asdis = solve(linsol, IterativeSolversJL_GMRES(), Pl=D)
        end
        linsol = LinearSolve.set_b(asdis.cache, gforce)
        @elapsed asdis  = solve(linsol, IterativeSolversJL_GMRES(), Pl=D)
        
        # ---------------------------------  
        #  Recover slave node values
        # ---------------------------------
        
        recover_slave_dof!(asdis, ncountm, ncounts, master, slave, npoin, ndofn)

        # update concentration field
        con = con + asdis.u

        # for small deviations

        for ipoin = 1:npoin
            if (con[ipoin] <= 0.0001)
                con[ipoin] = 0.0001
            end
            if (con[ipoin] >= 0.9999)
                con[ipoin] = 0.9999
            end
        end

        # check norm for convergence
        normF = norm(gforce, 2)
        if (normF <= toler)
            break
        end
        #println("Max:", maximum(gstif))
    end # end of Newton
    # print out
    if ( istep%nprnt == 0 )
        println("Done step: ",istep)
        # fclose(out1)
        write_vtk_fem(npoin, nelem, nnode, lnods, coord, istep, con)
    end
end
end 

#print("compute time: ", compute_time)

function init_micro_ch_fem(npoin, ndofn, conc0)
    npoin2 = 2*npoin
    nodcon  = zeros(npoin2)

    noise = 0.02
    for ipoin = 1 : npoin
        nodcon[ipoin] = conc0 + noise*(.5-rand())
        nodcon[ipoin+npoin] = 0.0
    end
    return nodcon
end

function apply_periodic_bc!(ncountm, ncounts, master, slave, ndofn, npoin, gstif, gforce, iter)
    for ipbc = 1:ncountm
        im = master[ipbc]
        is = slave[ipbc]
        if (iter == 1)
            ## add rows
            gstif[im,:] = gstif[im,:] + gstif[is,:]

            ## add columns
            gstif[:,im] = gstif[:,im] + gstif[:,is]

            # zero slave dofs
            gstif[is,:] .= 0.0
            gstif[:,is] .= 0.0

            gstif[is,is] = 1.0
        end

        ## add rhs
        gforce[im] = gforce[im] + gforce[is]
        gforce[is] = 0.0
    end
    #return (gstif, gforce)
    return gforce
end

function cart_deriv(npoin, nelem, nnode, nstre, ndime, ndofn, ngaus, ntype, lnods,
    coord, posgp, weigp)

    ngaus2 = ngaus
    if (nnode == 3)
        ngaus2 = 1
    end

    mgaus = ngaus*ngaus2
    dvolum = zeros(nelem, mgaus)
    dgdx   = zeros(nelem,mgaus,ndime,nnode)
    elcod   = zeros(ndime, nnode)

    for ielem =  1 : nelem
        for inode = 1:nnode
            lnode = lnods[ielem, inode]
            for idime = 1:ndime
                elcod[idime, inode] = coord[lnode,idime]
            end
        end

        ## gauss points
        kgasp = 0
        for igaus = 1: ngaus
            exisp = posgp[igaus]
            for jgaus = 1:ngaus2
                etasp = posgp[jgaus]
                if (nnode == 3)
                    etasp = posgp[ngaus+igaus]
                end

                kgasp = kgasp + 1
                mgaus = mgaus + 1

                shape, deriv = srf2(exisp, etasp, nnode)

                cartd, djacb, gpcod = jacob3(ielem,elcod,kgasp,shape,deriv,nnode,ndime)

                dvolu = djacb*weigp[igaus]*weigp[jgaus]

                if (nnode == 3)
                    dvolu = djab*weigp[igaus]
                end
                dvolum[ielem, kgasp] = dvolu

                for idime = 1:ndime
                    for inode = 1:nnode
                        dgdx[ielem,kgasp,idime,inode] = cartd[idime, inode]
                    end
                end
            end # igaus
        end
    end
    return (dgdx, dvolum)
end

function chem_stiff_v2!(npoin,nelem,nnode,nstre,ndime,
            ndofn, ngaus, ntype, lnods, coord, mobil, grcoef, con,
            con_old,dtime,posgp,weigp,istep, iter,gstif)

            eload1 = zeros(nnode)
            eload2 = zeros(nnode)
        
            # ========================================
            # global and local variables
            # ========================================
            ntotv = npoin*ndofn
            nevab = nnode*ndofn
        
            eload = zeros(nevab)
        
            ngaus2 = ngaus
            if (nnode == 3)
                ngaus2 = 1
            end
        
            gforce = zeros(ntotv, 1)
            for ielem = 1:nelem
                # ===============================
                # initialize elements stiffness
                #          & rhs
                # ================================
        
                # stiffness matrices
                if (iter == 1)           
                    kcc = zeros(nnode,nnode)
                    kcm = zeros(nnode,nnode)
                    kmm = zeros(nnode,nnode)
                    kmc1 = zeros(nnode,nnode)
                    kmc = zeros(nnode,nnode)
                end # if
        
                # --- rhs
                eload1 = zeros(nnode)
                eload2 = zeros(nnode)
                # for inode = 1: nnode
                #     eload1[inode] = 0.0
                #     eload2[inode] = 0.0
                # end
                eload = zeros(nevab)
                # for ievab = 1:nevab
                #     eload[ievab] = 0.0
                # end
        
                cv = zeros(nnode)
                cm = zeros(nnode)
                cv_old = zeros(nnode)
        
                # =================================
                #   elemental values
                # =================================
                for inode = 1:nnode
                    lnode = lnods[ielem,inode]
                    cv[inode] = con[lnode]
                    cm[inode] = con[npoin+lnode]
                    cv_old[inode] = con_old[lnode]
                end
        
                elcod = zeros(ndime, nnode)
        
                # coords of the element nodes
                for inode = 1:nnode
                    lnode = lnods[ielem,inode]
                    for idime = 1:ndime
                        elcod[idime,inode] = coord[lnode,idime]
                    end
                end
        
                # ========================================
                #     integrate element stiffness
                #              & rhs
                # ========================================
                kgasp = 0
        
                for igaus = 1:ngaus
                    exisp = posgp[igaus]
                    for jgaus = 1:ngaus2
                        etasp = posgp[jgaus]
                        if (nnode == 3)
                            etasp = posgp[ngaus + igaus]
                        end
                        kgasp = kgasp + 1
                        shape, deriv = srf2(exisp,etasp,nnode)
        
                        cartd, djacb, gpcod = jacob3(ielem, elcod, kgasp, shape,
                        deriv, nnode, ndime)
        
                        dvolu = djacb*weigp[igaus]*weigp[jgaus]
        
                        if (nnode == 3)
                            dvolu = djacb*weigp[igaus]
                        end
        
                        # values at the gauss points
                        cvgp = 0.0
                        cmgp = 0.0
                        cv_ogp = 0.0
        
                        for inode = 1:nnode
                            cvgp = cvgp + cv[inode] * shape[inode]
                            cmgp = cmgp + cm[inode] * shape[inode]
                            cv_ogp = cv_ogp + cv_old[inode] * shape[inode]
                        end
        
                        # chemical potential
                        dfdc, df2dc = free_energy_fem_v1(cvgp)
        
                        if (iter == 1)
                            # kcc matrix
                            for inode = 1:nnode
                                for jnode = 1:nnode
                                    kcc[inode,jnode] = kcc[inode,jnode] +
                                        shape[inode]*shape[jnode]*dvolu
                                end
                            end
        
                            # kcm matrix
                            for inode = 1:nnode
                                for jnode = 1:nnode
                                    for idime = 1:ndime
                                        kcm[inode,jnode] = kcm[inode,jnode] +
                                            dtime * mobil * cartd[idime,inode]*cartd[idime,jnode]*dvolu
                                    end
                                end
                            end
        
                            # --- kmm matrix
                            for inode = 1:nnode
                                for jnode = 1:nnode
                                    kmm[inode,jnode] = kmm[inode,jnode] +
                                            shape[inode] * shape[jnode] * dvolu
        
                                end
                            end
        
                            # --- kmc matrix
                            for inode = 1:nnode
                                for jnode = 1:nnode
                                    for idime = 1:ndime
                                        kmc[inode,jnode] = kmc[inode,jnode] -
                                            grcoef*cartd[idime,inode] * cartd[idime,jnode] * dvolu
                                    end
                                end
                            end
        
                            for inode = 1:nnode
                                for jnode = 1:nnode
                                    kmc[inode,jnode] = kmc[inode,jnode] -
                                            df2dc*shape[inode]*shape[jnode] *dvolu
                                end
                            end
                        end # if iter
        
                        # element rhs
                        for inode = 1:nnode
                            eload1[inode] = eload1[inode] -
                                    shape[inode] * (cvgp - cv_ogp) * dvolu
                        end
        
                        for inode = 1:nnode
                            for jnode = 1:nnode
                                for idime = 1:ndime
                                    eload1[inode] = eload1[inode] - dtime*mobil*cmgp * shape[jnode] * cartd[idime,inode] * cartd[idime,jnode] * dvolu
                                end
                            end
                        end
        
                        for inode = 1:nnode
                            eload2[inode] = eload2[inode] - shape[inode]*(cmgp-dfdc)*dvolu
                        end
        
                        for inode = 1:nnode
                            for jnode = 1:nnode
                                for idime = 1:ndime
                                    eload2[inode] = eload2[inode] + grcoef*cvgp*shape[jnode]*cartd[idime,inode]*cartd[idime,jnode] * dvolu
                                end
                            end
                        end
        
                    end
                end
        
                
                # assemble element stiffness
                # and rhs
                if (iter == 1)
                    estif = zeros(nevab, nevab)
                    for inode = 1:nnode
                        ievab = nnode + inode
                        for jnode = 1:nnode
                            jevab = nnode+jnode
                            estif[inode,jnode] = kcc[inode,jnode]
                            estif[inode,jevab] = kcm[inode,jnode]
                            estif[ievab,jnode] = kmc[inode,jnode]
                            estif[ievab,jevab] = kmm[inode,jnode]
                        end
                    end
                end # if iter
        
                # rhs
                for inode = 1:nnode
                    ievab = nnode +inode
                    eload[inode] = eload1[inode]
                    eload[ievab] = eload2[inode]
                end
        
                #println(ntotv)
            
                # form global stiffness and rhs
                if (iter == 1)
                    for idofn = 1:ndofn
                        for inode = 1:nnode
                            ievab = (idofn - 1) * nnode + inode
                            for jdofn = 1:ndofn
                                for jnode = 1:nnode
                                    jevab = (jdofn-1)*nnode + jnode
                                    I = (idofn-1)*npoin+lnods[ielem,inode]
                                    J = (jdofn-1)*npoin+lnods[ielem,jnode]
                                  
                                    #gstif = gstif + sparse([I],[J], estif[ievab,jevab],ntotv,ntotv)
                                    gstif[I,J] += estif[ievab,jevab]
                                end
                            end
                        end
                    end
                end #if iter
        
                # --- rhs
                for idofn = 1:ndofn
                    for inode = 1:nnode
                            ievab = (idofn-1)*nnode+inode
                            I     = (idofn-1)*npoin + lnods[ielem,inode]
                            gforce[I] += eload[ievab]
                    end
                end
        
            end # ielem
            #return (gstif, gforce)
            return gforce
end

"""
This function forms the global stiffness and
global rhs, load, vector for solution of Cahn-Hilliard
Equation for modified Newton-Raphson solution algorithm.
"""
function chem_stiff_v1(npoin, nelem, nnode, nstre, ndime,
    ndofn, ngaus, ntype, lnods, coord, mobil, grcoef, con, con_old,
    dtime, posgp, weigp, istep, iter, gstif)


    eload1 = zeros(nnode)
    eload2 = zeros(nnode)

    # ========================================
    # global and local variables
    # ========================================
    ntotv = npoin*ndofn
    nevab = nnode*ndofn

    eload = zeros(nevab)

    ngaus2 = ngaus
    if (nnode == 3)
        ngaus2 = 1
    end

    gforce = zeros(ntotv, 1)
    for ielem = 1:nelem
        # ===============================
        # initialize elements stiffness
        #          & rhs
        # ================================

        # stiffness matrices
        if (iter == 1)           
            kcc = zeros(nnode,nnode)
            kcm = zeros(nnode,nnode)
            kmm = zeros(nnode,nnode)
            kmc1 = zeros(nnode,nnode)
            kmc = zeros(nnode,nnode)
        end # if

        # --- rhs
        eload1 = zeros(nnode)
        eload2 = zeros(nnode)
        # for inode = 1: nnode
        #     eload1[inode] = 0.0
        #     eload2[inode] = 0.0
        # end
        eload = zeros(nevab)
        # for ievab = 1:nevab
        #     eload[ievab] = 0.0
        # end

        cv = zeros(nnode)
        cm = zeros(nnode)
        cv_old = zeros(nnode)

        # =================================
        #   elemental values
        # =================================
        for inode = 1:nnode
            lnode = lnods[ielem,inode]
            cv[inode] = con[lnode]
            cm[inode] = con[npoin+lnode]
            cv_old[inode] = con_old[lnode]
        end

        elcod = zeros(ndime, nnode)

        # coords of the element nodes
        for inode = 1:nnode
            lnode = lnods[ielem,inode]
            for idime = 1:ndime
                elcod[idime,inode] = coord[lnode,idime]
            end
        end

        # ========================================
        #     integrate element stiffness
        #              & rhs
        # ========================================
        kgasp = 0

        for igaus = 1:ngaus
            exisp = posgp[igaus]
            for jgaus = 1:ngaus2
                etasp = posgp[jgaus]
                if (nnode == 3)
                    etasp = posgp[ngaus + igaus]
                end
                kgasp = kgasp + 1
                shape, deriv = srf2(exisp,etasp,nnode)

                cartd, djacb, gpcod = jacob3(ielem, elcod, kgasp, shape,
                deriv, nnode, ndime)

                dvolu = djacb*weigp[igaus]*weigp[jgaus]

                if (nnode == 3)
                    dvolu = djacb*weigp[igaus]
                end

                # values at the gauss points
                cvgp = 0.0
                cmgp = 0.0
                cv_ogp = 0.0

                for inode = 1:nnode
                    cvgp = cvgp + cv[inode] * shape[inode]
                    cmgp = cmgp + cm[inode] * shape[inode]
                    cv_ogp = cv_ogp + cv_old[inode] * shape[inode]
                end

                # chemical potential
                dfdc, df2dc = free_energy_fem_v1(cvgp)

                if (iter == 1)
                    # kcc matrix
                    for inode = 1:nnode
                        for jnode = 1:nnode
                            kcc[inode,jnode] = kcc[inode,jnode] +
                                shape[inode]*shape[jnode]*dvolu
                        end
                    end

                    # kcm matrix
                    for inode = 1:nnode
                        for jnode = 1:nnode
                            for idime = 1:ndime
                                kcm[inode,jnode] = kcm[inode,jnode] +
                                    dtime * mobil * cartd[idime,inode]*cartd[idime,jnode]*dvolu
                            end
                        end
                    end

                    # --- kmm matrix
                    for inode = 1:nnode
                        for jnode = 1:nnode
                            kmm[inode,jnode] = kmm[inode,jnode] +
                                    shape[inode] * shape[jnode] * dvolu

                        end
                    end

                    # --- kmc matrix
                    for inode = 1:nnode
                        for jnode = 1:nnode
                            for idime = 1:ndime
                                kmc[inode,jnode] = kmc[inode,jnode] -
                                    grcoef*cartd[idime,inode] * cartd[idime,jnode] * dvolu
                            end
                        end
                    end

                    for inode = 1:nnode
                        for jnode = 1:nnode
                            kmc[inode,jnode] = kmc[inode,jnode] -
                                    df2dc*shape[inode]*shape[jnode] *dvolu
                        end
                    end
                end # if iter

                # element rhs
                for inode = 1:nnode
                    eload1[inode] = eload1[inode] -
                            shape[inode] * (cvgp - cv_ogp) * dvolu
                end

                for inode = 1:nnode
                    for jnode = 1:nnode
                        for idime = 1:ndime
                            eload1[inode] = eload1[inode] - dtime*mobil*cmgp * shape[jnode] * cartd[idime,inode] * cartd[idime,jnode] * dvolu
                        end
                    end
                end

                for inode = 1:nnode
                    eload2[inode] = eload2[inode] - shape[inode]*(cmgp-dfdc)*dvolu
                end

                for inode = 1:nnode
                    for jnode = 1:nnode
                        for idime = 1:ndime
                            eload2[inode] = eload2[inode] + grcoef*cvgp*shape[jnode]*cartd[idime,inode]*cartd[idime,jnode] * dvolu
                        end
                    end
                end

            end
        end

        
        # assemble element stiffness
        # and rhs
        if (iter == 1)
            estif = zeros(nevab, nevab)
            for inode = 1:nnode
                ievab = nnode + inode
                for jnode = 1:nnode
                    jevab = nnode+jnode
                    estif[inode,jnode] = kcc[inode,jnode]
                    estif[inode,jevab] = kcm[inode,jnode]
                    estif[ievab,jnode] = kmc[inode,jnode]
                    estif[ievab,jevab] = kmm[inode,jnode]
                end
            end
        end # if iter

        # rhs
        for inode = 1:nnode
            ievab = nnode +inode
            eload[inode] = eload1[inode]
            eload[ievab] = eload2[inode]
        end

        #println(ntotv)
    
        # form global stiffness and rhs
        if (iter == 1)
            for idofn = 1:ndofn
                for inode = 1:nnode
                    ievab = (idofn - 1) * nnode + inode
                    for jdofn = 1:ndofn
                        for jnode = 1:nnode
                            jevab = (jdofn-1)*nnode + jnode
                            I = (idofn-1)*npoin+lnods[ielem,inode]
                            J = (jdofn-1)*npoin+lnods[ielem,jnode]
                          
                            #gstif = gstif + sparse([I],[J], estif[ievab,jevab],ntotv,ntotv)
                            gstif[I,J] += estif[ievab,jevab]
                        end
                    end
                end
            end
        end #if iter

        # --- rhs
        for idofn = 1:ndofn
            for inode = 1:nnode
                    ievab = (idofn-1)*nnode+inode
                    I     = (idofn-1)*npoin + lnods[ielem,inode]
                    gforce[I] += eload[ievab]
            end
        end

    end # ielem
    return (gstif, gforce)
end
    

    
function free_energy_fem_v1(c)
    constA = 1.0

    dfdc = constA * (2.0*c-6.0*c^2+4.0*c^3)
    df2dc = constA * (2.0 - 12.0*c+12.0*c^2)
    return (dfdc, df2dc)
end

function input_fem_pf(infile)
    header = split(chomp(readline(infile)))

    print(header)
    # read the input data
    npoin, nelem, nvfix, ntype, nnode, ndofn, ndime, ngaus, nstre, nmats, nprop = parse.(Int,header)

        lnods = zeros(Int64,nelem,nnode)
        matno = zeros(Int64,nelem)
        coord = zeros(npoin,ndime)
        nofix = zeros(Int64,nvfix)
        iffix = zeros(Int64,nvfix, ndofn)
        fixed = zeros(nvfix, ndofn)
        props = zeros(nmats, nprop)

        # read the element node numbers and material property number
        for ielem = 1:nelem
            eleminf = split(chomp(readline(infile)))
            data = parse.(Int, eleminf)
            jelem = data[1]
            lnods[jelem,:] = data[2:nnode + 1]
            matno[jelem] = data[nnode+2]
        end

        # read nodal coordinates
        for ipoin = 1:npoin
            point = split(chomp(readline(infile)))
            jpoin = parse.(Int,point[1])
            coord[ipoin,:] = parse.(Float64,point[2:3])
        end

        for ipoin = 1:npoin
            if (coord[ipoin,1] < 0.0)
                coord[ipoin,1] = 0.0
            end
            if (coord[ipoin,2] < 0.0)
                coord[ipoin,2] = 0.0
            end
        end
        return (npoin, nelem, ntype, nnode, ndofn, ndime, ngaus, nstre, 
            nmats, nprop, lnods, matno, coord)
end

    """
    periodic_boundary(npoin, coord)

    This function finds nodes that are at
    boundary of the simulation cell. For
    each pair of the edge assign master and slave node.
    """
function periodic_boundary(npoin, coord)
    ncountm = 0
    ncounts = 0

    master = zeros(Int, npoin)
    slave  = zeros(Int, npoin)

    # determine xmax, xmin, ymax, ymin
    xmax = maximum(coord[:,1])
    xmin = minimum(coord[:,1])
    ymax = maximum(coord[:,2])
    ymin = minimum(coord[:,2])

    for ipoin = 1:npoin
        # left master nodes
        diff = xmin - coord[ipoin,1]
        if (abs(diff) <= 1.e-6)
            ncountm = ncountm + 1
            master[ncountm] = ipoin
        end

        # right slave nodes
        diff = xmax - coord[ipoin,1]
        if (abs(diff) <= 1.e-6)
            ncounts = ncounts + 1
            slave[ncounts] = ipoin
        end
    end

    for ipoin = 1:npoin
        # Bottom master nodes
        diff = ymin - coord[ipoin,2]
        if (abs(diff) <= 1.e-6)
            ncountm = ncountm + 1
            master[ncountm] = ipoin
        end

        # Top slave nodes
        diff = ymax - coord[ipoin,2]
        if (abs(diff) <= 1.e-6)
            ncounts = ncounts + 1
            slave[ncounts] = ipoin
        end
    end

    
    if (ncountm != ncounts)
        println("Xmin: ", xmin, " Xmax: ", xmax)
        println("Ymin: ", ymin, " Ymax: ", ymax)
        println("Number of master nodes: ", ncountm)
        println("Number of slave nodes: ", ncounts)
        error("ncountm should be equal ncounts")
    end

    return (ncountm, ncounts, master, slave)
        
end

function gauss(ngaus, nnode)
    posgp = zeros(2*ngaus)
    weigp = zeros(ngaus)
    
    if (nnode == 3)
        if (ngaus == 1)
            posgp[1] = 1.0/3.0
            posgp[2] = 1.0/3.0
            weigp[1] = 0.5
        end
        
        if (ngaus == 3)
            posgp[1] = 0.5
            posgp[2] = 0.5
            posgp[3] = 0.0
            
            posgp[4] = 0.0
            posgp[5] = 0.5
            posgp[6] = 0.5
            
            weigp[1] = 1.0/6.0
            weigp[2] = 1.0/6.0
            weigp[3] = 1.0/6.0    
        end
        
        if (ngaus == 7)
            posgp[1] = 0.0
            posgp[2] = 0.5
            posgp[3] = 1.0
            posgp[4] = 0.5
            posgp[5] = 0.0
            posgp[6] = 0.0
            posgp[7] = 1.0/3.0
            
            posgp[8] = 0.0
            posgp[9] = 0.0
            posgp[10] = 0.0
            posgp[11] = 0.5
            posgp[12] = 1.0
            posgp[13] = 0.5
            posgp[14] = 1.0/3.0
            
            weigp[1] = 1.0/40.0
            weigp[2] = 1.0/15.0
            weigp[3] = 1.0/40.0
            weigp[4] = 1.0/15.0
            weigp[5] = 1.0/40.0
            weigp[6] = 1.0/15.0
            weigp[7] = 9.0/40.0
            
        end
    end

    if (nnode != 3)
        if(ngaus == 2)
            posgp[1] = -0.57735026918963
            weigp[1] = 1.0
        end
        if(ngaus > 2)
            posgp[1] = -0.7745966241483
            posgp[2] = 0.0
            weigp[1] = 0.55555555555556
            weigp[2] = 0.88888888888889
        end
        kgaus =  Int(floor(ngaus/2))
        print(kgaus)
        for igash = 1:kgaus
            jgash = ngaus+1-igash
            posgp[jgash] = -posgp[igash]
            weigp[jgash] = weigp[igash]
        end
    end   
    return posgp, weigp
end

"""
    cart_deriv(npoin, nelem, nnode, nstre, ndime, ndofn,
    ngaus, ntype, lnods, coord, posgp, weigp)


This function precalculats the Cartesian derivatives
of shape functions at all integration points. In addition,
function also calculates the area/volume contributions.
"""
function cart_deriv(npoin, nelem, nnode, nstre, ndime, ndofn,
    ngaus, ntype, lnods, coord, posgp, weigp)

    ngaus2 = ngaus
    if (nnode == 3)
        ngaus2 = 1
    end

    mgaus = ngaus * ngaus2
    dvolum = zeros(nelem, mgaus)

    dgdx = zeros(nelem, mgaus, ndime, nnode)
    elcod = zeros(ndime, nnode)
    for ielem = 1:nelem
        for inode = 1:nnode
            lnode = lnods[ielem,inode]
            for idime = 1:ndime
                elcod[idime,inode] = coord[lnode,idime]
            end
        end

        # gauss points
        kgasp = 0
        for igaus = 1:ngaus
            exisp = posgp[igaus]
            for jgaus = 1:ngaus2
                etasp = posgp[jgaus]
                if (nnode == 3)
                    etasp = posgp[ngaus+igaus]
                end

                kgasp = kgasp + 1
                mgaus = mgaus + 1

                shape, deriv = srf2(exisp, etasp, nnode)
                cartd, djacb, gpcod = jacob3(ielem, elcod, kgasp, shape, deriv, nnode, ndime)

                dvolu = djacb*weigp[igaus]*weigp[jgaus]
                if (nnode == 3)
                    dvolu = djacb*weigp[igaus]
                end
                dvolum[ielem, kgasp] = dvolu

                for idime = 1:ndime
                    for inode = 1:nnode
                        dgdx[ielem, kgasp, idime, inode] = cartd[idime,inode]
                    end
                end
            end
        end
    end

    return (dgdx, dvolum)
end

"""
    srf2(exisp, etasp, nnode)
This function evaluates the values of shape
function and their derivatives in local coordinates.

"""
function srf2(exisp, etasp, nnode)
    deriv = zeros(2, nnode)
    shape = zeros(nnode)

    ## triangular element
    if (nnode == 3)
        s = exisp
        t = etasp
        p = 1.0 - s - t

        shape[1] = p
        shape[2] = s
        shape[3] = t
        deriv[1,1] = -1.0
        deriv[1,2] = 1.0
        deriv[1,3] = 0.0
        deriv[2,1] = -1.0
        deriv[2,2] = 0.0
        deriv[2,3] = 1.0
    end

    ## quad element
    if (nnode == 4)
        s = exisp
        t = etasp
        st = s*t

        shape[1] = (1.0-t-s+st) *0.25
        shape[2] = (1.0-t+s-st) *0.25
        shape[3] = (1.0+t+s+st) *0.25
        shape[4] = (1.0+t-s-st) *0.25
 
        deriv[1,1] = (-1.0+t) * 0.25
        deriv[1,2] = ( 1.0-t) * 0.25
        deriv[1,3] = ( 1.0+t) * 0.25
        deriv[1,4] = (-1.0-t) * 0.25

        deriv[2,1] = (-1.0+s) * 0.25
        deriv[2,2] = (-1.0-s) * 0.25
        deriv[2,3] = ( 1.0+s) * 0.25
        deriv[2,4] = ( 1.0-s) * 0.25
    end

    ## quadratic quad element
    if (nnode == 8)
        s = exisp
        t = etasp

        s2 = 2.0*s
        t2 = 2.0*t

        ss = s*s
        tt = t*t
        st = s*t
        sst = s*s*t
        stt = s*t*t
        st2 = 2.0*s*t

        shape[1] = (-1.0+st+ss+tt-sst-stt) * 0.25
        shape[2] = ( 1.0-t-ss+sst) * 0.5
        shape[3] = (-1.0-st+ss+tt-sst+stt) * 0.25
        shape[4] = ( 1.0+s-tt-stt) * 0.5
        shape[5] = (-1.0+st+ss+tt+sst+stt) * 0.25
        shape[6] = ( 1.0+t-ss-sst) * 0.5
        shape[7] = (-1.0-st+ss+tt+sst-stt) * 0.25
        shape[8] = ( 1.0-s-tt+stt) * 0.5

        deriv[1,1] = ( t+s2-st2-tt) * 0.25
        deriv[1,2] =  -s+st
        deriv[1,3] = (-t+s2-st2+tt) * 0.25
        deriv[1,4] = ( 1.0-tt) * 0.5
        deriv[1,5] = ( t+s2+st2+tt) * 0.25
        deriv[1,6] =  -s-st
        deriv[1,7] = (-t+s2+st2-tt) * 0.25
        deriv[1,8] = (-1.0+tt) * 0.5

        deriv[2,1] = ( s+t2-ss-st2) * 0.25
        deriv[2,2] = (-1.0+ss) * 0.5
        deriv[2,3] = (-s+t2-ss+st2) * 0.25
        deriv[2,4] =  -t-st
        deriv[2,5] = ( s+t2+ss+st2) * 0.25
        deriv[2,6] = ( 1.0-ss) * 0.5
        deriv[2,7] = (-s+t2+ss-st2) * 0.25
        deriv[2,8] = -t + st

    end
    return (shape, deriv)
end

function jacob3(ielem, elcod, kgasp, shape, deriv, nnode, ndime)

    gpcod = zeros(ndime,kgasp)
    xjaci = zeros(2,2)

    ## gauss point coordinates
    cg = elcod * shape
    gpcod[1,kgasp] = cg[1]
    gpcod[2,kgasp] = cg[2]

    ## jacobian
    xjacm = deriv*elcod'

    # determinate of jacobian
    djacb = xjacm[1,1] * xjacm[2,2] - xjacm[1,2] * xjacm[2,1]

    if (djacb <= 0.0)
        println("Element No:", ielem)
        error("Program terminated zero or negative area")
    end

    # @ Cartesian derivatives
    xjaci[1,1] = xjacm[2,2] / djacb
    xjaci[2,2] = xjacm[1,1] / djacb
    xjaci[1,2] = -xjacm[1,2] / djacb
    xjaci[2,1] = -xjacm[2,1] / djacb

    cartd = xjaci * deriv
    return(cartd, djacb, gpcod)
end


function recover_slave_dof!(asdis, ncountm, ncounts,
    master, slave, npoin, ndofn)
    
    for ipbc = 1:ncountm
        im = master[ipbc]
        is = slave[ipbc]

        asdis[is] = asdis[im]
    end
    #return asdis
end

using Printf
function write_vtk_fem(npoin, nelem, nnode, lnods, coord, istep, cont1)

    # open file 
    fname = "time_$(istep).vtk" 
    out = open(fname,"w")

    # ---- start writing
    # header
    @printf(out, "# vtk DataFile Version 2.0\n")
    @printf(out, "time_10.vtk\n")
    @printf(out, "ASCII\n")
    @printf(out, "DATASET UNSTRUCTURED_GRID\n")

    # write nodal coordinates
    @printf(out, "POINTS %5d float\n",npoin)
    dummy = 0.0

    for ipoin = 1:npoin
        @printf(out, "%14.6f %14.6f %14.6f\n", coord[ipoin,1],coord[ipoin,2], dummy)
    end

    # write element connectivity
    iconst1 = nelem * (nnode + 1)

    @printf(out,"CELLS %5d %5d\n", nelem, iconst1)

    for ielem = 1: nelem
        @printf(out, "%5d", nnode)
        for inode = 1:nnode
            @printf(out, "%5d", (lnods[ielem,inode]-1))
        end
        @printf(out,"\n")
    end

    # ---- write cell types

    if (nnode == 8)
        ntype = 23;
    end

    if (nnode == 4)
        ntype = 9
    end

    if (nnode == 3)
        ntype = 5
    end

    @printf(out, "CELL_TYPES %5d\n",nelem)

    for i = 1:nelem
        @printf(out, "%2d\n", ntype)
    end

    # ---- write nodal scalar and vector values
    @printf(out, "POINT_DATA %5d\n",npoin)

    # ----- write concentration values as scalar:
    @printf(out, "SCALARS Con float 1\n")

    @printf(out, "LOOKUP_TABLE default\n")

    for ipoin = 1:npoin
        @printf(out,"%14.6e\n", cont1[ipoin])
    end

    close(out)
end

