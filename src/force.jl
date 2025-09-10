# See https://arxiv.org/pdf/1401.1181.pdf for applying forces to atoms
# See OpenMM documentation and Gromacs manual for other aspects of forces

export
    accelerations,
    force,
    pairwise_force,
    build_shift_vectors,
    pbc_dx_aiuc,
    SpecificForce1Atoms,
    SpecificForce2Atoms,
    SpecificForce3Atoms,
    SpecificForce4Atoms,
    forces

"""
    accelerations(system, neighbors=find_neighbors(sys), step_n=0; n_threads=Threads.nthreads())

Calculate the accelerations of all atoms in a system using the pairwise,
specific and general interactions and Newton's second law of motion.
"""
function accelerations(sys; n_threads::Integer=Threads.nthreads())
    return accelerations(sys, find_neighbors(sys; n_threads=n_threads); n_threads=n_threads)
end

function accelerations(sys, neighbors, step_n::Integer=0; n_threads::Integer=Threads.nthreads())
    return forces(sys, neighbors, step_n; n_threads=n_threads) ./ masses(sys)
end

"""
    force(inter, vec_ij, atom_i, atom_j, force_units, special, coord_i, coord_j,
          boundary, velocity_i, velocity_j, step_n)
    force(inter, coord_i, boundary, atom_i, force_units, velocity_i, step_n)
    force(inter, coord_i, coord_j, boundary, atom_i, atom_j, force_units, velocity_i,
          velocity_j, step_n)
    force(inter, coord_i, coord_j, coord_k, boundary, atom_i, atom_j, atom_k,
          force_units, velocity_i, velocity_j, velocity_k, step_n)
    force(inter, coord_i, coord_j, coord_k, coord_l, boundary, atom_i, atom_j, atom_k,
          atom_l, force_units, velocity_i, velocity_j, velocity_k, velocity_l, step_n)

Calculate the force between atoms due to a given interaction type.

For pairwise interactions returns a single force vector and for specific interactions
returns a type such as [`SpecificForce2Atoms`](@ref).
Custom pairwise and specific interaction types should implement this function.
"""
function force end

# Allow GPU-specific force functions to be defined if required
force_gpu(inter, dr, ai, aj, fu, sp, ci, cj, bnd, vi, vj, sn) = force(inter, dr, ai, aj, fu, sp, ci, cj, bnd, vi, vj, sn)
force_gpu(inter, ci, bnd, ai, fu, vi, sn) = force(inter, ci, bnd, ai, fu, vi, sn)
force_gpu(inter, ci, cj, bnd, ai, aj, fu, vi, vj, sn) = force(inter, ci, cj, bnd, ai, aj, fu, vi, vj, sn)
force_gpu(inter, ci, cj, ck, bnd, ai, aj, ak, fu, vi, vj, vk, sn) = force(inter, ci, cj, ck, bnd, ai, aj, ak, fu, vi, vj, vk, sn)
force_gpu(inter, ci, cj, ck, cl, bnd, ai, aj, ak, al, fu, vi, vj, vk, vl, sn) = force(inter, ci, cj, ck, cl, bnd, ai, aj, ak, al, fu, vi, vj, vk, vl, sn)

"""
    pairwise_force(inter, r, params)

Calculate the force magnitude between two atoms separated by distance `r` due to a
pairwise interaction.

This function is used in [`force`](@ref) to apply cutoff strategies by calculating
the force at different values of `r`.
Consequently, the parameters `params` should not include terms that depend on distance.
"""
function pairwise_force end

"""
    SpecificForce1Atoms(f1)

Force on one atom arising from an interaction such as a position restraint.
"""
struct SpecificForce1Atoms{D, T}
    f1::SVector{D, T}
end

"""
    SpecificForce2Atoms(f1, f2)

Forces on two atoms arising from an interaction such as a bond potential.
"""
struct SpecificForce2Atoms{D, T}
    f1::SVector{D, T}
    f2::SVector{D, T}
end

"""
    SpecificForce3Atoms(f1, f2, f3)

Forces on three atoms arising from an interaction such as a bond angle potential.
"""
struct SpecificForce3Atoms{D, T}
    f1::SVector{D, T}
    f2::SVector{D, T}
    f3::SVector{D, T}
end

"""
    SpecificForce4Atoms(f1, f2, f3, f4)

Forces on four atoms arising from an interaction such as a torsion potential.
"""
struct SpecificForce4Atoms{D, T}
    f1::SVector{D, T}
    f2::SVector{D, T}
    f3::SVector{D, T}
    f4::SVector{D, T}
end

function SpecificForce1Atoms(f1::StaticArray{Tuple{D}, T}) where {D, T}
    return SpecificForce1Atoms{D, T}(f1)
end

function SpecificForce2Atoms(f1::StaticArray{Tuple{D}, T}, f2::StaticArray{Tuple{D}, T}) where {D, T}
    return SpecificForce2Atoms{D, T}(f1, f2)
end

function SpecificForce3Atoms(f1::StaticArray{Tuple{D}, T}, f2::StaticArray{Tuple{D}, T},
                            f3::StaticArray{Tuple{D}, T}) where {D, T}
    return SpecificForce3Atoms{D, T}(f1, f2, f3)
end

function SpecificForce4Atoms(f1::StaticArray{Tuple{D}, T}, f2::StaticArray{Tuple{D}, T},
                            f3::StaticArray{Tuple{D}, T}, f4::StaticArray{Tuple{D}, T}) where {D, T}
    return SpecificForce4Atoms{D, T}(f1, f2, f3, f4)
end

Base.:+(x::SpecificForce1Atoms, y::SpecificForce1Atoms) = SpecificForce1Atoms(x.f1 + y.f1)
Base.:+(x::SpecificForce2Atoms, y::SpecificForce2Atoms) = SpecificForce2Atoms(x.f1 + y.f1, x.f2 + y.f2)
Base.:+(x::SpecificForce3Atoms, y::SpecificForce3Atoms) = SpecificForce3Atoms(x.f1 + y.f1, x.f2 + y.f2, x.f3 + y.f3)
Base.:+(x::SpecificForce4Atoms, y::SpecificForce4Atoms) = SpecificForce4Atoms(x.f1 + y.f1, x.f2 + y.f2, x.f3 + y.f3, x.f4 + y.f4)

struct ForcesBufferCPU{F, A, V, VC}
    fs_nounits::F
    fs_chunks::A
    vir_nounits::V
    vir_chunks::VC
end

# --- GROMACS-style virial helpers (CPU) ---

"""
    build_shift_vectors(boundary; dBox=(1,1,1))

Precompute translation vectors for PBC images within ±dBox in each dimension.
Returns (shifts, index_map, central_index) where `index_map` maps integer
shift tuples (k,l,m) to indices in `shifts` and `central_index` corresponds
to the zero shift (0,0,0). For 2D, tuples are (k,l).
"""
function build_shift_vectors(boundary::AbstractBoundary{D}; dBox=ntuple(_->1, D)) where D
    T = eltype(eltype(boundary isa TriclinicBoundary ? boundary.basis_vectors : box_sides(boundary)))
    shifts = SVector{D,T}[]
    index_map = Dict{NTuple{D,Int},Int}()
    central_index = 0
    # Basis vectors per dimension
    basis = if boundary isa TriclinicBoundary
        boundary.basis_vectors
    else
        # Rectangular/Cubic – axis-aligned box sides
        bs = box_sides(boundary)
        D == 3 ? (SVector(bs[1],0*bs[1],0*bs[1]), SVector(0*bs[2],bs[2],0*bs[2]), SVector(0*bs[1],0*bs[1],bs[1])) :
                  (SVector(bs[1],0*bs[1]), SVector(0*bs[2],bs[2]))
    end
    # Iterate k,l,(m)
    ranges = ntuple(d -> -dBox[d]:dBox[d], D)
    idx = 0
    for inds in Iterators.product(ranges...)
        idx += 1
        v = zero(first(basis))
        for d in 1:D
            v += inds[d] * basis[d]
        end
        push!(shifts, v)
        index_map[Tuple(inds)] = idx
        if all(i -> i == 0, inds)
            central_index = idx
        end
    end
    return shifts, index_map, central_index
end

"""
    pbc_dx_aiuc(boundary, xi, xj)

Return the minimum-image displacement `dx` from `xi` to `xj` and the integer
shift tuple used (k,l,m) s.t. `xj - xi + shift = dx`, following Molly’s
vector() logic per boundary type.
"""
function pbc_dx_aiuc(b::CubicBoundary, xi, xj)
    δ = xj - xi
    L = b.side_lengths
    k = ntuple(d -> Int(floor(ustrip(δ[d] / L[d]) + 0.5)), 3)
    shift = SVector{3}(k[1]*L[1], k[2]*L[2], k[3]*L[3])
    dx = δ - shift
    return dx, k
end

function pbc_dx_aiuc(b::RectangularBoundary, xi, xj)
    δ = xj - xi
    L = b.side_lengths
    k = ntuple(d -> Int(floor(ustrip(δ[d] / L[d]) + 0.5)), 2)
    shift = SVector{2}(k[1]*L[1], k[2]*L[2])
    dx = δ - shift
    return dx, k
end

function pbc_dx_aiuc(b::TriclinicBoundary{3}, xi, xj)
    δ = xj - xi
    # Use the same approximation as vector(c1,c2,b) for images
    n3 = Int(floor(ustrip(δ[3] * b.reciprocal_size[3]) + 0.5))
    δ -= b.basis_vectors[3] * n3
    n2 = Int(floor(ustrip(δ[2] * b.reciprocal_size[2]) + 0.5))
    δ -= b.basis_vectors[2] * n2
    n1 = Int(floor(ustrip(δ[1] * b.reciprocal_size[1]) + 0.5))
    δ -= b.basis_vectors[1] * n1
    # shift added to xj to get dx from xi to nearest image of xj
    k = (n1, n2, n3)
    return δ, k
end

function init_forces_buffer!(sys, n_threads)
    fs_nounits  = ustrip_vec.(zero(sys.coords))
    vir_nounits = ustrip_vec.(zero(sys.virial))
    # Enzyme errors with nothing when n_threads is 1
    n_copies = (n_threads == 1 ? 0 : n_threads)
    fs_chunks  = [similar(fs_nounits) for _ in 1:n_copies]
    vir_chunks = [similar(vir_nounits) for _ in 1:n_copies] 
    return ForcesBufferCPU(fs_nounits, fs_chunks, vir_nounits, vir_chunks)
end

struct ForcesBufferGPU{F, V, VS, C, M, R}
    fs_mat::F
    virial_row_1::V
    virial_row_2::V
    virial_row_3::V
    virial_specific::VS
    box_mins::C
    box_maxs::C
    morton_seq::M
    morton_seq_buffer_1::M
    morton_seq_buffer_2::M
    compressed_eligible::R
    compressed_special::R
end

function init_forces_buffer!(sys::System{D, AT, T}, n_threads,
                             for_pe::Bool=false) where {D, AT <: AbstractGPUArray, T}
    N = length(sys)
    C = eltype(eltype(sys.coords))
    n_blocks = cld(N, 32)
    backend = get_backend(sys.coords)
    fs_mat          = KernelAbstractions.zeros(backend, T, D, N)
    virial_row_1    = KernelAbstractions.zeros(backend, T, D, N)
    virial_row_2    = KernelAbstractions.zeros(backend, T, D, N)
    virial_row_3    = KernelAbstractions.zeros(backend, T, D, N)
    virial_specific = KernelAbstractions.zeros(backend, T, D, D)
    box_mins = KernelAbstractions.zeros(backend, C, n_blocks, D)
    box_maxs = KernelAbstractions.zeros(backend, C, n_blocks, D)
    morton_seq = KernelAbstractions.zeros(backend, Int32, N)
    morton_seq_buffer_1 = KernelAbstractions.zeros(backend, Int32, N)
    morton_seq_buffer_2 = KernelAbstractions.zeros(backend, Int32, N)
    compressed_eligible = KernelAbstractions.zeros(backend, UInt32, 32, n_blocks, n_blocks)
    compressed_special = KernelAbstractions.zeros(backend, UInt32, 32, n_blocks, n_blocks)
    if !for_pe && sys.neighbor_finder isa GPUNeighborFinder
        sys.neighbor_finder.initialized = false
    end
    return ForcesBufferGPU(fs_mat, virial_row_1, virial_row_2, virial_row_3, virial_specific, box_mins, box_maxs, morton_seq, morton_seq_buffer_1,
                           morton_seq_buffer_2, compressed_eligible, compressed_special)
end

zero_forces(sys) = ustrip_vec.(zero(sys.coords)) .* sys.force_units

"""
    forces(system, neighbors=find_neighbors(sys), step_n=0; n_threads=Threads.nthreads())

Calculate the forces on all atoms in a system using the pairwise, specific and
general interactions. This call also populates the [`virial`](@ref) tensor of
the system.
"""
function forces(sys; Virial::Bool = false, n_threads::Integer=Threads.nthreads())
    return forces(sys, find_neighbors(sys; n_threads=n_threads); Virial = Virial, n_threads=n_threads)
end

function forces(sys, neighbors, step_n::Integer=0; Virial::Bool = false, n_threads::Integer=Threads.nthreads())
    forces_buffer = init_forces_buffer!(sys, n_threads)
    fs = zero_forces(sys)
    forces!(fs, sys, neighbors, forces_buffer, Val(Virial), step_n; n_threads=n_threads)
    return fs
end

function forces!(fs, sys::System{D, AT, T}, neighbors, buffers, ::Val{Virial}, step_n::Integer=0;
                 n_threads::Integer=Threads.nthreads()) where {D, AT, T, Virial}

    #=
    Virial calculation follows the LAMMPS definition based on
    per-atom forces. We first assemble the complete force vector
    on all atoms, then compute the virial tensor as
    W = Σ rᵢ ⊗ fᵢ. See:
      https://docs.lammps.org/compute_stress_atom.html
    =#
    fill!(sys.virial, zero(T)*sys.energy_units)
    fill!(sys.kin_tensor, zero(T)*sys.energy_units)
    fill!(sys.pres_tensor, zero(T)*(sys.energy_units == NoUnits ? zero(T) : u"bar"))

    pairwise_inters_nonl = filter(!use_neighbors, values(sys.pairwise_inters))
    pairwise_inters_nl   = filter( use_neighbors, values(sys.pairwise_inters))
    sils_1_atoms = filter(il -> il isa InteractionList1Atoms, values(sys.specific_inter_lists))
    sils_2_atoms = filter(il -> il isa InteractionList2Atoms, values(sys.specific_inter_lists))
    sils_3_atoms = filter(il -> il isa InteractionList3Atoms, values(sys.specific_inter_lists))
    sils_4_atoms = filter(il -> il isa InteractionList4Atoms, values(sys.specific_inter_lists))

    # Build shift vectors and fshift accumulator (GROMACS-style)
    shifts, shift_index_map, central_shift_index = build_shift_vectors(sys.boundary)
    fshift_nounits = [zero(eltype(buffers.fs_nounits)) for _ in 1:length(shifts)]

    if length(sys.pairwise_inters) > 0
        pairwise_forces_loop!(buffers.fs_nounits, buffers.fs_chunks, buffers.vir_nounits, buffers.vir_chunks,
                        sys.atoms, sys.coords, sys.velocities, sys.boundary, neighbors, sys.force_units,
                        length(sys), pairwise_inters_nonl, pairwise_inters_nl,
                        Val(n_threads), Val(false), step_n;
                        fshift_nounits=fshift_nounits, shift_index_map=shift_index_map,
                        central_shift_index=central_shift_index)
    else
        fill!(buffers.fs_nounits, zero(eltype(buffers.fs_nounits)))
    end

    if length(sys.specific_inter_lists) > 0
        specific_forces!(buffers.fs_nounits, buffers.vir_nounits, sys.atoms, sys.coords, sys.velocities, sys.boundary,
                         sys.force_units, sils_1_atoms, sils_2_atoms, sils_3_atoms, sils_4_atoms, Val(false),
                         step_n; fshift_nounits=fshift_nounits, shift_index_map=shift_index_map,
                         central_shift_index=central_shift_index)
    end

    fs         .= buffers.fs_nounits  .* sys.force_units

    for inter in values(sys.general_inters)
        AtomsCalculators.forces!(fs, sys, inter; neighbors=neighbors, step_n=step_n,
                                 n_threads=n_threads)
    end

    if Virial
        # Compute virial after full force vector is available (GROMACS-style)
        V = zero(sys.virial); fill!(V, zero(eltype(V)))
        # Shift-force contribution: Σ s ⊗ fshift[s]
        @inbounds for s in eachindex(shifts)
            V .+= uconvert.(sys.energy_units,
                            shifts[s] * transpose(fshift_nounits[s] .* sys.force_units))
        end
        # Per-atom contribution: Σ r_i ⊗ f_i
        @inbounds for i in eachindex(fs)
            V .+= sys.coords[i] * transpose(fs[i])
        end
        sys.virial .= V
    end

    return fs
end

function pairwise_forces_loop!(fs_nounits, fs_chunks, vir_nounits, vir_chunks, atoms, coords, velocities, boundary,
                               neighbors, force_units, n_atoms, pairwise_inters_nonl,
                               pairwise_inters_nl, ::Val{1}, ::Val{Virial}, step_n=0;
                               fshift_nounits=nothing, shift_index_map=nothing, central_shift_index=nothing) where Virial
    fill!(fs_nounits, zero(eltype(fs_nounits)))
    fill!(vir_nounits, zero(eltype(vir_nounits)))

    FT = eltype(coords)

    @inbounds if length(pairwise_inters_nonl) > 0
        for i in 1:n_atoms
            for j in (i + 1):n_atoms
                dr, kt = pbc_dx_aiuc(boundary, coords[i], coords[j])
                f = force(pairwise_inters_nonl[1], dr, atoms[i], atoms[j], force_units, false,
                          coords[i], coords[j], boundary, velocities[i], velocities[j], step_n)
                for inter in pairwise_inters_nonl[2:end]
                    f += force(inter, dr, atoms[i], atoms[j], force_units, false, coords[i],
                               coords[j], boundary, velocities[i], velocities[j], step_n)
                end
                check_force_units(f, force_units)
                f_ustrip = ustrip.(f)
                fs_nounits[i] -= f_ustrip
                fs_nounits[j] += f_ustrip
                if fshift_nounits !== nothing
                    idx = shift_index_map[Tuple(kt)]
                    fshift_nounits[idx]                 += f_ustrip
                    fshift_nounits[central_shift_index] -= f_ustrip
                end
            end
        end
    end

    @inbounds if length(pairwise_inters_nl) > 0
        if isnothing(neighbors)
            error("an interaction uses the neighbor list but neighbors is nothing")
        end
        for ni in eachindex(neighbors)
            i, j, special = neighbors[ni]
            dr, kt = pbc_dx_aiuc(boundary, coords[i], coords[j])
            f = force(pairwise_inters_nl[1], dr, atoms[i], atoms[j], force_units, special,
                      coords[i], coords[j], boundary, velocities[i], velocities[j], step_n)
            for inter in pairwise_inters_nl[2:end]
                f += force(inter, dr, atoms[i], atoms[j], force_units, special, coords[i],
                           coords[j], boundary, velocities[i], velocities[j], step_n)
            end
            check_force_units(f, force_units)
            f_ustrip = ustrip.(f)
            fs_nounits[i] -= f_ustrip
            fs_nounits[j] += f_ustrip
            if fshift_nounits !== nothing
                idx = shift_index_map[Tuple(kt)]
                fshift_nounits[idx]                 += f_ustrip
                fshift_nounits[central_shift_index] -= f_ustrip
            end

        end
    end

    return fs_nounits
end

function pairwise_forces_loop!(fs_nounits, fs_chunks, vir_nounits, vir_chunks, atoms, coords, velocities, boundary,
                               neighbors, force_units, n_atoms, pairwise_inters_nonl,
                               pairwise_inters_nl, ::Val{n_threads}, ::Val{Virial}, step_n=0;
                               fshift_nounits=nothing, shift_index_map=nothing, central_shift_index=nothing) where {n_threads, Virial}

    FT  = eltype(fs_nounits)
    FTv = eltype(vir_nounits)

    if isnothing(fs_chunks) || isnothing(vir_chunks)
        throw(ArgumentError("fs_chunks / vir_chunks is not set but n_threads is > 1"))
    end
    if (length(fs_chunks) != n_threads) || (length(vir_chunks) != n_threads)
        throw(ArgumentError("length of fs_chunks = ($(length(fs_chunks))) or vir_chunks = ($(length(vir_chunks))) does not " *
                            "match n_threads ($n_threads)"))
    end
    @inbounds for chunk_i in 1:n_threads
        fill!(fs_chunks[chunk_i],  zero(FT))
        fill!(vir_chunks[chunk_i], zero(FTv))
    end

    # local per-thread fshift accumulators
    fshift_chunks = nothing
    if fshift_nounits !== nothing
        fshift_chunks = [fill(zero(eltype(fshift_nounits)), length(fshift_nounits)) for _ in 1:n_threads]
    end

    @inbounds if length(pairwise_inters_nonl) > 0
        Threads.@threads for chunk_i in 1:n_threads
            for i in chunk_i:n_threads:n_atoms
                for j in (i + 1):n_atoms
                    dr, kt = pbc_dx_aiuc(boundary, coords[i], coords[j])
                    f = force(pairwise_inters_nonl[1], dr, atoms[i], atoms[j], force_units,
                              false, coords[i], coords[j], boundary, velocities[i],
                              velocities[j], step_n)
                    for inter in pairwise_inters_nonl[2:end]
                        f += force(inter, dr, atoms[i], atoms[j], force_units, false, coords[i],
                                   coords[j], boundary, velocities[i], velocities[j], step_n)
                    end
                    check_force_units(f, force_units)
                    f_ustrip = ustrip.(f)
                    fs_chunks[chunk_i][i] -= f_ustrip
                    fs_chunks[chunk_i][j] += f_ustrip
                    if fshift_chunks !== nothing
                        idx = shift_index_map[Tuple(kt)]
                        fshift_chunks[chunk_i][idx]                 += f_ustrip
                        fshift_chunks[chunk_i][central_shift_index] -= f_ustrip
                    end
                end
            end
        end
    end

    @inbounds if length(pairwise_inters_nl) > 0
        if isnothing(neighbors)
            error("an interaction uses the neighbor list but neighbors is nothing")
        end
        Threads.@threads for chunk_i in 1:n_threads
            for ni in chunk_i:n_threads:length(neighbors)
                i, j, special = neighbors[ni]
                dr, kt = pbc_dx_aiuc(boundary, coords[i], coords[j])
                f = force(pairwise_inters_nl[1], dr, atoms[i], atoms[j], force_units, special,
                          coords[i], coords[j], boundary, velocities[i], velocities[j], step_n)
                for inter in pairwise_inters_nl[2:end]
                    f += force(inter, dr, atoms[i], atoms[j], force_units, special, coords[i],
                               coords[j], boundary, velocities[i], velocities[j], step_n)
                end
                check_force_units(f, force_units)
                f_ustrip = ustrip.(f)
                fs_chunks[chunk_i][i] -= f_ustrip
                fs_chunks[chunk_i][j] += f_ustrip
                if fshift_chunks !== nothing
                    idx = shift_index_map[Tuple(kt)]
                    fshift_chunks[chunk_i][idx]                 += f_ustrip
                    fshift_chunks[chunk_i][central_shift_index] -= f_ustrip
                end

            end
        end
    end

    @inbounds fs_nounits  .= fs_chunks[1]
    if Virial
        @inbounds vir_nounits .= vir_chunks[1]
    end
    @inbounds for chunk_i in 2:n_threads
        fs_nounits  .+= fs_chunks[chunk_i]
        if Virial
            vir_nounits .+= vir_chunks[chunk_i]
        end
    end

    # Reduce fshift chunks
    if fshift_chunks !== nothing
        for chunk_i in 1:n_threads
            @inbounds for s in eachindex(fshift_nounits)
                fshift_nounits[s] += fshift_chunks[chunk_i][s]
            end
        end
    end

    return fs_nounits
end

function specific_forces!(fs_nounits, vir_nounits, atoms, coords, velocities, boundary, force_units,
                          sils_1_atoms, sils_2_atoms, sils_3_atoms, sils_4_atoms, ::Val{Virial}, step_n=0;
                          fshift_nounits=nothing, shift_index_map=nothing, central_shift_index=nothing) where Virial

    FT = eltype(vir_nounits)
    @inbounds for inter_list in sils_1_atoms
        for (i, inter) in zip(inter_list.is, inter_list.inters)
            
            r_i = coords[i]
            
            sf = force(inter, coords[i], boundary, atoms[i], force_units, velocities[i], step_n)
            check_force_units(sf.f1, force_units)
            fs_nounits[i] += ustrip.(sf.f1)

            # shift bookkeeping for 1-atom terms: use central shift only
            if fshift_nounits !== nothing
                fshift_nounits[central_shift_index] += ustrip.(sf.f1)
            end
        end
    end

    @inbounds for inter_list in sils_2_atoms
        for (i, j, inter) in zip(inter_list.is, inter_list.js, inter_list.inters)

            r_ij, kt = pbc_dx_aiuc(boundary, coords[i], coords[j])
            
            sf = force(inter, coords[i], coords[j], boundary, atoms[i], atoms[j], force_units,
                       velocities[i], velocities[j], step_n)
            check_force_units(sf.f1, force_units)
            check_force_units(sf.f2, force_units)
            fs_nounits[i] += ustrip.(sf.f1)
            fs_nounits[j] += ustrip.(sf.f2)

            if fshift_nounits !== nothing
                idx = shift_index_map[Tuple(kt)]
                fshift_nounits[idx]                 += ustrip.(sf.f2) # force on j
                fshift_nounits[central_shift_index] -= ustrip.(sf.f2)
            end

        end
    end

    @inbounds for inter_list in sils_3_atoms
        for (i, j, k, inter) in zip(inter_list.is, inter_list.js, inter_list.ks, inter_list.inters)

            r_ik, kt_i = pbc_dx_aiuc(boundary, coords[i], coords[k])
            r_jk, kt_j = pbc_dx_aiuc(boundary, coords[j], coords[k])

            sf = force(inter, coords[i], coords[j], coords[k], boundary, atoms[i], atoms[j],
                       atoms[k], force_units, velocities[i], velocities[j], velocities[k], step_n)
            check_force_units(sf.f1, force_units)
            check_force_units(sf.f2, force_units)
            check_force_units(sf.f3, force_units)
            fs_nounits[i] += ustrip.(sf.f1)
            fs_nounits[j] += ustrip.(sf.f2)
            fs_nounits[k] += ustrip.(sf.f3)

            if fshift_nounits !== nothing
                idx_i = shift_index_map[Tuple(kt_i)]
                idx_j = shift_index_map[Tuple(kt_j)]
                fshift_nounits[idx_i]                 += ustrip.(sf.f1)
                fshift_nounits[central_shift_index]   -= ustrip.(sf.f1)
                fshift_nounits[idx_j]                 += ustrip.(sf.f2)
                fshift_nounits[central_shift_index]   -= ustrip.(sf.f2)
            end

        end
    end

    @inbounds for inter_list in sils_4_atoms
        for (i, j, k, l, inter) in zip(inter_list.is, inter_list.js, inter_list.ks, inter_list.ls,
                                       inter_list.inters)

            r_il, kt_i = pbc_dx_aiuc(boundary, coords[i], coords[l])
            r_jl, kt_j = pbc_dx_aiuc(boundary, coords[j], coords[l])
            r_kl, kt_k = pbc_dx_aiuc(boundary, coords[k], coords[l])

            sf = force(inter, coords[i], coords[j], coords[k], coords[l], boundary, atoms[i],
                       atoms[j], atoms[k], atoms[l], force_units, velocities[i], velocities[j],
                       velocities[k], velocities[l], step_n)
            check_force_units(sf.f1, force_units)
            check_force_units(sf.f2, force_units)
            check_force_units(sf.f3, force_units)
            check_force_units(sf.f4, force_units)
            fs_nounits[i] += ustrip.(sf.f1)
            fs_nounits[j] += ustrip.(sf.f2)
            fs_nounits[k] += ustrip.(sf.f3)
            fs_nounits[l] += ustrip.(sf.f4)

            if fshift_nounits !== nothing
                idx_i = shift_index_map[Tuple(kt_i)]
                idx_j = shift_index_map[Tuple(kt_j)]
                idx_k = shift_index_map[Tuple(kt_k)]
                fshift_nounits[idx_i]                += ustrip.(sf.f1)
                fshift_nounits[central_shift_index]  -= ustrip.(sf.f1)
                fshift_nounits[idx_j]                += ustrip.(sf.f2)
                fshift_nounits[central_shift_index]  -= ustrip.(sf.f2)
                fshift_nounits[idx_k]                += ustrip.(sf.f3)
                fshift_nounits[central_shift_index]  -= ustrip.(sf.f3)
            end
        end
    end

    return fs_nounits
end

function forces!(fs, sys::System{D, AT, T}, neighbors, buffers, ::Val{Virial}, step_n::Integer=0;
                n_threads::Integer=Threads.nthreads()) where {D, AT <: AbstractGPUArray, T, Virial}

    #=
    Ensure the appropriate tensors are set to zero at the beginning of the force call.
    =#
    if Virial
        fill!(sys.virial, zero(T)*sys.energy_units)
    end
    fill!(sys.kin_tensor, zero(T)*sys.energy_units)
    fill!(sys.pres_tensor, zero(T)*(sys.energy_units == NoUnits ? zero(T) : u"bar"))
    
    #=
    Zero the buffers to store the calc. magnitudes. 
    =#
    fill!(buffers.fs_mat, zero(T))
    if Virial
        fill!(buffers.virial_row_1, zero(T))
        fill!(buffers.virial_row_2, zero(T))
        fill!(buffers.virial_row_3, zero(T))
        fill!(buffers.virial_specific, zero(T))
    end

    pairwise_inters_nonl = filter(!use_neighbors, values(sys.pairwise_inters))
    if length(pairwise_inters_nonl) > 0
        n = length(sys)
        nbs = NoNeighborList(n)
        # Compute only forces here; defer virial until after forces are assembled
        pairwise_forces_loop_gpu!(buffers, sys, pairwise_inters_nonl, nbs, step_n)
    end

    pairwise_inters_nl = filter(use_neighbors, values(sys.pairwise_inters))
    if length(pairwise_inters_nl) > 0
        # Compute only forces here; defer virial until after forces are assembled
        pairwise_forces_loop_gpu!(buffers, sys, pairwise_inters_nl, neighbors, Val(false), step_n)
    end

    for inter_list in values(sys.specific_inter_lists)
        # Compute only forces here; defer virial until after forces are assembled
        specific_forces_gpu!(buffers.fs_mat, buffers.virial_specific,
                            inter_list, sys.coords, sys.velocities, sys.atoms,
                            sys.boundary, Val(false), step_n, sys.force_units, Val(T))
    end

    fs          .= reinterpret(SVector{D, T}, vec(buffers.fs_mat)) .* sys.force_units

    for inter in values(sys.general_inters)
        AtomsCalculators.forces!(fs, sys, inter; neighbors=neighbors, step_n=step_n,
                                 n_threads=n_threads)
    end

    if Virial
        # Compute virial after full force vector is available: W = Σ r ⊗ f
        V = zero(sys.virial)
        fill!(V, zero(eltype(V)))
        coords_cpu = from_device(sys.coords)
        fs_cpu = from_device(fs)
        @inbounds for i in eachindex(fs_cpu)
            V .+= coords_cpu[i] * transpose(fs_cpu[i])
        end
        sys.virial .= uconvert.(sys.energy_units, V)
    end

    return fs
end
