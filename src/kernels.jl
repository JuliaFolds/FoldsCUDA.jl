_transduce_cuda(xf::Transducer, op, init, xs; kwargs...) =
    _transduce_cuda(xf'(op), init, xs; kwargs...)

function _transduce_cuda(op, init, xs;)
    xf0, coll = extract_transducer(xs)
    # TODO: more systematic approach to this (and also support product)
    if coll isa Iterators.Zip
        arrays = coll.is
        xf = xf0
    elseif coll isa Iterators.Pairs
        arrays = (keys(coll), values(coll))
        xf = xf0
    else
        arrays = (coll,)
        xf = opcompose(Map(first), xf0)
    end
    rf = _reducingfunction(xf, op; init = init)
    acc = transduce_impl(rf, init, arrays...)
    rf_dev = cudaconvert(rf)
    if rf_dev === rf
        result = complete(rf, acc)
    else
        result = complete_on_device(rf_dev, acc)
    end
    if unreduced(result) isa DefaultInitOf
        throw(EmptyResultError(rf))
    end
    return result
end

function transduce_impl(rf::F, init, arrays...) where {F}
    ys, = (dest, buf) = _transduce!(nothing, rf, init, arrays...)
    if buf === nothing
        # The accumulator is a singleton. Once we are finished with the
        # side-effects of the basecase, transduce is done:
        return ys
    end
    # @info "ys, = _transduce!(nothing, rf, ...)" Text(summary(ys))
    # @info "ys, = _transduce!(nothing, rf, ...)" collect(ys)
    length(ys) == 1 && return @allowscalar ys[1]
    rf2 = AlwaysCombine(rf)
    while true
        ys, = _transduce!(buf, rf2, CombineInit(), ys)
        # @info "ys, = _transduce!(buf, rf2, ...)" Text(summary(ys))
        # @info "ys, = _transduce!(buf, rf2, ...)" collect(ys)
        length(ys) == 1 && return @allowscalar ys[1]
        dest, buf = buf, dest
        # reusing buffer; is it useful?
    end
end

const _TRUE_ = Ref(true)

function fake_transduce(rf, xs, init, ::Val{IncludeInit} = Val(false)) where {IncludeInit}
    if IncludeInit
        if _TRUE_[]
            return start(rf, init)
        end
    end
    if _TRUE_[]
        acc1 = next(rf, start(rf, init), first(xs))
        for x in xs
            acc1 = next(rf, acc1, x)
        end
        return acc1
    else
        return _combine(rf, fake_transduce(rf, xs, init), fake_transduce(rf, xs, init))
    end
end

struct DisallowedElementTypeError{T} <: Exception end
Base.showerror(io::IO, ::DisallowedElementTypeError{T}) where {T} =
    print(io, "accumulator type must be `isbits` or `isbitsunion`; got: $T")

function allocate_buffer(::Type{T}, n) where {T}
    if isbitstype(T)
        return CuVector{T}(undef, n)
    elseif Base.isbitsunion(T)
        return UnionVector(undef, CuVector, T, n)
    else
        # TODO: Fallback to the mutate-or-widen appraoch? (e.g., run first
        # iteration on CPU, and then use it as the initial guess of the
        # accumulator?)
        throw(DisallowedElementTypeError{T}())
    end
end

Base.@propagate_inbounds getvalues(i) = ()
Base.@propagate_inbounds getvalues(i, a) = (a[i],)
Base.@propagate_inbounds getvalues(i, a, as...) = (a[i], getvalues(i, as...)...)

function _infer_acctype(rf, init, arrays, include_init::Bool = false)
    fake_args = (
        cudaconvert(rf),
        zip(map(cudaconvert, arrays)...),
        cudaconvert(init),
        Val(include_init),
    )
    fake_args_tt = Tuple{map(Typeof, fake_args)...}
    acctype = CUDA.return_type(fake_transduce, fake_args_tt)
    if acctype === Union{}
        host_args = (rf, zip(arrays...), init)
        acctype_host = Core.Compiler.return_type(fake_transduce, Tuple{map(Typeof, host_args)...})
        if RUN_ON_HOST_IF_NORETURN[] && acctype_host === Union{}
            fake_transduce(host_args...)
            error("unreachable: incorrect inference")
        end
        throw(FailedInference(fake_transduce, fake_args, acctype, host_args, acctype_host))
    end
    return acctype
    # Note: the result of `return_type` is not observable by the caller of the
    # API `transduce_impl`
end

function _transduce!(buf, rf::F, init, arrays...) where {F}
    idx = eachindex(arrays...)
    n = Int(length(idx))  # e.g., `length(UInt64(0):UInt64(1))` is not an `Int`

    wanted_threads = nextpow(2, n)
    compute_threads(max_threads) =
        wanted_threads > max_threads ? prevpow(2, max_threads) : wanted_threads

    acctype = if buf === nothing
        _infer_acctype(rf, init, arrays)
    else
        eltype(buf)
    end
    # @show acctype
    buf0 = if Base.issingletontype(acctype)
        nothing
    elseif buf === nothing
        # TODO: find a way to compute type for `cufunction` without
        # creating a dummy object.
        allocate_buffer(acctype, 0)
    else
        buf
    end
    args = (buf0, rf, init, 0, idx, arrays...)
    # global _KARGS = args
    kernel_tt = Tuple{map(x -> Typeof(cudaconvert(x)), args)...}
    # global KERNEL_TT = kernel_tt
    kernel = cufunction(transduce_kernel!, kernel_tt)
    effelsize = if isbitstype(acctype)
        sizeof(acctype)
    else
        sizeof(UnionArrays.buffereltypefor(acctype)) + sizeof(UInt8)
    end
    # @show acctype UnionArrays.buffereltypefor(acctype) effelsize
    compute_shmem(threads) = 2 * threads * effelsize
    kernel_config =
        launch_configuration(kernel.fun; shmem = compute_shmem ∘ compute_threads)
    threads = compute_threads(kernel_config.threads)
    shmem = compute_shmem(threads)

    basesize = cld(n, kernel_config.blocks * threads)
    blocks = cld(n, basesize * threads)
    @assert blocks <= kernel_config.blocks

    if Base.issingletontype(acctype)
        @cuda(
            threads = threads,
            blocks = blocks,
            shmem = shmem,
            transduce_kernel!(nothing, rf, init, basesize, idx, arrays...)
        )
        return acctype.instance, nothing
    end

    if buf === nothing
        dest_buf = allocate_buffer(acctype, blocks + cld(blocks, threads))
        dest = view(dest_buf, 1:blocks)
        buf = view(dest_buf, blocks+1:length(dest_buf))
    else
        dest = view(buf, 1:blocks)
    end
    # @show threads, blocks, shmem, basesize

    # global INVOKE_KERNEL = function ()
    #     @cuda(
    #         threads = threads,
    #         blocks = blocks,
    #         shmem = shmem,
    #         transduce_kernel!(dest, rf, init, basesize, idx, arrays...)
    #     )
    # end

    @cuda(
        threads = threads,
        blocks = blocks,
        shmem = shmem,
        transduce_kernel!(dest, rf, init, basesize, idx, arrays...)
    )

    return dest, buf
end

function transduce_kernel!(
    dest::Union{AbstractArray,Nothing},
    rf::F,
    init,
    basesize,
    idx,
    arrays...,
) where {F}

    # Use undef state of `acc` as an "extra Union"; i.e., treat as if the
    # initial iteration is unrolled, even though it may not be possible to do so
    # for all threads:
    local acc
    acc_isdefined = false
    let n = length(idx),
        offset = threadIdx().x - 1 + (blockIdx().x - 1) * blockDim().x,
        i1 = offset * basesize + 1,
        x1, xf
        if i1 <= n
            x1 = @inbounds getvalues(idx[i1], arrays...)
            @inline getinput(i) = @inbounds getvalues(idx[i], arrays...)
            xf = Map(getinput)
            acc = foldl_nocomplete(
                Reduction(xf, rf),
                next(rf, start(rf, init), x1),
                offset*basesize+2:min((offset + 1) * basesize, n),
            )
            acc_isdefined = true
        end
    end

    dest === nothing && return

    # NOTE: Here, `acc` may have a different type for each thread. Since the
    # following code contain `sync_threads()`, we cannot introduce any dispatch
    # bounary ("function barrier") here. Otherwise, since dispatch is just a
    # branch for the GPU, the resulting code tries to synchronize code across
    # different branches and hence deadlock.

    # shared mem for a complete reduction
    T = eltype(dest)
    if isbitstype(T)
        shared = @cuDynamicSharedMem(T, (2 * blockDim().x,))
    else
        S = UnionArrays.buffereltypefor(T)
        data = @cuDynamicSharedMem(S, (2 * blockDim().x,))
        offset = sizeof(S) * 2 * blockDim().x
        typeids = @cuDynamicSharedMem(UInt8, (2 * blockDim().x,), offset)
        @assert UInt(pointer(data, length(data) + 1)) == UInt(pointer(typeids))
        shared = UnionVector(T, data, typeids)
    end
    if acc_isdefined
        # Manual union splitting (required for non-type-stable reduction like
        # `Folds.sum(last, pairs(xs))`):
        @manual_union_split(
            isbitstype(T),
            acc isa UnionArrays.eltypebyid(shared, Val(1)),
            acc isa UnionArrays.eltypebyid(shared, Val(2)),
            acc isa UnionArrays.eltypebyid(shared, Val(3)),
            acc isa UnionArrays.eltypebyid(shared, Val(4)),
            acc isa UnionArrays.eltypebyid(shared, Val(5)),
            acc isa UnionArrays.eltypebyid(shared, Val(6)),
        ) do
            @inbounds shared[threadIdx().x] = acc
        end
    end

    # `iseven(m)` in the `while` loop below enforces that indexing on `shared`
    # is in bounds. But, for the last block we need to make sure to combine
    # accumulators only within the valid thread indices.
    bound = let n = length(idx),
        nbasecases = cld(n, basesize),
        offsetb = (blockIdx().x - 1) * blockDim().x
        max(0, nbasecases - offsetb)
    end

    m = threadIdx().x - 1
    t = threadIdx().x
    s = 1
    c = blockDim().x >> 1
    while c != 0
        sync_threads()
        if t + s <= bound && iseven(m)
            @inbounds shared[t] = _combine(rf, shared[t], shared[t+s])
            m >>= 1
        end
        s <<= 1
        c >>= 1
    end

    if t == 1
        @inbounds dest[blockIdx().x] =  shared[1]
    end

    return
end

struct CombineInit end

struct AlwaysCombine{I} <: AbstractReduction{I}
    inner::I
end

#=
AlwaysCombine(rf::Transducers.R_{Map}) = AlwaysCombine(Transducers.inner(rf))
AlwaysCombine(rf::Transducers.BottomRF) = AlwaysCombine(Transducers.inner(rf))
=#

@inline Transducers.start(rf::AlwaysCombine, init) = start(rf.inner, init)
@inline Transducers.start(::AlwaysCombine, init::CombineInit) = init
@inline Transducers.next(::AlwaysCombine, ::CombineInit, input) = first(input)
@inline Transducers.next(rf::F, acc, input) where {F<:AlwaysCombine} =
    _combine(rf.inner, acc, first(input))
@inline Transducers.complete(rf::F, result) where {F<:AlwaysCombine} =
    complete(rf.inner, result)
@inline Transducers.combine(rf::F, a, b) where {F<:AlwaysCombine} = _combine(rf.inner, a, b)

# Semantically correct but inefficient (eager) handling of `Reduced`.
@inline _combine(rf, a::Reduced, b::Reduced) = a
@inline _combine(rf, a::Reduced, b) = a
@inline _combine(rf::RF, a, b::Reduced) where {RF} = reduced(combine(rf, a, unreduced(b)))
@inline _combine(rf::RF, a, b) where {RF} = combine(rf, a, b)

# TODO: merge this into transduce_kernel!
function complete_kernel!(buf, rf, acc)
    buf[1] = complete(rf, acc)
    return
end

function complete_kernel!(rf, acc)
    complete(rf, acc)
    return
end

function complete_on_device(rf_dev::RF, acc::ACC) where {RF, ACC}
    # global CARGS = (rf_dev, acc)
    resulttype = CUDA.return_type(complete, Tuple{RF,ACC})
    if Base.issingletontype(resulttype)
        @cuda complete_kernel!(rf_dev, acc)
        return resulttype.instance
    end
    buf = allocate_buffer(resulttype, 1)
    @cuda complete_kernel!(buf, rf_dev, acc)
    return @allowscalar buf[1]
end
