transduce_cuda(xf::Transducer, op, init, xs; kwargs...) =
    transduce_cuda(xf'(op), init, xs; kwargs...)

function transduce_cuda(op, init, xs;)
    xf0, coll = extract_transducer(xs)
    if coll isa Iterators.Zip
        arrays = coll.is
        xf = xf0
    else
        arrays = (coll,)
        xf = opcompose(Map(first), xf0)
    end
    rf = _reducingfunction(xf, op; init = init)
    acc = transduce_impl(rf, init, arrays...)
    result = complete(rf, acc)
    if unreduced(result) isa DefaultInitOf
        throw(EmptyResultError(rf))
    end
    return result
end

function transduce_impl(rf::F, init, arrays...) where {F}
    ys, = (dest, buf) = _transduce!(nothing, rf, init, arrays...)
    # @info "ys, = _transduce!(nothing, rf, ...)" ys
    length(ys) == 1 && return @allowscalar ys[1]
    monoid = asmonoid(always_combine(rf))
    rf2 = Map(first)'(monoid)  # TODO: reduce wrapping
    while true
        ys, = _transduce!(buf, rf2, InitialValue(monoid), ys)
        # @info "ys, = _transduce!(buf, rf2, ...)" ys
        length(ys) == 1 && return @allowscalar ys[1]
        dest, buf = buf, dest
        # reusing buffer; is it useful?
    end
end

function fake_transduce(rf, xs, init)
    acc1 = next(rf, start(rf, init), first(xs))
    for x in xs
        acc1 = next(rf, acc1, x)
    end
    acc2 = acc1
    for x in xs
        acc2 = next(rf, acc2, x)
    end
    ys = [acc1, acc2]
    acc3 = acc2
    for y in ys
        acc3 = _combine(rf, acc3, y)
    end
    return acc3
end

Base.@propagate_inbounds getvalues(i) = ()
Base.@propagate_inbounds getvalues(i, a) = (a[i],)
Base.@propagate_inbounds getvalues(i, a, as...) = (a[i], getvalues(i, as...)...)

function _transduce!(buf, rf::F, init, arrays...) where {F}
    idx = eachindex(arrays...)
    n = length(idx)

    wanted_threads = nextpow(2, n)
    compute_threads(max_threads) =
        wanted_threads > max_threads ? prevpow(2, max_threads) : wanted_threads

    acctype = if buf === nothing
        # global _ARGS = (rf, zip(arrays...), init)
        # @show fake_transduce(rf, zip(arrays...), init)
        return_type(fake_transduce, Tuple{Typeof(rf),Typeof(zip(arrays...)),Typeof(init)})
        # Note: the result of `return_type` is not observable by the
        # caller of the API `transduce_impl`
    else
        eltype(buf)
    end
    # @show acctype
    buf0 = if buf === nothing
        # TODO: find a way to compute type for `cufunction` without
        # creating a dummy object.
        CuVector{acctype}(undef, 0)
    else
        buf
    end
    args = (buf0, rf, init, 0, idx, arrays...)
    # global _KARGS = args
    kernel_tt = Tuple{map(x -> Typeof(cudaconvert(x)), args)...}
    kernel = cufunction(transduce_kernel!, kernel_tt)
    compute_shmem(threads) = 2 * threads * sizeof(acctype)
    kernel_config =
        launch_configuration(kernel.fun; shmem = compute_shmem ∘ compute_threads)
    threads = compute_threads(kernel_config.threads)
    shmem = compute_shmem(threads)

    basesize = cld(n, kernel_config.blocks * threads)
    blocks = cld(n, basesize * threads)
    @assert blocks <= kernel_config.blocks

    if buf === nothing
        dest_buf = CuVector{acctype}(undef, blocks + cld(blocks, threads))
        dest = view(dest_buf, 1:blocks)
        buf = view(dest_buf, blocks+1:length(dest_buf))
    else
        dest = view(buf, 1:blocks)
    end
    # @show threads, blocks, shmem, basesize

    # TODO: do I need sync here?
    CUDA.@sync @cuda(
        threads = threads,
        blocks = blocks,
        shmem = shmem,
        transduce_kernel!(dest, rf, init, basesize, idx, arrays...)
    )

    return dest, buf
end

function transduce_kernel!(
    dest::AbstractArray{T},
    rf::F,
    init,
    basesize,
    idx,
    arrays...,
) where {F,T}
    acc = combineblock(
        rf,
        basecase(rf, init, idx, arrays, basesize),
        T,
        basesize,
        idx,
        arrays,
    )
    if threadIdx().x == 1
        @inbounds dest[blockIdx().x] = acc
    end
    return
end

@inline function basecase(rf::F, init, idx, arrays, basesize) where {F}
    n = length(idx)
    offset = threadIdx().x - 1 + (blockIdx().x - 1) * blockDim().x

    i1 = offset * basesize + 1
    if i1 <= n
        x1 = @inbounds getvalues(idx[i1], arrays...)
    else
        x1 = @inbounds getvalues(idx[1], arrays...) # random value (not used)
    end
    acc = next(rf, start(rf, init), x1)
    for i in offset*basesize+2:min((offset + 1) * basesize, n)
        x = @inbounds getvalues(idx[i], arrays...)
        acc = next(rf, acc, x)
    end

    return acc
end

@inline function combineblock(rf::F, acc, ::Type{T}, basesize, idx, arrays) where {F,T}
    n = length(idx)
    offsetb = (blockIdx().x - 1) * blockDim().x
    bound = max(0, n - offsetb * basesize)

    # shared mem for a complete reduction
    shared = @cuDynamicSharedMem(T, (2 * blockDim().x,))
    @inbounds shared[threadIdx().x] = acc

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
        acc = @inbounds shared[1]
    end

    return acc
end

function always_combine(rf::F) where {F}
    @inline function op(a, b)
        _combine(rf, a, b)
    end
    return op
end

# Semantically correct but inefficient (eager) handling of `Reduced`.
@inline _combine(rf, a::Reduced, b::Reduced) = a
@inline _combine(rf, a::Reduced, b) = a
@inline _combine(rf::RF, a, b::Reduced) where {RF} = reduced(combine(rf, a, unreduced(b)))
@inline _combine(rf::RF, a, b) where {RF} = combine(rf, a, b)
