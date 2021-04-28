transduce_shfl(xf::Transducer, op, init, xs; kwargs...) =
    transduce_shfl(xf'(op), init, xs; kwargs...)

function transduce_shfl(op, init, xs;)
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
    acc = transduce_shfl_impl(rf, init, arrays...)
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

macro _inbounds(ex)
    ex = :($Base.@inbounds $ex)
    esc(ex)
end

function transduce_shfl_impl(rf::F, init, arrays...) where {F}
    ys, = (dest, buf) = transduce_shfl!(nothing, rf, init, arrays...)
    if buf === nothing
        # The accumulator is a singleton. Once we are finished with the
        # side-effects of the basecase, transduce is done:
        return ys
    end
    # @info "ys, = transduce_shfl!(nothing, rf, ...)" Text(summary(ys))
    # @info "ys, = transduce_shfl!(nothing, rf, ...)" collect(ys)
    length(ys) == 1 && return @allowscalar ys[1]
    rf2 = AlwaysCombine(rf)
    @assert start(rf, init) === init
    while true
        ys, = transduce_shfl!(buf, rf2, init, ys)
        # @info "ys, = transduce_shfl!(buf, rf2, ...)" Text(summary(ys))
        # @info "ys, = transduce_shfl!(buf, rf2, ...)" collect(ys)
        length(ys) == 1 && return @allowscalar ys[1]
        dest, buf = buf, dest
        # reusing buffer; is it useful?
    end
end

@inline function transduce_shfl!(buf, rf, init, arrays...)
    idx = eachindex(arrays...)
    n = Int(length(idx))  # e.g., `length(UInt64(0):UInt64(1))` is not an `Int`

    dev = device()
    wsize = warpsize(dev)
    WARP_SIZE = Val(wsize)

    acctype = if buf === nothing
        # global _ARGS = (rf, zip(arrays...), init)
        # @show fake_transduce(rf, zip(arrays...), init)
        fake_args = (cudaconvert(rf), zip(map(cudaconvert, arrays)...), cudaconvert(init))
        fake_args_tt = Tuple{map(Typeof, fake_args)...}
        # global FAKE_ARGS = fake_args
        # global FAKE_ARGS_TT = fake_args_tt
        return_type(fake_transduce, fake_args_tt)
        # Note: the result of `return_type` is not observable by the
        # caller of the API `transduce_impl`
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
    args = (buf0, WARP_SIZE, rf, init, 0, idx, arrays...)
    # global _KARGS = args
    kernel_tt = Tuple{map(x -> Typeof(cudaconvert(x)), args)...}
    # global KERNEL_TT = kernel_tt
    kernel = cufunction(transduce_shfl_kernel!, kernel_tt)
    effelsize = if isbitstype(acctype)
        sizeof(acctype)
    else
        sizeof(UnionArrays.buffereltypefor(acctype)) + sizeof(UInt8)
    end
    # @show acctype UnionArrays.buffereltypefor(acctype) effelsize
    kernel_config = launch_configuration(kernel.fun)
    # @show kernel_config
    threads = let wanted_threads = nextwarp(dev, n)
        given_threads = kernel_config.threads
        # @show wanted_threads
        if wanted_threads > given_threads
            prevwarp(dev, given_threads)
        else
            wanted_threads
        end
    end

    @assert threads <= wsize * wsize  # = 32 * 32 = 1024

    nwarps_per_block, _nwarps_rem = divrem(threads, wsize)
    # @show threads nwarps_per_block _nwarps_rem
    @assert _nwarps_rem % wsize == 0
    basesize = max(wsize, cld(n, kernel_config.blocks * nwarps_per_block))
    blocks = cld(n, basesize * nwarps_per_block)
    @assert blocks <= kernel_config.blocks

    if Base.issingletontype(acctype)
        @cuda(
            threads = threads,
            blocks = blocks,
            transduce_shfl_kernel!(nothing, WARP_SIZE, rf, init, basesize, idx, arrays...)
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
    # @show threads, blocks, basesize

    @cuda(
        threads = threads,
        blocks = blocks,
        transduce_shfl_kernel!(dest, WARP_SIZE, rf, init, basesize, idx, arrays...)
    )

    return dest, buf
end

@inline function transduce_shfl_kernel!(
    dest::AbstractArray{T},
    ::Val{WARP_SIZE},
    rf::F,
    init,
    basesize,  # length of the main loop per warp
    idx,
    arrays...,
) where {T,WARP_SIZE,F}

    nwarps_per_block, _nwarps_rem = divrem(blockDim().x, WARP_SIZE)
    @assert _nwarps_rem == 0
    warpIdx0, warp_offset = divrem(threadIdx().x - 1, WARP_SIZE)
    warpIdx = warpIdx0 + 1
    warp_leader = 1 + warpIdx0 * WARP_SIZE  # first thread of this warp
    @assert warp_leader + warp_offset == threadIdx().x

    main_offset = warpIdx0 + (blockIdx().x - 1) * nwarps_per_block
    main_bound0 = basesize * (main_offset + 1)  # bound for this warp in `eachindex(idx)`
    need_remainder = main_bound0 > lastindex(idx)  # `idx` too short for the last block
    main_bound = min(main_bound0, lastindex(idx)) - WARP_SIZE
    need_remainder && @assert blockIdx().x == gridDim().x

    # if warp_offset == 0
    #     @cuprintf("%03ld: warpIdx0 = %ld main_offset = %d\n", threadIdx().x, Int(warpIdx0), Int(main_offset))
    # end

    # Main O(N) loop:
    acc = start(rf, init)
    warp_leader_offset = basesize * main_offset  # offset for this warp
    while warp_leader_offset <= main_bound
        i = warp_leader_offset + warp_offset + 1
        # @cuprintf("%03ld: i = %d\n", threadIdx().x, Int(i))
        acc = next(rf, acc, @_inbounds getvalues(idx[i], arrays...))

        # Warp-wide merge:
        delta = 1
        while delta < WARP_SIZE
            acc = _combine(rf, acc, shfl_down_sync(typemax(UInt32), acc, delta, WARP_SIZE))
            delta <<= 1
        end
        if warp_offset != 0
            acc = start(rf, init)
        end
        warp_leader_offset += WARP_SIZE
    end

    # Remainder of the main loop:
    if need_remainder
        let i = warp_leader_offset + warp_offset + 1
            # @cuprintf("%03ld: (rem) i = %d\n", threadIdx().x, Int(i))
            if i <= lastindex(idx)
                acc = next(rf, acc, @_inbounds getvalues(idx[i], arrays...))
            end

            delta = 1
            while delta < WARP_SIZE
                acc = _combine(rf, acc, shfl_down_sync(typemax(UInt32), acc, delta, WARP_SIZE))
                delta <<= 1
            end
        end
    end
    # @cuprintf("%03ld: acc = %f\n", threadIdx().x, acc)

    # Preparing for block-wide merge:
    @assert nwarps_per_block <= 32
    if isbitstype(T)
        shared = @cuStaticSharedMem(T, 32)
    else
        S = UnionArrays.buffereltypefor(T)
        data = @cuStaticSharedMem(S, 32)
        typeids = @cuStaticSharedMem(UInt8, 32)
        @assert UInt(pointer(data, length(data) + 1)) == UInt(pointer(typeids))
        shared = UnionVector(T, data, typeids)
    end
    if warp_offset == 0
        @_inbounds shared[warpIdx] = acc
    end

    shared_bound = let n = length(idx),
        nbasecases = cld(n, basesize),
        offsetb = (blockIdx().x - 1) * nwarps_per_block,
        input_bound = nbasecases - offsetb
        min(input_bound, nwarps_per_block)
    end

    # Block-wide merge:
    shared_delta = 1
    while shared_delta < nwarps_per_block

        # Gather `WARP_SIZE` elements:
        i = warp_leader + shared_delta * warp_offset
        acc = start(rf, init)

        sync_threads()
        if i <= shared_bound
            acc = @_inbounds shared[i]
        end

        # Warp-wide merge:
        delta = 1
        while delta < WARP_SIZE
            acc = _combine(rf, acc, shfl_down_sync(typemax(UInt32), acc, delta, WARP_SIZE))
            delta <<= 1
        end

        if warp_offset == 0 && threadIdx().x <= lastindex(shared)
            @_inbounds shared[threadIdx().x] = acc
        end

        shared_delta *= WARP_SIZE
    end

    if threadIdx().x == 1
        @_inbounds dest[blockIdx().x] =  acc
    end

    return
end
