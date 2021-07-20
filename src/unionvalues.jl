const NTypes{N} = NTuple{N, Val}

valueof(::Val{x}) where {x} = x

# Not exactly `Base.aligned_sizeof`
Base.@pure function sizeof_aligned(T::Type)
    if isbitstype(T)
        al = Base.datatype_alignment(T)
        return (Core.sizeof(T) + al - 1) & -al
    else
        return nothing
    end
end

@inline foldlargs(op, x) = x
@inline foldlargs(op, x1, x2, xs...) = foldlargs(op, @return_if_reduced(op(x1, x2)), xs...)

terminating_foldlargs(op, fallback) = fallback()
@inline function terminating_foldlargs(op, fallback::F, x1, x2, xs...) where {F}
    acc = op(x1, x2)
    acc isa Reduced && return unreduced(acc)
    xs isa Tuple{} && return fallback()
    return terminating_foldlargs(op, fallback, acc, xs...)
end

@inline foldrunion(op, ::Type{T}, init) where {T} =
    if T isa Union
        acc = @return_if_reduced foldrunion(op, T.b, init)
        foldrunion(op, T.a, acc)
    else
        op(T, init)
    end

@generated asntypes(::Type{T}) where {T} =
    QuoteNode(foldrunion((S, types) -> (Val(S), types...), T, ()))

struct UnionValue{T <: NTypes,NBytes}
    types::T
    data::NTuple{NBytes,UInt32}
    typeid::UInt8
end

@noinline unreachable() = error("unreachable")

@inline function unionvalue(::Type{T}, v::T) where {T}
    if T isa Union
        types = asntypes(T)
        nbytes = foldrunion(T, 0) do S, n
            Base.@_inline_meta
            max(sizeof_aligned(S), n)
        end
        dest = Ref(ntuple(_ -> UInt32(0), Val(nbytes)))
        GC.@preserve dest begin
            unsafe_store!(Ptr{typeof(v)}(pointer_from_objref(dest)), v)
        end
        @inline function searchid((v, id), t)
            if v isa valueof(t)
                Reduced(id)
            else
                (v, id + 1)
            end
        end
        typeid = terminating_foldlargs(searchid, unreachable, (v, 1), types...)
        return UnionValue(types, dest[], UInt8(typeid))
    else
        return v
    end
end

@noinline invalid_typeid() = error("invalid typeid")

interpret(x) = x
@inline function interpret(uv::UnionValue)
    data = uv.data
    typeid = uv.typeid
    @inline function _get(id, t)
        if id == typeid
            T = valueof(t)
            ref = Ref(data)
            GC.@preserve ref begin
                v = unsafe_load(Ptr{T}(pointer_from_objref(ref)))
            end
            return Reduced(v)
        else
            id + 1
        end
    end
    return terminating_foldlargs(_get, invalid_typeid, 1, uv.types...)
end

@inline function CUDA.shfl_recurse(op, uv::UnionValue)
    data = map(op, uv.data)
    typeid = op(uv.typeid)
    return UnionValue(uv.types, data, typeid)
end
