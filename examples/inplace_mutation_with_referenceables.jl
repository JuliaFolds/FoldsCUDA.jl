# # In-place mutation with Referenceables.jl

using CUDA
using Folds
using FoldsCUDA
using Referenceables: referenceable

using Test                                                             #src

# ## Simple mutation

function increment_folds!(xs)
    Folds.foreach(referenceable(xs)) do x
        x[] += 1
    end
    return xs
end

if has_cuda_gpu()
    xs = CuArray(1:5)
else
    xs = Array(1:5)
end

@test begin
    collect(increment_folds!(xs))
end == 2:6

# This can also be written with FLoops.jl:

using FLoops

function increment_floops!(xs, ex = nothing)
    @floop ex for x in referenceable(xs)
        x[] += 1
    end
    return xs
end

@test begin
    collect(increment_floops!(xs))
end == 3:7

# ## Fusing reduction and mutationg

# Computing `sum(f, xs)` and `f.(xs)` in one go:

function mutation_with_folds(f, xs)
    ys = similar(xs)
    s = Folds.sum(zip(referenceable(ys), xs)) do (r, x)
        r[] = y = f(x)
        return y
    end
    return s, ys
end
nothing  # hide
#-

if has_cuda_gpu()
    xs = CuArray(1:5)
else
    xs = Array(1:5)
end

s, ys = mutation_with_folds(x -> x^2, xs)
@test begin
    s
end == 55
#-

@test begin
    collect(ys)
end == xs .^ 2
#-

# An equilvalent computaton with FLoops.jl:

using FLoops

function mutation_with_floop(f, xs)
    ys = similar(xs)
    z = zero(eltype(ys))
    @floop for (r, x) in zip(referenceable(ys), xs)
        r[] = y = f(x)
        @reduce(s = z + y)
    end
    return s, ys
end
nothing  # hide
#-

@assert mutation_with_folds(x -> x^2, xs) == (s, ys)
#-
