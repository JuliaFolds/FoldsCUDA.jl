# FoldsCUDA

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://juliafolds.github.io/FoldsCUDA.jl/dev)
[![Buildkite status](https://badge.buildkite.com/c4196ef2fa588454c146bab0001d0f8de876aa864ab7c5de80.svg?branch=master)](https://buildkite.com/julialang/foldscuda-dot-jl)
[![Run tests w/o GPU](https://github.com/JuliaFolds/FoldsCUDA.jl/workflows/Run%20tests%20w/o%20GPU/badge.svg)](https://github.com/JuliaFolds/FoldsCUDA.jl/actions?query=workflow%3A%22Run+tests+w%2Fo+GPU%22)

FoldsCUDA.jl provides
[Transducers.jl](https://github.com/JuliaFolds/Transducers.jl)-compatible
fold (reduce) implemented using
[CUDA.jl](https://github.com/JuliaGPU/CUDA.jl).  This brings the
transducers and reducing function combinators implemented in
Transducers.jl to GPU.  Furthermore, using
[FLoops.jl](https://github.com/JuliaFolds/FLoops.jl), you can write
parallel `for` loops that run on GPU.

## API

FoldsCUDA exports `CUDAEx`, a parallel loop
[executor](https://juliafolds.github.io/Transducers.jl/dev/explanation/glossary/#glossary-executor).
It can be used with the parallel `for` loop created with
[`FLoops.@floop`](https://github.com/JuliaFolds/FLoops.jl),
`Base`-like high-level parallel API in
[Folds.jl](https://github.com/JuliaFolds/Folds.jl), and extensible
transducers provided by
[Transducers.jl](https://github.com/JuliaFolds/Transducers.jl).

## Examples

### `findmax` using FLoops.jl

You can pass CUDA executor `FoldsCUDA.CUDAEx()` to `@floop` to run a
parallel `for` loop on GPU:

```julia
julia> using FoldsCUDA, CUDA, FLoops

julia> using GPUArrays: @allowscalar

julia> xs = CUDA.rand(10^8);

julia> @allowscalar xs[100] = 2;

julia> @allowscalar xs[200] = 2;

julia> @floop CUDAEx() for (x, i) in zip(xs, eachindex(xs))
           @reduce() do (imax = -1; i), (xmax = -Inf32; x)
               if xmax < x
                   xmax = x
                   imax = i
               end
           end
       end

julia> xmax
2.0f0

julia> imax  # the *first* position for the largest value
100
```

### `extrema` using `Transducers.TeeRF`

```julia
julia> using Transducers, Folds

julia> @allowscalar xs[300] = -0.5;

julia> Folds.reduce(TeeRF(min, max), xs, CUDAEx())
(-0.5f0, 2.0f0)

julia> Folds.reduce(TeeRF(min, max), (2x for x in xs), CUDAEx())  # iterator comprehension works
(-1.0f0, 4.0f0)

julia> Folds.reduce(TeeRF(min, max), Map(x -> 2x), xs, CUDAEx())  # equivalent, using a transducer
(-1.0f0, 4.0f0)
```

### More examples

For more examples, see the
[examples section in the documentation](https://juliafolds.github.io/FoldsCUDA.jl/dev/examples/).
