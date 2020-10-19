# FoldsCUDA

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://juliafolds.github.io/FoldsCUDA.jl/dev)
[![GitLab CI](https://gitlab.com/JuliaGPU/FoldsCUDA.jl/badges/master/pipeline.svg)](https://gitlab.com/JuliaGPU/FoldsCUDA.jl/-/pipelines)
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

* `foldx_cuda`: a GPU equivalent of parallel extended fold
  [`Transducers.foldxt`](https://juliafolds.github.io/Transducers.jl/dev/reference/manual/#Transducers.foldxt).
* `CUDAEx`: a parallel loop executor for FLoops.jl.

See the documentation of
[Transducers.jl](https://juliafolds.github.io/Transducers.jl/dev/) and
[FLoops.jl](https://juliafolds.github.io/FLoops.jl/dev/) for more
information.

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
julia> using Transducers

julia> @allowscalar xs[300] = -0.5;

julia> foldx_cuda(TeeRF(min, max), xs)
(-0.5f0, 2.0f0)

julia> foldx_cuda(TeeRF(min, max), (2x for x in xs))  # iterator comprehension works
(-1.0f0, 4.0f0)

julia> foldx_cuda(TeeRF(min, max), Map(x -> 2x), xs)  # equivalent, using a transducer
(-1.0f0, 4.0f0)
```

### More examples

For more examples, see the
[examples section in the documentation](https://juliafolds.github.io/FoldsCUDA.jl/dev/examples/).
