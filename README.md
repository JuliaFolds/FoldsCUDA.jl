# CUDAFolds

CUDAFolds.jl provides
[Transducers.jl](https://github.com/JuliaFolds/Transducers.jl)-compatible
fold (reduce) implemented using
[CUDA.jl](https://github.com/JuliaGPU/CUDA.jl).  This brings the
transducers and monoid combinators implemented in Transducers.jl to
GPU.  Furthermore, using
[FLoops.jl](https://github.com/JuliaFolds/FLoops.jl), you can run
write parallel `for` loops that run on GPU.

## API

* `foldx_cuda`: a GPU equivalent of parallel extended fold
  [`Transducers.foldxt`](https://juliafolds.github.io/Transducers.jl/dev/reference/manual/#Transducers.foldxt).
* `CUDAEx`: a parallel loop executor for FLoops.jl.

See the documentation of Transducers.jl and FLoops.jl for more
information.

## Examples

### `findmax` using FLoops.jl

You can pass CUDA executor `CUDAFolds.CUDAEx()` to `@floop` to run a
parallel `for` loop on GPU:

```julia
julia> using CUDAFolds, CUDA, FLoops

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
