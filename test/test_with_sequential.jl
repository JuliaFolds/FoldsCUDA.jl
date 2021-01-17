module TestWithSequential

using Adapt
using CUDA
using Folds.Testing: test_with_sequential
using FoldsCUDA

function upload(x)
    Base.isbitsunion(eltype(x)) && return nothing
    return adapt(CuArray, x)
end

include_test(test) = :nogpu ∉ test.tags

test_with_sequential([CUDAEx()]; upload = upload, include_test = include_test)

end  # module
