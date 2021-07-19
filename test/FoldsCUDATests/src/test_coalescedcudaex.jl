module TestCoalescedCUDAEx

using Adapt
using CUDA
using Folds.Testing: parse_tests, test_with_sequential
using FoldsCUDA
using Transducers

rawdata = """
mapreduce(x -> x^2, +, 1:10; init = 0)
mapreduce(x -> x^2, +, 1:10)
mapreduce(*, +, 1:10, 11:20; init = 0)
mapreduce(*, +, 1:10, 11:20)
mapreduce(*, +, 1:10, 11:20, 21:30; init = 0)
mapreduce(*, +, 1:10, 11:20, 21:30)
prod(y for x in 1:11 if isodd(x) for y in 1:x:x^2; init = 1)
prod(y for x in 1:11 if isodd(x) for y in 1:x:x^2)
sum(1:10; init = 0)
sum(1:10)
sum(first, pairs(1:10); init = 0)
sum(first, pairs(1:10))
sum(last, pairs(1:10); init = 0)
sum(last, pairs(1:10))
sum(x^2 for x in 1:11; init = 0)
sum(x^2 for x in 1:11)
sum(x^2 for x in 1:11 if isodd(x); init = 0)
sum(x^2 for x in 1:11 if isodd(x))
"""

tests = parse_tests(rawdata, @__MODULE__)

upload(x) = adapt(CuArray, x)

test() = test_with_sequential(tests, [CoalescedCUDAEx()]; upload = upload)

end  # module
