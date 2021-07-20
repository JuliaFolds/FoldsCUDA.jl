module TestCoalescedCUDAEx

using Adapt
using CUDA
using Folds.Testing: parse_tests, test_with_sequential
using FoldsCUDA
using Transducers

# Note: each line must ends with `; init = .*)`. See below.
rawdata = """
mapreduce(x -> x^2, +, 1:10; init = 0)
mapreduce(*, +, 1:10, 11:20; init = 0)
mapreduce(*, +, 1:10, 11:20, 21:30; init = 0)
sum(1:10; init = 0)
sum(1:2^30; init = 0)
sum(first, pairs(1:10); init = 0)
sum(last, pairs(1:10); init = 0)
sum(x^2 for x in 1:11; init = 0)
sum(x^2 for x in 1:11 if isodd(x); init = 0)
sum(y for x in 1:11 if isodd(x) for y in 1:x:x^2; init = 0)
"""

tests = parse_tests(rawdata, @__MODULE__)

if VERSION >= v"1.6"
    lines = map(eachline(IOBuffer(rawdata))) do ln
        replace(ln, r"; init = .*?\)$" => ")")
    end
    tests = append!(tests, parse_tests(join(lines, "\n"), @__MODULE__))
end

upload(x) = adapt(CuArray, x)

test() = test_with_sequential(tests, [CoalescedCUDAEx()]; upload = upload)

end  # module
