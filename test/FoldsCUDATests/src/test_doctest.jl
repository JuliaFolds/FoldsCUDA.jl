module TestDoctest

using FoldsCUDA
using Documenter: doctest
using Test

test_doctest() = doctest(FoldsCUDA, manual = false)

end  # module
