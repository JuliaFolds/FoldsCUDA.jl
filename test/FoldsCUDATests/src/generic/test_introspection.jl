module TestIntrospection

using CUDA
using FoldsCUDA: FailedInference
using Test

function test_failedinference_regular()
    err = FailedInference(nothing, (1,), Union{}, (1,), Union{})
    msg = sprint(showerror, err)
    @test occursin("not return", msg)
    @test !occursin("on the host,", msg)
end

function test_failedinference_ok_on_host()
    err = FailedInference(nothing, (1,), Union{}, (1,), Int)
    msg = sprint(showerror, err)
    @test occursin("not return", msg)
    @test occursin("on the host,", msg)
end

function test_failedinference_invalid_type()
    err = FailedInference(nothing, (1,), Int, (1,), Int)
    msg = sprint(showerror, err)
    @test occursin("invalid type", msg)
end

function test_failedinference_code_typed()
    CUDA.functional() || return
    err = FailedInference(identity, (1,), Int, (1,), Int)
    @test CUDA.code_typed(err) isa Any
end

end  # module
