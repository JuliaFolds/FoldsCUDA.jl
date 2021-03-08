module FoldsCUDATests

using GPUArrays
using Pkg
using Test

function include_tests(dir)
    for file in readdir(dir)
        if match(r"^test_.*\.jl$", file) !== nothing
            include(joinpath(dir, file))
        end
    end
end

include_tests(@__DIR__)

module Generic
using ..FoldsCUDATests: include_tests
include_tests(joinpath(@__DIR__, "generic"))
end

should_test_gpu() =
    lowercase(get(ENV, "JULIA_PKGEVAL", "false")) != "true" &&
    lowercase(get(ENV, "CUDAFOLDS_JL_TEST_GPU", "true")) == "true"

requires_gpu(m::Module) = parentmodule(m) === @__MODULE__

function collect_modules(root::Module)
    modules = Module[]
    for n in names(root, all = true)
        m = getproperty(root, n)
        m isa Module || continue
        m === root && continue
        startswith(string(nameof(m)), "Test") || continue
        push!(modules, m)
    end
    return modules
end

function collect_modules()
    modules = collect_modules(@__MODULE__)
    append!(modules, collect_modules(Generic))
    return modules
end

function with_project(f)
    oldproject = Base.active_project()
    try
        Pkg.activate(joinpath(@__DIR__, ".."))
        f()
    finally
        Pkg.activate(oldproject)
    end
end

function runtests(modules = collect_modules())
    with_project() do
        runtests_impl(modules)
    end
end

function runtests_impl(modules)
    GPUArrays.allowscalar(false)
    test_gpu = should_test_gpu()
    @testset "$(nameof(m))" for m in modules
        if !test_gpu && requires_gpu(m)
            continue  # branch inside `for` loop for printing skipped tests
        end
        if m === TestTypeChangingAccumulators
            VERSION < v"1.6-" && continue
        end
        tests = map(names(m, all = true)) do n
            n == :test || startswith(string(n), "test_") || return nothing
            f = getproperty(m, n)
            f !== m || return nothing
            parentmodule(f) === m || return nothing
            applicable(f) || return nothing  # removed by Revise?
            return f
        end
        filter!(!isnothing, tests)
        @testset "$f" for f in tests
            f()
        end
    end
end

end # module
