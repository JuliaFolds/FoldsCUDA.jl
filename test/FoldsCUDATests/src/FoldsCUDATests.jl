module FoldsCUDATests

using GPUArrays
using Test

function include_tests(m = @__MODULE__, dir = @__DIR__)
    for file in readdir(dir)
        if match(r"^test_.*\.jl$", file) !== nothing
            Base.include(m, joinpath(dir, file))
        end
    end
end

include_tests()

module Generic
using ..FoldsCUDATests: include_tests
include_tests(@__MODULE__, joinpath(@__DIR__, "generic"))
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

this_project() = joinpath(dirname(@__DIR__), "Project.toml")

function is_in_path()
    project = this_project()
    paths = Base.load_path()
    project in paths && return true
    realproject = realpath(project)
    realproject in paths && return true
    matches(path) = path == project || path == realproject
    return any(paths) do path
        matches(path) || matches(realpath(path))
    end
end

function with_project(f)
    is_in_path() && return f()
    load_path = copy(LOAD_PATH)
    push!(LOAD_PATH, this_project())
    try
        f()
    finally
        append!(empty!(LOAD_PATH), load_path)
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
        if m in (TestTypeChangingAccumulators, Generic.TestReducePartitionBy)
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

""" A list of tests to be run from UnionArrays.jl """
const UNIONARRAYS_TESTS = [TestTypeChangingAccumulators]

function runtests_unionarrays()
    with_project() do
        runtests_impl(UNIONARRAYS_TESTS)
    end
end

end # module
