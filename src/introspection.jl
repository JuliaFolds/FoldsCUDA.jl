const RUN_ON_HOST_IF_NORETURN = Ref(false)

struct FailedInference <: Exception
    f::Any
    kernel_args::Tuple
    kernel_return::Any
    host_args::Tuple
    host_return::Any
end

function Base.showerror(io::IO, err::FailedInference)
    print(io, FailedInference, ": ")
    if err.kernel_return === Union{}
        print(io, "Kernel is inferred to not return (return type is `Union{}`)")
    else
        print(io, "Kernel is inferred to return invalid type: ", err.kernel_return)
    end
    if err.kernel_return === Union{} && err.host_return !== Union{}
        println(io)
        print(io, "Note: on the host, the return type is inferred as ", err.host_return)
    end
    println(io)
    printstyled(io, "HINT"; bold = true, color = :light_black)
    printstyled(
        io,
        ": if this exception is caught as `err``, use `CUDA.code_typed(err)` to",
        " introspect the erronous code.";
        color = :light_black,
    )
end

CUDA.code_typed(err::FailedInference; options...) =
    CUDA.code_typed(err.f, typeof(err.kernel_args); options...)
