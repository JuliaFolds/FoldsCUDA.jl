using Documenter
using FoldsCUDA

makedocs(
    # See:
    # https://juliadocs.github.io/Documenter.jl/stable/lib/public/#Documenter.makedocs
    sitename = "FoldsCUDA",
    format = Documenter.HTML(),
    modules = [FoldsCUDA],
    doctest = false,
)

deploydocs(
    # See:
    # https://juliadocs.github.io/Documenter.jl/stable/lib/public/#Documenter.deploydocs
    repo = "github.com/JuliaFolds/FoldsCUDA.jl",
    push_preview = true,
)
