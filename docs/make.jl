using Documenter
using FoldsCUDA

makedocs(
    sitename = "FoldsCUDA",
    format = Documenter.HTML(),
    modules = [FoldsCUDA]
)

deploydocs(
    repo = "github.com/JuliaFolds/FoldsCUDA.jl"
    push_preview = true,
)
