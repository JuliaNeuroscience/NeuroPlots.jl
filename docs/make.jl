using NeuroPlots
using Documenter

DocMeta.setdocmeta!(NeuroPlots, :DocTestSetup, :(using NeuroPlots); recursive=true)

makedocs(;
    modules=[NeuroPlots],
    authors="Zachary P. Christensen <zchristensen7@gmail.com> and contributors",
    repo="https://github.com/Tokazama/NeuroPlots.jl/blob/{commit}{path}#{line}",
    sitename="NeuroPlots.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://Tokazama.github.io/NeuroPlots.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/Tokazama/NeuroPlots.jl",
)
