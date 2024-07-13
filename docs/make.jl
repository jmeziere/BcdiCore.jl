using Documenter, BcdiCore

makedocs(
    sitename="BcdiCore.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    ),
    pages = [
        "Main"=>"index.md",
        "Usage"=>[
            "Overview"=>"use/overview.md",
            "Atomic Models"=>"use/atomic.md",
            "Mesoscale Models"=>"use/meso.md",
            "Traditional Models"=>"use/trad.md",
            "Multiscale Modes"=>"use/multi.md"
        ]
    ]
)

deploydocs(
    repo = "github.com/byu-cig/BcdiCore.jl.git",
)
