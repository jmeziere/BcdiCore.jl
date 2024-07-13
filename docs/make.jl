using Documenter, BcdiCore

makedocs(
    sitename="BcdiCore.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    ),
    pages = [
        "Main"=>"index.md",
        "Usage"=>[
            "Overview"=>"overview.md",
            "Atomic Models"=>"atomic.md",
            "Mesoscale Models"=>"meso.md",
            "Traditional Models"=>"trad.md",
            "Multiscale Modes"=>"multi.md"
        ]
    ]
)

deploydocs(
    repo = "github.com/byu-cig/BcdiCore.jl.git",
)
