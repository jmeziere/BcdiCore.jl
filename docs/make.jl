using Documenter, DocumenterCitations, BcdiCore

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"))

makedocs(
    sitename="BcdiCore.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    ),
    pages = [
        "BcdiCore"=>"index.md",
        "Usage"=>[
            "Overview"=>"use/overview.md",
            "Atomic Models"=>"use/atomic.md",
            "Mesoscale Models"=>"use/meso.md",
            "Traditional Models"=>"use/trad.md",
            "Multiscale Modes"=>"use/multi.md"
        ]
    ],
    plugins = [bib]
)

deploydocs(
    repo = "github.com/byu-cxi/BcdiCore.jl.git",
)
