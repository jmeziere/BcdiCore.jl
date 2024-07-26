module BcdiCore
    using CUDA
    using CUDA.CUFFT
    using FINUFFT
    using LogExpFunctions

    include("Plans.jl")
    include("State.jl")
    include("Losses.jl")
    include("Setup.jl")
# Write your package code here.

end
