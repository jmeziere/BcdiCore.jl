struct AtomicState{T,F1,F2}
    losstype::Int64
    scale::Bool
    intens::CuArray{Int64, 3, CUDA.Mem.DeviceBuffer}
    G::Vector{Float64}
    h::CuArray{Int64, 3, CUDA.Mem.DeviceBuffer}
    k::CuArray{Int64, 3, CUDA.Mem.DeviceBuffer}
    l::CuArray{Int64, 3, CUDA.Mem.DeviceBuffer}
    plan::T
    realSpace::CuArray{ComplexF64, 1, CUDA.Mem.DeviceBuffer}
    recipSpace::CuArray{ComplexF64, 3, CUDA.Mem.DeviceBuffer}
    tempSpace::CuArray{ComplexF64, 3, CUDA.Mem.DeviceBuffer}
    recSupport::CuArray{Bool, 3, CUDA.Mem.DeviceBuffer}
    working::CuArray{ComplexF64, 3, CUDA.Mem.DeviceBuffer}
    xDeriv::CuArray{Float64, 1, CUDA.Mem.DeviceBuffer}
    yDeriv::CuArray{Float64, 1, CUDA.Mem.DeviceBuffer}
    zDeriv::CuArray{Float64, 1, CUDA.Mem.DeviceBuffer}
    scalingManyAtomicKernel!::F1
    scaleThreads::Int64
    lossManyAtomicKernel!::F2
    lossThreads::Int64

    function AtomicState(losstype, scale, intens, G, h, k, l, recSupport)
        plan = NUGpuPlan(size(intens))
        realSpace = CUDA.zeros(ComplexF64, 0)
        recipSpace = CUDA.zeros(ComplexF64, size(intens))
        tempSpace = CUDA.zeros(ComplexF64, size(intens))
        working = CUDA.zeros(ComplexF64, size(intens))
        xDeriv = CUDA.zeros(Float64, 0)
        yDeriv = CUDA.zeros(Float64, 0)
        zDeriv = CUDA.zeros(Float64, 0)

        intLosstype = -1
        if losstype == "likelihood"
            intLosstype = 0
        elseif losstype == "L2"
            intLosstype = 1
        end

        scalingManyAtomicKernel! = @cuda launch=false scalingManyAtomic!(
            CUDA.zeros(Float64, 1),
            0,
            CUDA.zeros(Float64, 1),
            CUDA.zeros(Float64, 1),
            CUDA.zeros(Float64, 1),
            CUDA.zeros(Bool, 1),
            CUDA.zeros(Int64, 1,1,1),
            CUDA.zeros(ComplexF64, 1,1,1),
            CUDA.zeros(Int64, 1,1,1),
            CUDA.zeros(Int64, 1,1,1),
            CUDA.zeros(Int64, 1,1,1),
            CUDA.zeros(Bool, 1,1,1),
            0.0,  0.0, 0.0
        )
        configScaling = launch_configuration(scalingManyAtomicKernel!.fun)
        scaleThreads = configScaling.threads

        lossManyAtomicKernel! = @cuda launch=false lossManyAtomic!(
            CUDA.zeros(Float64, 1),
            0,
            CUDA.zeros(Float64, 1),
            CUDA.zeros(Float64, 1),
            CUDA.zeros(Float64, 1),
            CUDA.zeros(Bool, 1),
            CUDA.zeros(Float64, 1),
            CUDA.zeros(Int64, 1,1,1),
            CUDA.zeros(ComplexF64, 1,1,1),
            CUDA.zeros(Int64, 1,1,1),
            CUDA.zeros(Int64, 1,1,1),
            CUDA.zeros(Int64, 1,1,1),
            CUDA.zeros(Bool, 1,1,1),
            0.0,  0.0, 0.0
        )
        configLoss = launch_configuration(scalingManyAtomicKernel!.fun)
        lossThreads = configLoss.threads

        new{typeof(plan), typeof(scalingManyAtomicKernel!), typeof(lossManyAtomicKernel!)}(intLosstype, scale, intens, G, h, k, l, plan, realSpace, recipSpace, tempSpace, recSupport, working, xDeriv, yDeriv, zDeriv, scalingManyAtomicKernel!, scaleThreads, lossManyAtomicKernel!, lossThreads)
    end
end

function setpts!(state::AtomicState, x, y, z, getDeriv)
    if maximum(x) > 2*pi || minimum(x) < 0 ||
       maximum(y) > 2*pi || minimum(y) < 0 ||
       maximum(z) > 2*pi || minimum(z) < 0
        @warn "Positions are not between 0 and 2π, these need to be scaled."
    end
    resize!(state.realSpace, length(x))
    state.realSpace .= exp.(-1im .* (state.G[1] .* x .+ state.G[2] .* y .+ state.G[3] .* z))
    resize!(state.plan.realSpace, length(x))
    if getDeriv
        resize!(state.xDeriv, length(x))
        resize!(state.yDeriv, length(x))
        resize!(state.zDeriv, length(x))
    end
    FINUFFT.cufinufft_setpts!(state.plan.forPlan, x, y, z)
    FINUFFT.cufinufft_setpts!(state.plan.revPlan, x, y, z)
end

function forwardProp(state::AtomicState, saveRecip)
    state.plan * state.realSpace
    state.plan.recipSpace .+= state.recipSpace

    if saveRecip
        state.recipSpace .= state.plan.recipSpace
    end
end

function slowForwardProp(state::AtomicState, x, y, z, adds, saveRecip)
    state.plan.recipSpace .= state.recipSpace
    for i in 1:length(x)
        if isnan(x[i])
            continue
        end
        state.plan.recipSpace .+= (2 .* adds[i] .- 1) .* exp.(-1im .* (
            x[i] .* (state.G[1] .+ state.h) .+
            y[i] .* (state.G[2] .+ state.k) .+
            z[i] .* (state.G[3] .+ state.l)
        ))
    end

    if saveRecip
        state.recipSpace .= state.plan.recipSpace
    end
end

function backProp(state::AtomicState)
    state.plan.tempSpace .= state.plan.recipSpace
    state.tempSpace .= state.recipSpace
    # Calculate the x derivatives
    state.xDeriv.= 0
    state.recipSpace .= state.working .* (state.h .+ state.G[1])
    state.plan \ state.recipSpace
    state.xDeriv .+= real.(state.plan.realSpace) .* imag.(state.realSpace)
    state.xDeriv .-= imag.(state.plan.realSpace) .* real.(state.realSpace)

    # Calculate the y derivatives
    state.yDeriv.= 0
    state.recipSpace .= state.working .* (state.k .+ state.G[2])
    state.plan \ state.recipSpace
    state.yDeriv .+= real.(state.plan.realSpace) .* imag.(state.realSpace)
    state.yDeriv .-= imag.(state.plan.realSpace) .* real.(state.realSpace)

    # Calculate the z derivatives
    state.zDeriv.= 0
    state.recipSpace .= state.working .* (state.l .+ state.G[3])
    state.plan \ state.recipSpace
    state.zDeriv .+= real.(state.plan.realSpace) .* imag.(state.realSpace)
    state.zDeriv .-= imag.(state.plan.realSpace) .* real.(state.realSpace)

    state.plan.recipSpace .= state.plan.tempSpace
    state.recipSpace .= state.tempSpace
end

struct TradState{T}
    losstype::Int64
    scale::Bool
    intens::CuArray{Int64, 3, CUDA.Mem.DeviceBuffer}
    plan::T
    realSpace::CuArray{ComplexF64, 3, CUDA.Mem.DeviceBuffer}
    recipSpace::CuArray{ComplexF64, 3, CUDA.Mem.DeviceBuffer}
    recSupport::CuArray{Bool, 3, CUDA.Mem.DeviceBuffer}
    working::CuArray{ComplexF64, 3, CUDA.Mem.DeviceBuffer}
    deriv::CuArray{ComplexF64, 3, CUDA.Mem.DeviceBuffer}

    function TradState(losstype, scale, realSpace, intens, recSupport)
        plan = UGpuPlan(size(intens))
        recipSpace = CUDA.zeros(ComplexF64, size(intens))
        working = CUDA.zeros(ComplexF64, size(intens))
        deriv = CUDA.zeros(ComplexF64, size(intens))

        intLosstype = -1
        if losstype == "likelihood"
            intLosstype = 0
        elseif losstype == "L2"
            intLosstype = 1
        end

        new{typeof(plan)}(intLosstype, scale, intens, plan, realSpace, recipSpace, recSupport, working, deriv)
    end

    function TradState(intLosstype, scale, intens, plan, realSpace, recSupport, working, deriv)
        recipSpace = CUDA.zeros(ComplexF64, size(intens))
        new{typeof(plan)}(intLosstype, scale, intens, plan, realSpace, recipSpace, recSupport, working, deriv)
    end
end

function forwardProp(state::TradState, saveRecip)
    state.plan * state.realSpace
    state.plan.recipSpace .+= state.recipSpace

    if saveRecip
        state.recipSpace .= state.plan.recipSpace
    end
end

function backProp(state::TradState)
    state.plan.tempSpace .= state.plan.recipSpace

    state.plan \ state.working
    state.deriv .= state.plan.realSpace

    state.plan.recipSpace .= state.plan.tempSpace
end

struct MesoState{T}
    losstype::Int64
    scale::Bool
    intens::CuArray{Int64, 3, CUDA.Mem.DeviceBuffer}
    G::Vector{Float64}
    h::CuArray{Int64, 3, CUDA.Mem.DeviceBuffer}
    k::CuArray{Int64, 3, CUDA.Mem.DeviceBuffer}
    l::CuArray{Int64, 3, CUDA.Mem.DeviceBuffer}
    plan::T
    realSpace::CuArray{ComplexF64, 1, CUDA.Mem.DeviceBuffer}
    rholessRealSpace::CuArray{ComplexF64, 1, CUDA.Mem.DeviceBuffer}
    recipSpace::CuArray{ComplexF64, 3, CUDA.Mem.DeviceBuffer}
    tempSpace::CuArray{ComplexF64, 3, CUDA.Mem.DeviceBuffer}
    recSupport::CuArray{Bool, 3, CUDA.Mem.DeviceBuffer}
    working::CuArray{ComplexF64, 3, CUDA.Mem.DeviceBuffer}
    rhoDeriv::CuArray{Float64, 1, CUDA.Mem.DeviceBuffer}
    uxDeriv::CuArray{Float64, 1, CUDA.Mem.DeviceBuffer}
    uyDeriv::CuArray{Float64, 1, CUDA.Mem.DeviceBuffer}
    uzDeriv::CuArray{Float64, 1, CUDA.Mem.DeviceBuffer}

    function MesoState(losstype, scale, intens, G, h, k, l, recSupport)
        plan = NUGpuPlan(size(intens))
        realSpace = CUDA.zeros(ComplexF64, 0)
        rholessRealSpace = CUDA.zeros(ComplexF64, 0)
        recipSpace = CUDA.zeros(ComplexF64, size(intens))
        tempSpace = CUDA.zeros(ComplexF64, size(intens))
        working = CUDA.zeros(ComplexF64, size(intens))
        rhoDeriv = CUDA.zeros(Float64, 0)
        uxDeriv = CUDA.zeros(Float64, 0)
        uyDeriv = CUDA.zeros(Float64, 0)
        uzDeriv = CUDA.zeros(Float64, 0)

        intLosstype = -1
        if losstype == "likelihood"
            intLosstype = 0
        elseif losstype == "L2"
            intLosstype = 1
        end

        new{typeof(plan)}(intLosstype, scale, intens, G, h, k, l, plan, realSpace, rholessRealSpace, recipSpace, tempSpace, recSupport, working, rhoDeriv, uxDeriv, uyDeriv, uzDeriv)
    end
end

function setpts!(state::MesoState, x, y, z, rho, ux, uy, uz, getDeriv)
    if maximum(x+ux) > 2*pi || minimum(x+ux) < 0 ||
       maximum(y+uy) > 2*pi || minimum(y+uy) < 0 ||
       maximum(z+uz) > 2*pi || minimum(z+uz) < 0
        println(maximum(x+ux))
        println(maximum(y+uy))
        println(maximum(z+uz))
        println(minimum(x+ux))
        println(minimum(y+uy))
        println(minimum(z+uz))
        @warn "Positions are not between 0 and 2π, these need to be scaled."
    end
    resize!(state.rholessRealSpace, length(x))
    resize!(state.realSpace, length(x))
    state.rholessRealSpace .= exp.(-1im .* (state.G[1] .* ux .+ state.G[2] .* uy .+ state.G[3] .* uz))
    state.realSpace .= rho .* state.rholessRealSpace
    resize!(state.plan.realSpace, length(x))
    if getDeriv
        resize!(state.rhoDeriv, length(x))
        resize!(state.uxDeriv, length(x))
        resize!(state.uyDeriv, length(x))
        resize!(state.uzDeriv, length(x))
    end
    mesoX = x .+ ux
    mesoY = y .+ uy
    mesoZ = z .+ uz
    
    FINUFFT.cufinufft_setpts!(state.plan.forPlan, mesoX, mesoY, mesoZ)
    FINUFFT.cufinufft_setpts!(state.plan.revPlan, mesoX, mesoY, mesoZ)
end

function forwardProp(state::MesoState, saveRecip)
    state.plan * state.realSpace
    state.plan.recipSpace .+= state.recipSpace

    if saveRecip
        state.recipSpace .= state.plan.recipSpace
    end
end

function backProp(state::MesoState)
    state.plan.tempSpace .= state.plan.recipSpace
    state.tempSpace .= state.recipSpace

    # Calculate the ux derivatives
    state.rhoDeriv.= 0
    state.recipSpace .= state.working
    state.plan \ state.recipSpace
    state.rhoDeriv .+= real.(state.plan.realSpace) .* real.(state.rholessRealSpace)
    state.rhoDeriv .+= imag.(state.plan.realSpace) .* imag.(state.rholessRealSpace)

    # Calculate the ux derivatives
    state.uxDeriv.= 0
    state.recipSpace .= state.working .* (state.h .+ state.G[1])
    state.plan \ state.recipSpace
    state.uxDeriv .+= real.(state.plan.realSpace) .* imag.(state.realSpace)
    state.uxDeriv .-= imag.(state.plan.realSpace) .* real.(state.realSpace)

    # Calculate the uy derivatives
    state.uyDeriv.= 0
    state.recipSpace .= state.working .* (state.k .+ state.G[2])
    state.plan \ state.recipSpace
    state.uyDeriv .+= real.(state.plan.realSpace) .* imag.(state.realSpace)
    state.uyDeriv .-= imag.(state.plan.realSpace) .* real.(state.realSpace)

    # Calculate the uz derivatives
    state.uzDeriv.= 0
    state.recipSpace .= state.working .* (state.l .+ state.G[3])
    state.plan \ state.recipSpace
    state.uzDeriv .+= real.(state.plan.realSpace) .* imag.(state.realSpace)
    state.uzDeriv .-= imag.(state.plan.realSpace) .* real.(state.realSpace)

    state.plan.recipSpace .= state.plan.tempSpace
    state.recipSpace .= state.tempSpace
end

struct MultiState{T}
    losstype::Int64
    scale::Bool
    intens::CuArray{Int64, 3, CUDA.Mem.DeviceBuffer}
    G::Vector{Float64}
    h::CuArray{Int64, 3, CUDA.Mem.DeviceBuffer}
    k::CuArray{Int64, 3, CUDA.Mem.DeviceBuffer}
    l::CuArray{Int64, 3, CUDA.Mem.DeviceBuffer}
    plan::T
    rhoPlan::T
    realSpace::CuArray{ComplexF64, 1, CUDA.Mem.DeviceBuffer}
    rholessRealSpace::CuArray{ComplexF64, 1, CUDA.Mem.DeviceBuffer}
    recipSpace::CuArray{ComplexF64, 3, CUDA.Mem.DeviceBuffer}
    tempSpace::CuArray{ComplexF64, 3, CUDA.Mem.DeviceBuffer}
    recSupport::CuArray{Bool, 3, CUDA.Mem.DeviceBuffer}
    working::CuArray{ComplexF64, 3, CUDA.Mem.DeviceBuffer}
    xDeriv::CuArray{Float64, 1, CUDA.Mem.DeviceBuffer}
    yDeriv::CuArray{Float64, 1, CUDA.Mem.DeviceBuffer}
    zDeriv::CuArray{Float64, 1, CUDA.Mem.DeviceBuffer}
    rhoDeriv::CuArray{Float64, 1, CUDA.Mem.DeviceBuffer}
    uxDeriv::CuArray{Float64, 1, CUDA.Mem.DeviceBuffer}
    uyDeriv::CuArray{Float64, 1, CUDA.Mem.DeviceBuffer}
    uzDeriv::CuArray{Float64, 1, CUDA.Mem.DeviceBuffer}

    function MultiState(losstype, scale, intens, G, h, k, l, recSupport)
        plan = NUGpuPlan(size(intens))
        rhoPlan = NUGpuPlan(size(intens))
        realSpace = CUDA.zeros(ComplexF64, 0)
        rholessRealSpace = CUDA.zeros(ComplexF64, 0)
        recipSpace = CUDA.zeros(ComplexF64, size(intens))
        tempSpace = CUDA.zeros(ComplexF64, size(intens))
        working = CUDA.zeros(ComplexF64, size(intens))
        xDeriv = CUDA.zeros(Float64, 0)
        yDeriv = CUDA.zeros(Float64, 0)
        zDeriv = CUDA.zeros(Float64, 0)
        rhoDeriv = CUDA.zeros(Float64, 0)
        uxDeriv = CUDA.zeros(Float64, 0)
        uyDeriv = CUDA.zeros(Float64, 0)
        uzDeriv = CUDA.zeros(Float64, 0)

        intLosstype = -1
        if losstype == "likelihood"
            intLosstype = 0
        elseif losstype == "L2"
            intLosstype = 1
        end

        new{typeof(plan)}(intLosstype, scale, intens, G, h, k, l, plan, rhoPlan, realSpace, rholessRealSpace, recipSpace, tempSpace, recSupport, working, xDeriv, yDeriv, zDeriv, rhoDeriv, uxDeriv, uyDeriv, uzDeriv)
    end
end

function setpts!(state::MultiState, x, y, z, mx, my, mz,  rho, ux, uy, uz, getDeriv)
    if maximum(x) > 2*pi || minimum(x) < 0 ||
       maximum(y) > 2*pi || minimum(y) < 0 ||
       maximum(z) > 2*pi || minimum(z) < 0 ||
       maximum(mx+ux) > 2*pi || minimum(mx+ux) < 0 ||
       maximum(my+uy) > 2*pi || minimum(my+uy) < 0 ||
       maximum(mz+uz) > 2*pi || minimum(mz+uz) < 0
        @warn "Positions are not between 0 and 2π, these need to be scaled."
    end
    resize!(state.rholessRealSpace, length(mx))
    resize!(state.realSpace, length(x)+length(mx))
    state.rholessRealSpace .= exp.(-1im .* (state.G[1] .* ux .+ state.G[2] .* uy .+ state.G[3] .* uz))
    state.realSpace[1:length(mx)] .= rho .* state.rholessRealSpace
    state.realSpace[length(mx)+1:end] .= exp.(-1im .* (state.G[1] .* x .+ state.G[2] .* y .+ state.G[3] .* z))
    resize!(state.rhoPlan.realSpace, length(mx))
    resize!(state.plan.realSpace, length(x)+length(mx))
    if getDeriv
        resize!(state.xDeriv, length(x))
        resize!(state.yDeriv, length(x))
        resize!(state.zDeriv, length(x))
        resize!(state.rhoDeriv, length(mx))
        resize!(state.uxDeriv, length(mx))
        resize!(state.uyDeriv, length(mx))
        resize!(state.uzDeriv, length(mx))
    end
    fullX = vcat(mx .+ ux, x)
    fullY = vcat(my .+ uy, y)
    fullZ = vcat(mz .+ uz, z)
    mesoX = mx .+ ux
    mesoY = my .+ uy
    mesoZ = mz .+ uz
    FINUFFT.cufinufft_setpts!(state.plan.forPlan, fullX, fullY, fullZ)
    FINUFFT.cufinufft_setpts!(state.plan.revPlan, fullX, fullY, fullZ)
    FINUFFT.cufinufft_setpts!(state.rhoPlan.revPlan, mesoX, mesoY, mesoZ)
end

function forwardProp(state::MultiState, saveRecip)
    state.plan * state.realSpace
    state.plan.recipSpace .+= state.recipSpace

    if saveRecip
        state.recipSpace .= state.plan.recipSpace
    end
end

function backProp(state::MultiState)
    nm = length(state.rholessRealSpace)
    state.plan.tempSpace .= state.plan.recipSpace
    state.tempSpace .= state.recipSpace

    # Calculate the rho derivatives
    state.rhoDeriv.= 0
    state.recipSpace .= state.working
    state.rhoPlan \ state.recipSpace
    state.rhoDeriv .+= real.(state.rhoPlan.realSpace) .* real.(state.rholessRealSpace)
    state.rhoDeriv .+= imag.(state.rhoPlan.realSpace) .* imag.(state.rholessRealSpace)

    # Calculate the x & ux derivatives
    state.xDeriv.= 0
    state.uxDeriv.= 0
    state.recipSpace .= state.working .* (state.h .+ state.G[1])
    state.plan \ state.recipSpace
    @views state.xDeriv .+= real.(state.plan.realSpace[nm+1:end]) .* imag.(state.realSpace[nm+1:end])
    @views state.xDeriv .-= imag.(state.plan.realSpace[nm+1:end]) .* real.(state.realSpace[nm+1:end])
    @views state.uxDeriv .+= real.(state.plan.realSpace[1:nm]) .* imag.(state.realSpace[1:nm])
    @views state.uxDeriv .-= imag.(state.plan.realSpace[1:nm]) .* real.(state.realSpace[1:nm])

    # Calculate the y & uy derivatives
    state.yDeriv.= 0
    state.uyDeriv.= 0
    state.recipSpace .= state.working .* (state.k .+ state.G[2])
    state.plan \ state.recipSpace
    @views state.yDeriv .+= real.(state.plan.realSpace[nm+1:end]) .* imag.(state.realSpace[nm+1:end])
    @views state.yDeriv .-= imag.(state.plan.realSpace[nm+1:end]) .* real.(state.realSpace[nm+1:end])
    @views state.uyDeriv .+= real.(state.plan.realSpace[1:nm]) .* imag.(state.realSpace[1:nm])
    @views state.uyDeriv .-= imag.(state.plan.realSpace[1:nm]) .* real.(state.realSpace[1:nm])

    # Calculate the z & uz derivatives
    state.zDeriv.= 0
    state.uzDeriv.= 0
    state.recipSpace .= state.working .* (state.l .+ state.G[3])
    state.plan \ state.recipSpace
    @views state.zDeriv .+= real.(state.plan.realSpace[nm+1:end]) .* imag.(state.realSpace[nm+1:end])
    @views state.zDeriv .-= imag.(state.plan.realSpace[nm+1:end]) .* real.(state.realSpace[nm+1:end])
    @views state.uzDeriv .+= real.(state.plan.realSpace[1:nm]) .* imag.(state.realSpace[1:nm])
    @views state.uzDeriv .-= imag.(state.plan.realSpace[1:nm]) .* real.(state.realSpace[1:nm])

    state.plan.recipSpace .= state.plan.tempSpace
    state.recipSpace .= state.tempSpace
end
