struct AtomicState{T1,T2,F1}
    loss::T1
    scale::Bool
    intens::CuArray{Int64, 3, CUDA.Mem.DeviceBuffer}
    G::Vector{Float64}
    h::CuArray{Int64, 3, CUDA.Mem.DeviceBuffer}
    k::CuArray{Int64, 3, CUDA.Mem.DeviceBuffer}
    l::CuArray{Int64, 3, CUDA.Mem.DeviceBuffer}
    plan::T2
    realSpace::CuArray{ComplexF64, 1, CUDA.Mem.DeviceBuffer}
    recipSpace::CuArray{ComplexF64, 3, CUDA.Mem.DeviceBuffer}
    tempSpace::CuArray{ComplexF64, 3, CUDA.Mem.DeviceBuffer}
    recSupport::CuArray{Bool, 3, CUDA.Mem.DeviceBuffer}
    working::CuArray{ComplexF64, 3, CUDA.Mem.DeviceBuffer}
    xDeriv::CuArray{Float64, 1, CUDA.Mem.DeviceBuffer}
    yDeriv::CuArray{Float64, 1, CUDA.Mem.DeviceBuffer}
    zDeriv::CuArray{Float64, 1, CUDA.Mem.DeviceBuffer}
    many!::F1
    manyThreads::Int64

    function AtomicState(losstype, scale, intens, G, h, k, l, recSupport)
        plan = NUGpuPlan(size(intens))
        realSpace = CUDA.zeros(ComplexF64, 0)
        recipSpace = CUDA.zeros(ComplexF64, size(intens))
        tempSpace = CUDA.zeros(ComplexF64, size(intens))
        working = CUDA.zeros(ComplexF64, size(intens))
        xDeriv = CUDA.zeros(Float64, 0)
        yDeriv = CUDA.zeros(Float64, 0)
        zDeriv = CUDA.zeros(Float64, 0)

        if losstype == "likelihood"
            loss = PoissonLikelihoodLoss()

            if scale
                manyKernel! = @cuda launch=false manyLikeScal!(
                    CUDA.zeros(Float64, 1),
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
            else
                manyKernel! = @cuda launch=false manyLikeNoScal!(
                    CUDA.zeros(Float64, 1),
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
            end
        elseif losstype == "L2"
            loss = L2Loss()

            if scale
                manyKernel! = @cuda launch=false manyL2Scal!(
                    CUDA.zeros(Float64, 1),
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
            else
                manyKernel! = @cuda launch=false manyL2NoScal!(
                    CUDA.zeros(Float64, 1),
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
            end
        elseif losstype == "Huber"
            loss = HuberLoss(0.5*sqrt(maximum(intens)))

            if scale
                manyKernel! = @cuda launch=false manyHuberScal!(
                    CUDA.zeros(Float64, 1),
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
            else
                manyKernel! = @cuda launch=false manyHuberNoScal!(
                    CUDA.zeros(Float64, 1),
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
            end
        end
        config = launch_configuration(manyKernel!.fun)
        manyThreads = min(length(intens),config.threads)

        new{
            typeof(loss),
            typeof(plan), 
            typeof(manyKernel!) 
        }(
            loss, scale, intens, G, h, k, l, plan, realSpace, 
            recipSpace, tempSpace, recSupport, working, xDeriv, 
            yDeriv, zDeriv, manyKernel!, manyThreads, 
        )
    end
end

function setpts!(state::AtomicState, x, y, z, getDeriv)
    if maximum(x) > 3*pi || minimum(x) < -3*pi ||
       maximum(y) > 3*pi || minimum(y) < -3*pi ||
       maximum(z) > 3*pi || minimum(z) < -3*pi
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

struct TradState{T1,T2,I}
    loss::T1
    scale::Bool
    intens::CuArray{Int64, 3, CUDA.Mem.DeviceBuffer}
    plan::T2
    realSpace::CuArray{ComplexF64, I, CUDA.Mem.DeviceBuffer}
    recipSpace::CuArray{ComplexF64, 3, CUDA.Mem.DeviceBuffer}
    recSupport::CuArray{Bool, 3, CUDA.Mem.DeviceBuffer}
    working::CuArray{ComplexF64, 3, CUDA.Mem.DeviceBuffer}
    deriv::CuArray{ComplexF64, I, CUDA.Mem.DeviceBuffer}

    function TradState(losstype, scale, realSpace, intens, recSupport)
        recipSpace = CUDA.zeros(ComplexF64, size(intens))
        working = CUDA.zeros(ComplexF64, size(intens))

        if losstype == "likelihood"
            loss = PoissonLikelihoodLoss()
        elseif losstype == "L2"
            loss = L2Loss()
        elseif losstype == "Huber"
            loss = HuberLoss(0.1*sqrt(maximum(intens)))
        end

        if ndims(realSpace) == 3
            plan = UGpuPlan(size(intens))
            deriv = CUDA.zeros(ComplexF64, size(intens))

            new{typeof(loss),typeof(plan),3}(
                loss, scale, intens, plan, realSpace, 
                recipSpace, recSupport, working, deriv
            )
        else
            plan = NUGpuPlan(size(intens))
            deriv = CUDA.zeros(Float64, 0)

            new{typeof(loss),typeof(plan),1}(
                loss, scale, intens, plan, realSpace, 
                recipSpace, recSupport, working, deriv
            )
        end
    end

    function TradState(losstype, scale, intens, plan, realSpace, recSupport, working, deriv)
        if losstype == "likelihood"
            loss = PoissonLikelihoodLoss()
        elseif losstype == "L2"
            loss = L2Loss()
        elseif losstype == "Huber"
            loss = HuberLoss(0.03*sqrt(maximum(intens)))
        end

        recipSpace = CUDA.zeros(ComplexF64, size(intens))
        new{typeof(loss),typeof(plan),ndims(realSpace)}(
            loss, scale, intens, plan, realSpace, 
            recipSpace, recSupport, working, deriv
        )
    end
end

function setpts!(state::TradState, x, y, z, getDeriv)
    if maximum(x) > 3*pi || minimum(x) < -3*pi ||
       maximum(y) > 3*pi || minimum(y) < -3*pi ||
       maximum(z) > 3*pi || minimum(z) < -3*pi
        @warn "Positions are not between -3π and 3π, these need to be scaled."
    end
    resize!(state.plan.realSpace, length(x))
    if getDeriv
        resize!(state.deriv, length(x))
    end

    FINUFFT.cufinufft_setpts!(state.plan.forPlan, x, y, z)
    FINUFFT.cufinufft_setpts!(state.plan.revPlan, x, y, z)
end

function backProp(state::TradState)
    state.plan.tempSpace .= state.plan.recipSpace

    state.plan \ state.working
    state.deriv .= state.plan.realSpace

    state.plan.recipSpace .= state.plan.tempSpace
end

struct MesoState{T1,T2,I,J,B}
    loss::T1
    scale::Bool
    intens::CuArray{Int64, 3, CUDA.Mem.DeviceBuffer}
    G::Vector{Float64}
    h::CuArray{Int64, J, CUDA.Mem.DeviceBuffer}
    k::CuArray{Int64, J, CUDA.Mem.DeviceBuffer}
    l::CuArray{Int64, J, CUDA.Mem.DeviceBuffer}
    plan::T2
    realSpace::CuArray{ComplexF64, I, CUDA.Mem.DeviceBuffer}
    rholessRealSpace::CuArray{ComplexF64, I, CUDA.Mem.DeviceBuffer}
    recipSpace::CuArray{ComplexF64, 3, CUDA.Mem.DeviceBuffer}
    tempSpace::CuArray{ComplexF64, 3, CUDA.Mem.DeviceBuffer}
    recSupport::CuArray{Bool, 3, CUDA.Mem.DeviceBuffer}
    working::CuArray{ComplexF64, 3, CUDA.Mem.DeviceBuffer}
    rhoDeriv::CuArray{Float64, I, CUDA.Mem.DeviceBuffer}
    uxDeriv::CuArray{Float64, I, CUDA.Mem.DeviceBuffer}
    uyDeriv::CuArray{Float64, I, CUDA.Mem.DeviceBuffer}
    uzDeriv::CuArray{Float64, I, CUDA.Mem.DeviceBuffer}
    disuxDeriv::CuArray{Float64, I, CUDA.Mem.DeviceBuffer}
    disuyDeriv::CuArray{Float64, I, CUDA.Mem.DeviceBuffer}
    disuzDeriv::CuArray{Float64, I, CUDA.Mem.DeviceBuffer}

    function MesoState(losstype, scale, intens, G, recSupport, nonuniform, highstrain, disjoint)
        recipSpace = CUDA.zeros(ComplexF64, size(intens))
        tempSpace = CUDA.zeros(ComplexF64, size(intens))
        working = CUDA.zeros(ComplexF64, size(intens))

        if losstype == "likelihood"
            loss = PoissonLikelihoodLoss()
        elseif losstype == "L2"
            loss = L2Loss()
        elseif losstype == "Huber"
            loss = HuberLoss(0.1*sqrt(maximum(intens)))
        end

        if nonuniform
            plan = NUGpuPlan(size(intens))
            realSpace = CUDA.zeros(ComplexF64, 0)
            rholessRealSpace = CUDA.zeros(ComplexF64, 0)
            rhoDeriv = CUDA.zeros(Float64, 0)
            uxDeriv = CUDA.zeros(Float64, 0)
            uyDeriv = CUDA.zeros(Float64, 0)
            uzDeriv = CUDA.zeros(Float64, 0)
            disuxDeriv = CUDA.zeros(Float64, 0)
            disuyDeriv = CUDA.zeros(Float64, 0)
            disuzDeriv = CUDA.zeros(Float64, 0)
            if highstrain
                h, k, l = generateRecSpace(size(intens))
            else
                h = CUDA.zeros(Float64,1)
                k = CUDA.zeros(Float64,1)
                l = CUDA.zeros(Float64,1)
            end
        else
            plan = UGpuPlan(size(intens))
            realSpace = CUDA.zeros(ComplexF64, size(intens))
            rholessRealSpace = CUDA.zeros(ComplexF64, size(intens))
            rhoDeriv = CUDA.zeros(Float64, size(intens))
            uxDeriv = CUDA.zeros(Float64, size(intens))
            uyDeriv = CUDA.zeros(Float64, size(intens))
            uzDeriv = CUDA.zeros(Float64, size(intens))
            disuxDeriv = CUDA.zeros(Float64, 0,0,0)
            disuyDeriv = CUDA.zeros(Float64, 0,0,0)
            disuzDeriv = CUDA.zeros(Float64, 0,0,0)
            h = CUDA.zeros(Float64,1)
            k = CUDA.zeros(Float64,1)
            l = CUDA.zeros(Float64,1)
        end

        new{typeof(loss),typeof(plan),ndims(realSpace),ndims(h),disjoint}(
            loss, scale, intens, G, h, k, l, plan, realSpace, rholessRealSpace, recipSpace,
            tempSpace, recSupport, working, rhoDeriv, uxDeriv, uyDeriv, uzDeriv,
            disuxDeriv, disuyDeriv, disuzDeriv
        )
    end
end

function setpts!(state::MesoState{T1,T2,1,T3,false}, x, y, z, rho, ux, uy, uz, getDeriv) where{T1,T2,T3}
    if maximum(x+ux) > 3*pi || minimum(x+ux) < -3*pi ||
       maximum(y+uy) > 3*pi || minimum(y+uy) < -3*pi ||
       maximum(z+uz) > 3*pi || minimum(z+uz) < -3*pi
        @warn "Positions are not between -3π and 3π, these need to be scaled."
    end

    if length(x) != length(state.rholessRealSpace)
        resize!(state.rholessRealSpace, length(x))
        resize!(state.realSpace, length(x))
        resize!(state.plan.realSpace, length(x))
    end
    state.rholessRealSpace .= exp.(-1im .* (state.G[1] .* ux .+ state.G[2] .* uy .+ state.G[3] .* uz))
    state.realSpace .= rho .* state.rholessRealSpace
    if getDeriv && length(x) != length(state.rhoDeriv)
        resize!(state.rhoDeriv, length(x))
        resize!(state.uxDeriv, length(x))
        resize!(state.uyDeriv, length(x))
        resize!(state.uzDeriv, length(x))
    end
    
    if ndims(state.h) == 1
        FINUFFT.cufinufft_setpts!(state.plan.forPlan, x, y, z)
        FINUFFT.cufinufft_setpts!(state.plan.revPlan, x, y, z)
    else
        FINUFFT.cufinufft_setpts!(state.plan.forPlan, x.+ux, y.+uy, z.+uz)
        FINUFFT.cufinufft_setpts!(state.plan.revPlan, x.+ux, y.+uy, z.+uz)
    end
end

function setpts!(
    state::MesoState{T1,T2,1,T3,true}, x, y, z, rho, ux, uy, uz, disux, disuy, disuz, getDeriv
) where{T1,T2,T3}
    if maximum(x+ux) > 3*pi || minimum(x+ux) < -3*pi ||
       maximum(y+uy) > 3*pi || minimum(y+uy) < -3*pi ||
       maximum(z+uz) > 3*pi || minimum(z+uz) < -3*pi
        @warn "Positions are not between -3π and 3π, these need to be scaled."
    end

    if length(x) != length(state.rholessRealSpace)
        resize!(state.rholessRealSpace, length(x))
        resize!(state.realSpace, length(x))
        resize!(state.plan.realSpace, length(x))
    end
    state.rholessRealSpace .= exp.(-1im .* (state.G[1] .* ux .+ state.G[2] .* uy .+ state.G[3] .* uz))
    state.realSpace .= rho .* state.rholessRealSpace
    if getDeriv && length(x) != length(state.rhoDeriv)
        resize!(state.rhoDeriv, length(x))
        resize!(state.uxDeriv, length(x))
        resize!(state.uyDeriv, length(x))
        resize!(state.uzDeriv, length(x))
        resize!(state.disuxDeriv, length(x))
        resize!(state.disuyDeriv, length(x))
        resize!(state.disuzDeriv, length(x))
    end

    FINUFFT.cufinufft_setpts!(state.plan.forPlan, x.+disux, y.+disuy, z.+disuz)
    FINUFFT.cufinufft_setpts!(state.plan.revPlan, x.+disux, y.+disuy, z.+disuz)
end

function setpts!(state::MesoState{T1,T2,3,T3,B}, rho, ux, uy, uz, getDeriv) where{T1,T2,T3,B}
    state.rholessRealSpace .= exp.(-1im .* (state.G[1] .* ux .+ state.G[2] .* uy .+ state.G[3] .* uz))
    state.realSpace .= rho .* state.rholessRealSpace
end

function backProp(state::MesoState{T1,T2,T3,3,false}) where{T1,T2,T3}
    state.plan.tempSpace .= state.plan.recipSpace
    state.tempSpace .= state.recipSpace

    # Calculate the rho derivatives
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

function backProp(state::MesoState{T1,T2,T3,T4,true}) where{T1,T2,T3,T4}
    state.plan.tempSpace .= state.plan.recipSpace
    state.tempSpace .= state.recipSpace

    state.recipSpace .= state.working
    state.plan \ state.recipSpace

    # Calculate the rho derivatives
    state.rhoDeriv.= 0
    state.rhoDeriv .+= real.(state.plan.realSpace) .* real.(state.rholessRealSpace)
    state.rhoDeriv .+= imag.(state.plan.realSpace) .* imag.(state.rholessRealSpace)

    # Calculate the ux derivatives
    state.uxDeriv.= 0
    state.uxDeriv .+= state.G[1] .* (real.(state.plan.realSpace) .* imag.(state.realSpace))
    state.uxDeriv .-= state.G[1] .* (imag.(state.plan.realSpace) .* real.(state.realSpace))

    # Calculate the uy derivatives
    state.uyDeriv.= 0
    state.uyDeriv .+= state.G[2] .* (real.(state.plan.realSpace) .* imag.(state.realSpace))
    state.uyDeriv .-= state.G[2] .* (imag.(state.plan.realSpace) .* real.(state.realSpace))

    # Calculate the uz derivatives
    state.uzDeriv.= 0
    state.uzDeriv .+= state.G[3] .* (real.(state.plan.realSpace) .* imag.(state.realSpace))
    state.uzDeriv .-= state.G[3] .* (imag.(state.plan.realSpace) .* real.(state.realSpace))

    # Calculate the disux derivatives
    state.disuxDeriv.= 0
    state.recipSpace .= state.working .* state.h
    state.plan \ state.recipSpace
    state.disuxDeriv .+= real.(state.plan.realSpace) .* imag.(state.realSpace)
    state.disuxDeriv .-= imag.(state.plan.realSpace) .* real.(state.realSpace)

    # Calculate the disuy derivatives
    state.disuyDeriv.= 0
    state.recipSpace .= state.working .* state.k
    state.plan \ state.recipSpace
    state.disuyDeriv .+= real.(state.plan.realSpace) .* imag.(state.realSpace)
    state.disuyDeriv .-= imag.(state.plan.realSpace) .* real.(state.realSpace)

    # Calculate the disuz derivatives
    state.disuzDeriv.= 0
    state.recipSpace .= state.working .* state.l
    state.plan \ state.recipSpace
    state.disuzDeriv .+= real.(state.plan.realSpace) .* imag.(state.realSpace)
    state.disuzDeriv .-= imag.(state.plan.realSpace) .* real.(state.realSpace)

    state.plan.recipSpace .= state.plan.tempSpace
    state.recipSpace .= state.tempSpace
end

function backProp(state::MesoState{T1,T2,T3,1,false}) where{T1,T2,T3}
    state.plan.tempSpace .= state.plan.recipSpace
    state.tempSpace .= state.recipSpace

    state.recipSpace .= state.working
    state.plan \ state.recipSpace

    # Calculate the rho derivatives
    state.rhoDeriv.= 0
    state.rhoDeriv .+= real.(state.plan.realSpace) .* real.(state.rholessRealSpace)
    state.rhoDeriv .+= imag.(state.plan.realSpace) .* imag.(state.rholessRealSpace)

    # Calculate the ux derivatives
    state.uxDeriv.= 0
    state.uxDeriv .+= state.G[1] .* (real.(state.plan.realSpace) .* imag.(state.realSpace))
    state.uxDeriv .-= state.G[1] .* (imag.(state.plan.realSpace) .* real.(state.realSpace))

    # Calculate the uy derivatives
    state.uyDeriv.= 0
    state.uyDeriv .+= state.G[2] .* (real.(state.plan.realSpace) .* imag.(state.realSpace))
    state.uyDeriv .-= state.G[2] .* (imag.(state.plan.realSpace) .* real.(state.realSpace))

    # Calculate the uz derivatives
    state.uzDeriv.= 0
    state.uzDeriv .+= state.G[3] .* (real.(state.plan.realSpace) .* imag.(state.realSpace))
    state.uzDeriv .-= state.G[3] .* (imag.(state.plan.realSpace) .* real.(state.realSpace))

    state.plan.recipSpace .= state.plan.tempSpace
    state.recipSpace .= state.tempSpace
end

struct MultiState{T1,T2}
    loss::T1
    scale::Bool
    intens::CuArray{Int64, 3, CUDA.Mem.DeviceBuffer}
    G::Vector{Float64}
    h::CuArray{Int64, 3, CUDA.Mem.DeviceBuffer}
    k::CuArray{Int64, 3, CUDA.Mem.DeviceBuffer}
    l::CuArray{Int64, 3, CUDA.Mem.DeviceBuffer}
    plan::T2
    rhoPlan::T2
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

        if losstype == "likelihood"
            loss = PoissonLikelihoodLoss()
        elseif losstype == "L2"
            loss = L2Loss()
        elseif losstype == "Huber"
            loss = HuberLoss(0.5*sqrt(maximum(intens)))
        end

        new{typeof(loss),typeof(plan)}(
            loss, scale, intens, G, h, k, l, plan, rhoPlan, realSpace, rholessRealSpace, 
            recipSpace, tempSpace, recSupport, working, xDeriv, yDeriv, zDeriv, rhoDeriv, 
            uxDeriv, uyDeriv, uzDeriv
        )
    end
end

function setpts!(state::MultiState, x, y, z, mx, my, mz,  rho, ux, uy, uz, getDeriv)
    if maximum(x) > 3*pi || minimum(x) < -3*pi ||
       maximum(y) > 3*pi || minimum(y) < -3*pi ||
       maximum(z) > 3*pi || minimum(z) < -3*pi ||
       maximum(mx+ux) > 3*pi || minimum(mx+ux) < -3*pi ||
       maximum(my+uy) > 3*pi || minimum(my+uy) < -3*pi ||
       maximum(mz+uz) > 3*pi || minimum(mz+uz) < -3*pi
        @warn "Positions are not between -3π and 3π, these need to be scaled."
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

function forwardProp(state, saveRecip)
    state.plan * state.realSpace
    state.plan.recipSpace .+= state.recipSpace

    if saveRecip
        state.recipSpace .= state.plan.recipSpace
    end

    return nothing
end
