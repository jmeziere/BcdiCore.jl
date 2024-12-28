function loss(state, needDeriv, needLoss, saveRecip)
    forwardProp(state, saveRecip)

    c = CUDA.ones(Float64, 1)
    retVal = CUDA.ones(Float64, 1)
    if state.scale
        c .= getScale(state, state.loss)
    end
    if needDeriv
        getPartial(state, state.loss, c)
        backProp(state)
    end
    if needLoss
        retVal .= getLoss(state, state.loss, c)
    end

    return retVal
end

function emptyLoss(state)
    c = CUDA.ones(Float64, 1)
    retVal = CUDA.ones(Float64, 1)
    if state.scale
        c .= getScale(state, state.loss)
    end
    retVal .= getLoss(state, state.loss, c)
    return retVal
end

struct PoissonLikelihoodLoss
end

function getScale(state, loss::PoissonLikelihoodLoss)
    return mapreduce((i,sup) -> sup ? i : 0, +, state.intens, state.recSupport, dims=(1,2,3)) ./
           mapreduce((rsp, sup) -> sup ? abs2(rsp) : 0.0, +, state.plan.recipSpace, state.recSupport, dims=(1,2,3))
end

function getPartial(state, loss::PoissonLikelihoodLoss, c)
    state.working .= 2 .* state.recSupport .* (
        c .* state.plan.recipSpace .-
        state.intens .* exp.(1im .* angle.(state.plan.recipSpace)) ./ abs.(state.plan.recipSpace)
    )
end

function getLoss(state, loss::PoissonLikelihoodLoss, c)
    state.plan.tempSpace .= sqrt.(c) .* state.plan.recipSpace
    return mapreduce(
        (i,rsp,sup) -> sup ? abs2(rsp) - LogExpFunctions.xlogy(i, abs2(rsp)) - i + LogExpFunctions.xlogx(i) : 0.0, +,
        state.intens, state.plan.tempSpace, state.recSupport, dims=(1,2,3)
    )./length(state.recipSpace)
end

struct L2Loss
end

function getScale(state, loss::L2Loss)
    return mapreduce((i,rsp,sup) -> sup ? sqrt(i) * abs(rsp) : 0.0, +, state.intens, state.plan.recipSpace, state.recSupport, dims=(1,2,3)) ./
           mapreduce((rsp,sup) -> sup ? abs2(rsp) : 0.0, +, state.plan.recipSpace, state.recSupport, dims=(1,2,3))
end

function getPartial(state, loss::L2Loss, c)
    state.working .= 2 .* c .* state.recSupport .* (
        c .* state.plan.recipSpace .- sqrt.(state.intens) .* exp.(1im .* angle.(state.plan.recipSpace))
    )
end

function getLoss(state, loss::L2Loss, c)
    state.plan.tempSpace .= c .* state.plan.recipSpace
    return mapreduce(
        (i,rsp,sup) -> sup ? (abs(rsp) - sqrt(i))^2 : 0.0, +,
        state.intens, state.plan.tempSpace, state.recSupport, dims=(1,2,3)
    ) ./ length(state.recipSpace)
end

struct HuberLoss
    delta::Ref{Float64}
    a::CuArray{Float64, 1, CUDA.Mem.DeviceBuffer}
    da::CuArray{Float64, 1, CUDA.Mem.DeviceBuffer}

    function HuberLoss(delta, a=1)
        new(delta, [a], [a])
    end
end

function getScale(state, loss::HuberLoss)
    return mapreduce((i,rsp,sup) -> sup ? sqrt(i) * abs(rsp) : 0.0, +, state.intens, state.plan.recipSpace, state.recSupport, dims=(1,2,3)) ./
           mapreduce((rsp,sup) -> sup ? abs2(rsp) : 0.0, +, state.plan.recipSpace, state.recSupport, dims=(1,2,3))
end

function getPartial(state, loss::HuberLoss, c)
    state.plan.tempSpace .= loss.a .* c .* state.plan.recipSpace
    delta = loss.delta[]
    map!(
        (i,rsp,sup) -> sup ? (
            abs(abs(rsp)-sqrt(i)) <= delta ? 2*(abs(rsp)-sqrt(i))+0.0*1im : 2*delta*sign(abs(rsp)-sqrt(i))+0.0*1im
        ) : 0.0 + 0.0*1im, state.working, state.intens, state.plan.tempSpace, state.recSupport
    ) 

    loss.da .= mapreduce((rsp,w) -> abs(rsp)*real(w), +, state.plan.recipSpace, state.working)
    loss.da .*= c ./ length(state.recipSpace)

    state.working .*= loss.a
    if state.scale
        state.working .= 
            mapreduce((rsp,w)->abs(rsp)*real(w), +, state.plan.recipSpace, state.working, dims=(1,2,3)) .* 
            state.recSupport .* (sqrt.(state.intens) .- 2 .* c .* abs.(state.plan.recipSpace)) ./ 
            mapreduce((rsp,sup) -> sup ? abs2(rsp) : 0.0, +, state.plan.recipSpace, state.recSupport, dims=(1,2,3)) .+
            c .* state.working
    end
    state.working .*= exp.(1im .* angle.(state.plan.recipSpace))
end

function getLoss(state, loss::HuberLoss, c)
    state.plan.tempSpace .= loss.a .* c .* state.plan.recipSpace
    delta = loss.delta[]
    return mapreduce(
        (i,rsp,sup) -> sup ? (
            abs(abs(rsp)-sqrt(i)) <= delta ? (abs(rsp) - sqrt(i))^2 : 2*delta*(abs(abs(rsp)-sqrt(i))-delta/2)
        ) : 0.0, +, state.intens, state.plan.tempSpace, state.recSupport, dims=(1,2,3)
    ) ./ length(state.recipSpace)
end

struct L1Reg{I}
    lambda::Float64
    support::CuArray{Bool, I, CUDA.Mem.DeviceBuffer}
    neg::Bool

    function L1Reg(lambda, support; neg=false)
        new{ndims(support)}(lambda, support, neg)
    end
end

function modifyDeriv(state, reg::L1Reg)
    if hasproperty(state, :deriv)
        state.deriv .+= (.-1).^reg.neg .* reg.lambda .* exp.(1im .* angle.(state.realSpace)) .* reg.support
    else
        state.rhoDeriv .+= (.-1).^reg.neg .* reg.lambda .* reg.support
    end
end

function modifyLoss(state, reg::L1Reg)
    return (.-1).^reg.neg .* reg.lambda .* mapreduce(
        (r,s) -> s ? abs(r) : 0.0, +, state.realSpace, reg.support, dims=(1,2,3)
    )
end

struct L2Reg{I}
    lambda::Float64
    support::CuArray{Bool, I, CUDA.Mem.DeviceBuffer}
    neg::Bool

    function L2Reg(lambda, support; neg=false)
        new{ndims(support)}(lambda, support, neg)
    end
end

function modifyDeriv(state, reg::L2Reg)
    if hasproperty(state, :deriv)
        state.deriv .+= (.-1).^reg.neg .* 2 .* reg.lambda .* state.realSpace .* reg.support
    else
        state.rhoDeriv .+= (.-1).^reg.neg .* 2 .* reg.lambda .* abs.(state.realSpace) .* reg.support
    end
end

function modifyLoss(state, reg::L2Reg)
    return (.-1).^reg.neg .* reg.lambda .* mapreduce(
        (r,s) -> s ? abs2(r) : 0.0, +, state.realSpace, reg.support, dims=(1,2,3)
    )
end

struct TVMagReg
    lambda::Float64
    neighbors::CuArray{Int64, 2, CUDA.Mem.DeviceBuffer}

    function TVMagReg(lambda, neighbors)
        newNeighs = CUDA.zeros(Int64, size(neighbors))
        newNeighs[neighbors .!= nothing] .= neighbors[neighbors .!= nothing]
        new(lambda, newNeighs)
    end
end

function modifyDeriv(state, reg::TVMagReg)
    if hasproperty(state, :deriv)
        for i in 1:6
            inds = reg.neighbors[i,:] .!= 0
            neighs = reg.neighbors[i,inds]
            @views state.deriv[inds] .+= reg.lambda .* sign.(
                abs.(state.realSpace[inds]) .- abs.(state.realSpace[neighs])
            ) .* exp.(1im .* angle.(state.realSpace[inds])) ./ 3
        end
    else
        for i in 1:6
            inds = reg.neighbors[i,:] .!= 0
            neighs = reg.neighbors[i,inds]
            @views state.rhoDeriv[inds] .+= reg.lambda .* sign.(
                abs.(state.realSpace[inds]) .- abs.(state.realSpace[neighs])
            ) ./ 3
        end
    end
end

function modifyLoss(state, reg::TVMagReg)
    mLoss = CUDA.zeros(Float64, 1)
    for i in 1:6
        inds = reg.neighbors[i,:] .!= 0
        @views mLoss .+= mapreduce(
            (r,n) -> abs(abs(r) - abs(n)), +, 
            state.realSpace[inds], state.realSpace[reg.neighbors[i,inds]], dims=(1)
        )
    end
    return reg.lambda .* mLoss ./ 6
end

struct TVReg
    lambda::Float64
    neighbors::CuArray{Int64, 2, CUDA.Mem.DeviceBuffer}
    working::CuArray{ComplexF64, 1, CUDA.Mem.DeviceBuffer}

    function TVReg(lambda, neighbors)
        newNeighs = CUDA.zeros(Int64, size(neighbors))
        newNeighs[neighbors .!= nothing] .= neighbors[neighbors .!= nothing]
        working = CUDA.zeros(Float64, size(neighbors, 2))
        new(lambda, newNeighs, working)
    end
end

function modifyDeriv(state, reg::TVReg)
    if hasproperty(state, :deriv)
        for i in 1:6
            inds = reg.neighbors[i,:] .!= 0
            neighs = reg.neighbors[i,inds]
            @views state.deriv[inds] .+= reg.lambda .* exp.(1im .* angle.(
                state.realSpace[inds] .- state.realSpace[neighs]
            )) ./ 3
        end
    else
        for i in 1:6
            @views inds = reg.neighbors[i,:] .!= 0
            @views neighs = reg.neighbors[i,inds]

            @views reg.working[inds] .= reg.lambda .* exp.(1im .* angle.(
                state.realSpace[inds] .- state.realSpace[neighs]
            )) ./ 3

            @views state.rhoDeriv[inds] .+= (
                real.(reg.working[inds]) .* cos.(angle.(state.realSpace[inds])) .+
                imag.(reg.working[inds]) .* sin.(angle.(state.realSpace[inds]))
            )
            @views state.uxDeriv[inds] .+= state.G[1] .* (
                real.(reg.working[inds]) .* imag.(state.realSpace[inds]) .-
                imag.(reg.working[inds]) .* real.(state.realSpace[inds])
            )
            @views state.uyDeriv[inds] .+= state.G[2] .* (
                real.(reg.working[inds]) .* imag.(state.realSpace[inds]) .-
                imag.(reg.working[inds]) .* real.(state.realSpace[inds])
            )
            @views state.uzDeriv[inds] .+= state.G[3] .* (
                real.(reg.working[inds]) .* imag.(state.realSpace[inds]) .-
                imag.(reg.working[inds]) .* real.(state.realSpace[inds])
            )
        end
    end
end

function modifyLoss(state, reg::TVReg)
    mLoss = CUDA.zeros(Float64, 1)
    for i in 1:6
        @views inds = reg.neighbors[i,:] .!= 0
        @views mLoss .+= mapreduce(
            (r,n) -> abs(r - n), +, 
            state.realSpace[inds], state.realSpace[reg.neighbors[i,inds]], dims=(1)
        )
    end
    return reg.lambda .* mLoss ./ 6
end

struct BetaReg
    lambda::Float64
    a::Float64
    b::Float64
    c::Float64
    m::Ref{Float64}

    function BetaReg(lambda, a, b, c, m=1)
        new(lambda, a, b, c, m)
    end
end

function modifyDeriv(state, reg::BetaReg)
    a = reg.a
    b = reg.b
    c = reg.c
    m = reg.m[]
    if hasproperty(state, :deriv)
        state.deriv .+= reg.lambda .* (
            a .* (abs.(state.realSpace) .+ 0.001).^(a-1) .* (m+0.001 .- abs.(state.realSpace)).^b .-
            b .* (abs.(state.realSpace) .+ 0.001).^a .* (m+0.001 .- abs.(state.realSpace)).^(b-1) .+ c
        ).* exp.(1im .* angle.(state.realSpace))
    else
        state.rhoDeriv .+= reg.lambda .* (
            a .* (abs.(state.realSpace) .+ 0.001).^(a-1) .* (m+0.001 .- abs.(state.realSpace)).^b .-
            b .* (abs.(state.realSpace) .+ 0.001).^a .* (m+0.001 .- abs.(state.realSpace)).^(b-1) .+ c
        )
    end
end

function modifyLoss(state, reg::BetaReg)
    a = reg.a
    b = reg.b
    c = reg.c
    m = reg.m[]
    return reg.lambda .* mapreduce(x -> (abs(x)+0.001)^a*(m+0.001-abs(x))^b+c*abs(x), +, state.realSpace, dims=(1,2,3))
end

function manyLikeScal!(losses, x, y, z, adds, intens, recipSpace, h, k, l, recSupport, gh, gk, gl)
    recStart = threadIdx().x
    recStride = blockDim().x
    atInd = blockIdx().x

    cache = @cuDynamicSharedMem(Float64, 3*recStride)
    cache[recStart] = 0
    cache[recStart + recStride] = 0
    cache[recStart + 2*recStride] = 0
    for recInd in recStart:recStride:length(intens)
        if !recSupport[recInd]
            continue
        end
        rsp = recipSpace[recInd] + (2 * adds[atInd] - 1) * exp(-1im * (
            x[atInd] * (h[recInd]+gh) +
            y[atInd] * (k[recInd]+gk) +
            z[atInd] * (l[recInd]+gl)
        ))

        cache[recStart] += LogExpFunctions.xlogy(intens[recInd], intens[recInd]/abs2(rsp))
        cache[recStart + recStride] += intens[recInd]
        cache[recStart + 2*recStride] += abs2(rsp)
    end

    sync_threads()

    i = div(blockDim().x, 2)
    while i != 0
        if recStart <= i
            cache[recStart] += cache[recStart + i]
            cache[recStart + recStride] += cache[recStart + i + recStride]
            cache[recStart + 2*recStride] += cache[recStart + i + 2*recStride]
        end
        sync_threads()
        i = div(i, 2)
    end

    if recStart == 1
        losses[atInd] += (cache[1] - LogExpFunctions.xlogy(cache[1 + recStride],cache[1 + recStride]/cache[1 + 2*recStride])) / length(intens)
    end
    nothing
end

function manyLikeNoScal!(losses, x, y, z, adds, intens, recipSpace, h, k, l, recSupport, gh, gk, gl)
    recStart = threadIdx().x
    recStride = blockDim().x
    atInd = blockIdx().x

    cache = @cuDynamicSharedMem(Float64, recStride)
    cache[recStart] = 0
    for recInd in recStart:recStride:length(intens)
        if !recSupport[recInd]
            continue
        end
        rsp = recipSpace[recInd] + (2 * adds[atInd] - 1) * exp(-1im * (
            x[atInd] * (h[recInd]+gh) +
            y[atInd] * (k[recInd]+gk) +
            z[atInd] * (l[recInd]+gl)
        ))
        cache[recStart] += abs2(rsp) + LogExpFunctions.xlogy(intens[recInd],intens[recInd]/abs2(rsp)) - intens[recInd]
    end

    sync_threads()

    i = div(blockDim().x, 2)
    while i != 0
        if recStart <= i
            cache[recStart] += cache[recStart + i]
        end
        sync_threads()
        i = div(i, 2)
    end

    if recStart == 1
        losses[atInd] += cache[1] / length(intens)
    end
    nothing
end

function manyL2Scal!(losses, x, y, z, adds, intens, recipSpace, h, k, l, recSupport, gh, gk, gl)
    recStart = threadIdx().x 
    recStride = blockDim().x 
    atInd = blockIdx().x

    cache = @cuDynamicSharedMem(Float64, 3*recStride)
    cache[recStart] = 0
    cache[recStart + recStride] = 0
    cache[recStart + 2*recStride] = 0
    for recInd in recStart:recStride:length(intens)
        if !recSupport[recInd]
            continue
        end
        rsp = recipSpace[recInd] + (2 * adds[atInd] - 1) * exp(-1im * (
            x[atInd] * (h[recInd]+gh) +
            y[atInd] * (k[recInd]+gk) +
            z[atInd] * (l[recInd]+gl)
        ))

        cache[recStart] += intens[recInd]
        cache[recStart + recStride] += sqrt(intens[recInd]) * abs(rsp)
        cache[recStart + 2*recStride] += abs2(rsp)
    end

    sync_threads()

    i = div(blockDim().x, 2)
    while i != 0
        if recStart <= i
            cache[recStart] += cache[recStart + i]
            cache[recStart + recStride] += cache[recStart + i + recStride]
            cache[recStart + 2*recStride] += cache[recStart + i + 2*recStride]
        end
        sync_threads()
        i = div(i, 2)
    end

    if recStart == 1
        losses[atInd] += (cache[1] - cache[1 + recStride]^2 / cache[1 + 2*recStride]) / length(intens)
    end
    nothing
end

function manyL2NoScal!(losses, x, y, z, adds, intens, recipSpace, h, k, l, recSupport, gh, gk, gl)
    recStart = threadIdx().x
    recStride = blockDim().x
    atInd = blockIdx().x

    cache = @cuDynamicSharedMem(Float64, recStride)
    cache[recStart] = 0
    for recInd in recStart:recStride:length(intens)
        if !recSupport[recInd]
            continue
        end
        rsp = recipSpace[recInd] + (2 * adds[atInd] - 1) * exp(-1im * (
            x[atInd] * (h[recInd]+gh) +
            y[atInd] * (k[recInd]+gk) +
            z[atInd] * (l[recInd]+gl)
        ))

        cache[recStart] += (abs(rsp) - sqrt(intens[recInd]))^2
    end

    sync_threads()

    i = div(blockDim().x, 2)
    while i != 0
        if recStart <= i
            cache[recStart] += cache[recStart + i]
        end
        sync_threads()
        i = div(i, 2)
    end
    
    if recStart == 1
        losses[atInd] += cache[1] / length(intens)
    end
    nothing
end

function lossManyAtomic!(losses, state, x, y, z, adds, addLoss)
    if !addLoss
        losses .= 0
    end

    threads=state.manyThreads
    blocks=length(x)
    shmem=3 * threads * sizeof(Float64)
    state.many!(
        losses, x, y, z, adds, state.intens, state.recipSpace, state.h, state.k, state.l, state.recSupport, 
        state.G[1], state.G[2], state.G[3]; threads=threads, blocks=blocks, shmem=shmem
    )
end
