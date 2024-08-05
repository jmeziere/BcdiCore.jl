function loss(state, getDeriv, getLoss, saveRecip)
    forwardProp(state, saveRecip)

    if state.losstype == 0
        c = 1.0
        if state.scale
            c = mapreduce((i,sup) -> sup ? i : 0.0, +, state.intens, state.recSupport, init=0.0) / 
                mapreduce((rsp, sup) -> sup ? abs2(rsp) : 0.0, +, state.plan.recipSpace, state.recSupport, init=0.0)
        end
        if getDeriv
            state.working .= 2 .* state.recSupport .* (c .* state.plan.recipSpace .- state.intens .* exp.(1im .* angle.(state.plan.recipSpace)) ./ abs.(state.plan.recipSpace))
            backProp(state)
        end
        if getLoss
            state.plan.tempSpace .= sqrt(c) .* state.plan.recipSpace
            return mapreduce(
                (i,rsp,sup) -> sup ? abs2(rsp) - LogExpFunctions.xlogy(i, abs2(rsp)) - i + LogExpFunctions.xlogx(i) : 0.0, +, 
                state.intens, state.plan.tempSpace, state.recSupport, init = 0.0
            )/length(state.recipSpace)
        end
    elseif state.losstype == 1
        c = 1.0
        if state.scale
            c = mapreduce((i,rsp,sup) -> sup ? sqrt(i) * abs(rsp) : 0.0, +, state.intens, state.plan.recipSpace, state.recSupport, init=0.0) /
                mapreduce((rsp,sup) -> sup ? abs2(rsp) : 0.0, +, state.plan.recipSpace, state.recSupport, init=0.0)
        end
        if getDeriv
            state.working .= 2 .* c .* state.recSupport .* (c .* state.plan.recipSpace .- sqrt.(state.intens) .* exp.(1im .* angle.(state.plan.recipSpace)))
            backProp(state)
        end
        if getLoss
            state.plan.tempSpace .= c .* state.plan.recipSpace
            return mapreduce(
                (i,rsp,sup) -> sup ? (abs(rsp) - sqrt(i))^2 : 0.0, +, 
                state.intens, state.plan.tempSpace, state.recSupport
            )/length(state.recipSpace)
        end
    end
    return 0.0
end

function emptyLoss(state)
    state.plan.recipSpace .= state.recipSpace

    if state.losstype == 0
        c = 1.0
        if state.scale
            c = mapreduce((i,sup) -> sup ? i : 0.0, +, state.intens, state.recSupport, init=0.0) / 
                mapreduce((rsp, sup) -> sup ? abs2(rsp) : 0.0, +, state.plan.recipSpace, state.recSupport, init=0.0)
        end
        state.plan.tempSpace .= sqrt(c) .* state.plan.recipSpace
        return mapreduce(
            (i,rsp,sup) -> sup ? abs2(rsp) - LogExpFunctions.xlogy(i, abs2(rsp)) - i + LogExpFunctions.xlogx(i) : 0.0, +, 
            state.intens, state.plan.tempSpace, state.recSupport, init = 0.0
        )/length(state.recipSpace)
    elseif state.losstype == 1
        c = 1.0
        if state.scale
            c = mapreduce((i,rsp,sup) -> sup ? sqrt(i) * abs(rsp) : 0.0, +, state.intens, state.plan.recipSpace, state.recSupport) /
                mapreduce((rsp,sup) -> sup ? abs2(rsp) : 0.0, +, state.plan.recipSpace, state.recSupport)
        end
        state.plan.tempSpace .= c .* state.plan.recipSpace
        return mapreduce(
            (i,rsp,sup) -> sup ? (abs(rsp) - sqrt(i))^2 : 0.0, +, 
            state.intens, state.plan.tempSpace, state.recSupport
        )/length(state.recipSpace)
    end
end

function lossManyAtomic!(losses, losstype, x, y, z, adds, scalings, intens, recipSpace, h, k, l, recSupport, gh, gk, gl)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    for i in index:stride:length(losses)
        for j in CartesianIndices(intens)
            if !recSupport[j]
                continue
            end
            rsp = recipSpace[j] + (2 * adds[i] - 1) * exp(-1im * (
                x[i] * (h[j]+gh) +
                y[i] * (k[j]+gk) +
                z[i] * (l[j]+gl)
            ))

            if losstype == 0
                losses[i] += (
                    scalings[i]*abs2(rsp) - LogExpFunctions.xlogy(intens[j],scalings[i]*abs2(rsp)) -
                    intens[j] + LogExpFunctions.xlogx(intens[j])
                )/length(intens)
            elseif losstype == 1
                losses[i] += ((scalings[i]*abs(rsp) - sqrt(intens[j]))^2)/length(intens)
            end
        end
    end
end

function scalingManyAtomic!(scalings, losstype, x, y, z, adds, intens, recipSpace, h, k, l, recSupport, gh, gk, gl)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    for i in index:stride:length(scalings)
        num = 0.0
        den = 0.0
        for j in CartesianIndices(intens)
            if !recSupport[j]
                continue
            end
            rsp = recipSpace[j] + (2 * adds[i] - 1) * exp(-1im * (
                x[i] * (h[j]+gh) +
                y[i] * (k[j]+gk) +
                z[i] * (l[j]+gl)
            ))

            if losstype == 0
                num += intens[j]
                den += abs2(rsp)
            elseif losstype == 1
                num += sqrt(intens[j]) * abs(rsp)
                den += abs2(rsp)
            end
        end
        scalings[i] = num/den
    end
end

function lossManyAtomic!(losses, state, x, y, z, adds, addLoss)
    if !addLoss
        losses .= 0
    end
    scalings = CUDA.ones(Float64, length(x))

    if state.scale
        threads = min(length(losses), state.scaleThreads)
        blocks = cld(length(losses), threads)
        state.scalingManyAtomicKernel!(scalings, state.losstype, x, y, z, adds, state.intens, state.recipSpace, state.h, state.k, state.l, state.recSupport, state.G[1], state.G[2], state.G[3]; threads, blocks)
    end

    threads = min(length(losses), state.lossThreads)
    blocks = cld(length(losses), threads)
    state.lossManyAtomicKernel!(losses, state.losstype, x, y, z, adds, scalings, state.intens, state.recipSpace, state.h, state.k, state.l, state.recSupport, state.G[1], state.G[2], state.G[3]; threads, blocks)
end
