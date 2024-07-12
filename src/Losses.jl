function loss(state, getDeriv, getLoss)
    forwardProp(state)

    if state.losstype == "likelihood"
        c = 1.0
        if state.scale
            c = reduce(+, state.intens) / mapreduce(abs2, +, state.plan.recipSpace)
        end
        if getDeriv
            state.working .= 2 .* (c .- state.intens ./ abs2.(state.plan.recipSpace))
            backProp(state)
        end
        if getLoss
            state.plan.recipSpace .*= sqrt(c)
            return mapreduce((i,rsp) -> abs2(rsp) - LogExpFunctions.xlogy(i, abs2(rsp)), +, state.intens, state.plan.recipSpace)/length(state.recipSpace)
        end
    elseif state.losstype == "L2"
        c = 1.0
        if state.scale
            c = mapreduce((i,rsp) -> sqrt(i) * abs(rsp), +, state.intens, state.plan.recipSpace) / mapreduce(abs2, +, state.plan.recipSpace)
        end
        if getDeriv
            state.working .= 2 .* c .* (c .- sqrt.(state.intens) ./ abs.(state.plan.recipSpace))
            backProp(state)
        end
        if getLoss
            state.plan.recipSpace .*= c
            return mapreduce((i,rsp) -> (abs(rsp) - sqrt(i))^2, +, state.intens, state.plan.recipSpace)/length(state.recipSpace)
        end
    end
    return 0.0
end


