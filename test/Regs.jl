function TVRMag(realSpace, lambda, neighbors)
    loss = 0.0
    for i in 1:length(realSpace)
        for j in 1:6
            loss += abs(abs(realSpace[i]) - abs(realSpace[neighbors[j,i]]))
        end
    end
    loss *= lambda/6
    return loss
end

function TVR(realSpace, lambda, neighbors)
    loss = 0.0
    for i in 1:length(realSpace)
        for j in 1:6
            loss += abs(realSpace[i] - realSpace[neighbors[j,i]])
        end
    end
    loss *= lambda/6
    return loss
end

function TVR(rho, ux, uy, uz, G, lambda, neighbors)
    loss = 0.0
    realSpace = rho .* exp.(-1im .* (ux .* G[1] .+ uy .* G[2] .+ uz .* G[3]))
    for i in 1:length(realSpace)
        for j in 1:6
            loss += abs(realSpace[i] - realSpace[neighbors[j,i]])
        end
    end
    loss *= lambda/6
    return loss
end

function BetaR(realSpace, lambda, a, b, c)
    loss = 0.0
    for i in 1:length(realSpace)
        loss += (abs(realSpace[i])+0.001)^a*(1.001-abs(realSpace[i]))^b+c*abs(realSpace[i])
    end
    loss *= lambda
    return loss
end

function L2R(realSpace, lambda)
    loss = 0.0
    for i in 1:length(realSpace)
        loss += abs2(realSpace[i])
    end
    loss *= lambda
    return loss
end
