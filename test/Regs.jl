function complexTVR(realSpace, lambda, neighbors)
    loss = 0.0
    for i in 1:length(realSpace)
        for j in 1:6
            loss += abs(abs(realSpace[i]) - abs(realSpace[neighbors[j,i]]))
        end
    end
    loss *= lambda/(6*4^3)
    return loss
end

function floatTVR(rho, lambda, neighbors)
    loss = 0.0
    for i in 1:length(rho)
        for j in 1:6
            loss += abs(rho[i] - rho[neighbors[j,i]])
        end
    end
    loss *= lambda/(6*length(rho))
    return loss
end
