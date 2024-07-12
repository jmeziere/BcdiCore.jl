function mesoLikelihoodWithScaling(x, y, z, rho, ux, uy, uz, h, k, l, hp, kp, lp, intens)
    diffType = Float64
    if typeof(rho[1]) != Float64
        diffType = typeof(rho[1])
    elseif typeof(ux[1]) != Float64
        diffType = typeof(ux[1])
    elseif typeof(uy[1]) != Float64
        diffType = typeof(uy[1])
    elseif typeof(uz[1]) != Float64
        diffType = typeof(uz[1])
    end
    recipSpace = zeros(Complex{diffType}, 4,4,4)
    for i in 1:length(h)
        for j in 1:length(x)
            recipSpace[i] += rho[j] * exp(-1im * (x[j] * h[i] + y[j] * k[i] + z[j] * l[i] + ux[j] * hp[i] + uy[j] * kp[i] + uz[j] * lp[i]))
        end
    end
    c = reduce(+, intens)/mapreduce(abs2, +, recipSpace)
    return mapreduce((i,r) -> c*abs2(r) - LogExpFunctions.xlogy(i,c*abs2(r)), +, intens, recipSpace)/length(intens)
end

function mesoLikelihoodWithoutScaling(x, y, z, rho, ux, uy, uz, h, k, l, hp, kp, lp, intens)
    diffType = Float64
    if typeof(rho[1]) != Float64
        diffType = typeof(rho[1])
    elseif typeof(ux[1]) != Float64
        diffType = typeof(ux[1])
    elseif typeof(uy[1]) != Float64
        diffType = typeof(uy[1])
    elseif typeof(uz[1]) != Float64
        diffType = typeof(uz[1])
    end
    recipSpace = zeros(Complex{diffType}, 4,4,4)
    for i in 1:length(h)
        for j in 1:length(x)
            recipSpace[i] += rho[j] * exp(-1im * (x[j] * h[i] + y[j] * k[i] + z[j] * l[i] + ux[j] * hp[i] + uy[j] * kp[i] + uz[j] * lp[i]))
        end
    end
    return mapreduce((i,r) -> abs2(r) - LogExpFunctions.xlogy(i,abs2(r)), +, intens, recipSpace)/length(intens)
end

function mesoL2WithScaling(x, y, z, rho, ux, uy, uz, h, k, l, hp, kp, lp, intens)
    diffType = Float64
    if typeof(rho[1]) != Float64
        diffType = typeof(rho[1])
    elseif typeof(ux[1]) != Float64
        diffType = typeof(ux[1])
    elseif typeof(uy[1]) != Float64
        diffType = typeof(uy[1])
    elseif typeof(uz[1]) != Float64
        diffType = typeof(uz[1])
    end
    recipSpace = zeros(Complex{diffType}, 4,4,4)
    for i in 1:length(h)
        for j in 1:length(x)
            recipSpace[i] += rho[j] * exp(-1im * (x[j] * h[i] + y[j] * k[i] + z[j] * l[i] + ux[j] * hp[i] + uy[j] * kp[i] + uz[j] * lp[i]))
        end
    end
    c = mapreduce((i,r)-> sqrt(i)*abs(r), +, intens, recipSpace)/mapreduce(abs2, +, recipSpace)
    return mapreduce((i,r) -> (c*abs(r) - sqrt(i))^2, +, intens, recipSpace)/length(intens)
end

function mesoL2WithoutScaling(x, y, z, rho, ux, uy, uz, h, k, l, hp, kp, lp, intens)
    diffType = Float64
    if typeof(rho[1]) != Float64
        diffType = typeof(rho[1])
    elseif typeof(ux[1]) != Float64
        diffType = typeof(ux[1])
    elseif typeof(uy[1]) != Float64
        diffType = typeof(uy[1])
    elseif typeof(uz[1]) != Float64
        diffType = typeof(uz[1])
    end
    recipSpace = zeros(Complex{diffType}, 4,4,4)
    for i in 1:length(h)
        for j in 1:length(x)
            recipSpace[i] += rho[j] * exp(-1im * (x[j] * h[i] + y[j] * k[i] + z[j] * l[i] + ux[j] * hp[i] + uy[j] * kp[i] + uz[j] * lp[i]))
        end
    end
    return mapreduce((i,r) -> (abs(r) - sqrt(i))^2, +, intens, recipSpace)/length(intens)
end
