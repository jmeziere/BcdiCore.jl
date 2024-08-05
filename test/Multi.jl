function multiLikelihoodWithScaling(x, y, z, mx, my, mz, rho, ux, uy, uz, h, k, l, hp, kp, lp, intens, recSupport)
    diffType = Float64
    if typeof(x[1]) != Float64
        diffType = typeof(x[1])
    elseif typeof(y[1]) != Float64
        diffType = typeof(y[1])
    elseif typeof(z[1]) != Float64
        diffType = typeof(z[1])
    elseif typeof(rho[1]) != Float64
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
            recipSpace[i] += exp(-1im * (x[j] * hp[i] + y[j] * kp[i] + z[j] * lp[i]))
        end
        for j in 1:length(mx)
            recipSpace[i] += rho[j] * exp(-1im * (mx[j] * h[i] + my[j] * k[i] + mz[j] * l[i] + ux[j] * hp[i] + uy[j] * kp[i] + uz[j] * lp[i]))
        end
    end
    recipSpace .*= recSupport
    intens = intens .* recSupport
    c = reduce(+, intens)/mapreduce(abs2, +, recipSpace)
    return mapreduce((i,r) -> c*abs2(r) - LogExpFunctions.xlogy(i,c*abs2(r)) - i + LogExpFunctions.xlogx(i), +, intens, recipSpace)/length(intens)
end

function multiLikelihoodWithoutScaling(x, y, z, mx, my, mz, rho, ux, uy, uz, h, k, l, hp, kp, lp, intens, recSupport)
    diffType = Float64
    if typeof(x[1]) != Float64
        diffType = typeof(x[1])
    elseif typeof(y[1]) != Float64
        diffType = typeof(y[1])
    elseif typeof(z[1]) != Float64
        diffType = typeof(z[1])
    elseif typeof(rho[1]) != Float64
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
            recipSpace[i] += exp(-1im * (x[j] * hp[i] + y[j] * kp[i] + z[j] * lp[i]))
        end
        for j in 1:length(mx)
            recipSpace[i] += rho[j] * exp(-1im * (mx[j] * h[i] + my[j] * k[i] + mz[j] * l[i] + ux[j] * hp[i] + uy[j] * kp[i] + uz[j] * lp[i]))
        end
    end
    recipSpace .*= recSupport
    intens = intens .* recSupport
    return mapreduce((i,r) -> abs2(r) - LogExpFunctions.xlogy(i,abs2(r)) - i + LogExpFunctions.xlogx(i), +, intens, recipSpace)/length(intens)
end

function multiL2WithScaling(x, y, z, mx, my, mz, rho, ux, uy, uz, h, k, l, hp, kp, lp, intens, recSupport)
    diffType = Float64
    if typeof(x[1]) != Float64
        diffType = typeof(x[1])
    elseif typeof(y[1]) != Float64
        diffType = typeof(y[1])
    elseif typeof(z[1]) != Float64
        diffType = typeof(z[1])
    elseif typeof(rho[1]) != Float64
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
            recipSpace[i] += exp(-1im * (x[j] * hp[i] + y[j] * kp[i] + z[j] * lp[i]))
        end
        for j in 1:length(mx)
            recipSpace[i] += rho[j] * exp(-1im * (mx[j] * h[i] + my[j] * k[i] + mz[j] * l[i] + ux[j] * hp[i] + uy[j] * kp[i] + uz[j] * lp[i]))
        end
    end
    absRecipSpace = abs.(recipSpace) .* recSupport
    sqIntens = sqrt.(intens) .* recSupport
    c = mapreduce((sqi,absr)-> sqi*absr, +, sqIntens, absRecipSpace)/mapreduce(x -> x^2, +, absRecipSpace)
    return mapreduce((sqi,absr) -> (c*absr - sqi)^2, +, sqIntens, absRecipSpace)/length(intens)
end

function multiL2WithoutScaling(x, y, z, mx, my, mz, rho, ux, uy, uz, h, k, l, hp, kp, lp, intens, recSupport)
    diffType = Float64
    if typeof(x[1]) != Float64
        diffType = typeof(x[1])
    elseif typeof(y[1]) != Float64
        diffType = typeof(y[1])
    elseif typeof(z[1]) != Float64
        diffType = typeof(z[1])
    elseif typeof(rho[1]) != Float64
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
            recipSpace[i] += exp(-1im * (x[j] * hp[i] + y[j] * kp[i] + z[j] * lp[i]))
        end
        for j in 1:length(mx)
            recipSpace[i] += rho[j] * exp(-1im * (mx[j] * h[i] + my[j] * k[i] + mz[j] * l[i] + ux[j] * hp[i] + uy[j] * kp[i] + uz[j] * lp[i]))
        end
    end
    absRecipSpace = abs.(recipSpace) .* recSupport
    sqIntens = sqrt.(intens) .* recSupport
    return mapreduce((sqi,absr) -> (absr - sqi)^2, +, sqIntens, absRecipSpace)/length(intens)
end
