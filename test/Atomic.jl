function atomicLikelihoodWithScaling(x, y, z, h, k, l, intens, recSupport)
    diffType = Float64
    if typeof(x[1]) != Float64
        diffType = typeof(x[1])
    elseif typeof(y[1]) != Float64
        diffType = typeof(y[1])
    elseif typeof(z[1]) != Float64
        diffType = typeof(z[1])
    end
    recipSpace = zeros(Complex{diffType}, 4,4,4)
    for i in 1:length(h)
        for j in 1:length(x)
            recipSpace[i] += exp(-1im * (x[j] * h[i] + y[j] * k[i] + z[j] * l[i]))
        end
    end
    recipSpace .*= recSupport
    intens = intens .* recSupport
    c = reduce(+, intens)/mapreduce(abs2, +, recipSpace)
    return mapreduce((i,r) -> c*abs2(r) - LogExpFunctions.xlogy(i,c*abs2(r)) - i + LogExpFunctions.xlogx(i), +, intens, recipSpace)/length(intens)
end

function atomicLikelihoodWithoutScaling(x, y, z, h, k, l, intens, recSupport)
    diffType = Float64
    if typeof(x[1]) != Float64
        diffType = typeof(x[1])
    elseif typeof(y[1]) != Float64
        diffType = typeof(y[1])
    elseif typeof(z[1]) != Float64
        diffType = typeof(z[1])
    end
    recipSpace = zeros(Complex{diffType}, 4,4,4)
    for i in 1:length(h)
        for j in 1:length(x)
            recipSpace[i] += exp(-1im * (x[j] * h[i] + y[j] * k[i] + z[j] * l[i]))
        end
    end
    recipSpace .*= recSupport
    intens = intens .* recSupport
    return mapreduce((i,r) -> abs2(r) - LogExpFunctions.xlogy(i,abs2(r)) - i + LogExpFunctions.xlogx(i), +, intens, recipSpace)/length(intens)
end

function atomicL2WithScaling(x, y, z, h, k, l, intens, recSupport)
    diffType = Float64
    if typeof(x[1]) != Float64
        diffType = typeof(x[1])
    elseif typeof(y[1]) != Float64
        diffType = typeof(y[1])
    elseif typeof(z[1]) != Float64
        diffType = typeof(z[1])
    end
    recipSpace = zeros(Complex{diffType}, 4,4,4)
    for i in 1:length(h)
        for j in 1:length(x)
            recipSpace[i] += exp(-1im * (x[j] * h[i] + y[j] * k[i] + z[j] * l[i]))
        end
    end
    absRecipSpace = abs.(recipSpace) .* recSupport
    sqIntens = sqrt.(intens) .* recSupport
    c = mapreduce((sqi,absr)-> sqi*absr, +, sqIntens, absRecipSpace)/mapreduce(x -> x^2, +, absRecipSpace)
    return mapreduce((sqi,absr) -> (c*absr - sqi)^2, +, sqIntens, absRecipSpace)/length(intens)
end

function atomicL2WithoutScaling(x, y, z, h, k, l, intens, recSupport)
    diffType = Float64
    if typeof(x[1]) != Float64
        diffType = typeof(x[1])
    elseif typeof(y[1]) != Float64
        diffType = typeof(y[1])
    elseif typeof(z[1]) != Float64
        diffType = typeof(z[1])
    end
    recipSpace = zeros(Complex{diffType}, 4,4,4)
    for i in 1:length(h)
        for j in 1:length(x)
            recipSpace[i] += exp(-1im * (x[j] * h[i] + y[j] * k[i] + z[j] * l[i]))
        end
    end
    absRecipSpace = abs.(recipSpace) .* recSupport
    sqIntens = sqrt.(intens) .* recSupport
    return mapreduce((sqi,absr) -> (absr - sqi)^2, +, sqIntens, absRecipSpace)/length(intens)
end
