function mesoLikelihoodWithScaling(x, y, z, rho, ux, uy, uz, h, k, l, G, intens, recSupport)
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
            recipSpace[i] += rho[j] * exp(-1im * (x[j] * h[i] + y[j] * k[i] + z[j] * l[i] + ux[j] * (G[1]+h[i]) + uy[j] * (G[2]+k[i]) + uz[j] * (G[3]+l[i])))
        end
    end
    recipSpace .*= recSupport
    intens = intens .* recSupport
    c = reduce(+, intens)/mapreduce(abs2, +, recipSpace)
    return mapreduce((i,r) -> c*abs2(r) - LogExpFunctions.xlogy(i,c*abs2(r)) - i + LogExpFunctions.xlogx(i), +, intens, recipSpace)/length(intens)
end

function mesoLikelihoodWithoutScaling(x, y, z, rho, ux, uy, uz, h, k, l, G, intens, recSupport)
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
            recipSpace[i] += rho[j] * exp(-1im * (x[j] * h[i] + y[j] * k[i] + z[j] * l[i] + ux[j] * (G[1]+h[i]) + uy[j] * (G[2]+k[i]) + uz[j] * (G[3]+l[i])))
        end
    end
    recipSpace .*= recSupport
    intens = intens .* recSupport
    return mapreduce((i,r) -> abs2(r) - LogExpFunctions.xlogy(i,abs2(r)) - i + LogExpFunctions.xlogx(i), +, intens, recipSpace)/length(intens)
end

function mesoL2WithScaling(x, y, z, rho, ux, uy, uz, h, k, l, G, intens, recSupport)
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
            recipSpace[i] += rho[j] * exp(-1im * (x[j] * h[i] + y[j] * k[i] + z[j] * l[i] + ux[j] * (G[1]+h[i]) + uy[j] * (G[2]+k[i]) + uz[j] * (G[3]+l[i])))
        end
    end
    absRecipSpace = abs.(recipSpace) .* recSupport
    sqIntens = sqrt.(intens) .* recSupport
    c = mapreduce((sqi,absr)-> sqi*absr, +, sqIntens, absRecipSpace)/mapreduce(x -> x^2, +, absRecipSpace)
    return mapreduce((sqi,absr) -> (c*absr - sqi)^2, +, sqIntens, absRecipSpace)/length(intens)
end

function mesoL2WithoutScaling(x, y, z, rho, ux, uy, uz, h, k, l, G, intens, recSupport)
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
            recipSpace[i] += rho[j] * exp(-1im * (x[j] * h[i] + y[j] * k[i] + z[j] * l[i] + ux[j] * (G[1]+h[i]) + uy[j] * (G[2]+k[i]) + uz[j] * (G[3]+l[i])))
        end
    end
    absRecipSpace = abs.(recipSpace) .* recSupport
    sqIntens = sqrt.(intens) .* recSupport
    return mapreduce((sqi,absr) -> (absr - sqi)^2, +, sqIntens, absRecipSpace)/length(intens)

end

function mesoLikelihoodWithScaling(rho, ux, uy, uz, G, intens, recSupport)
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
    s1 = size(intens,1)
    s2 = size(intens,2)
    s3 = size(intens,3)
    for i in 0:s1-1
    for j in 0:s2-1
    for k in 0:s3-1
        for l in 0:s1-1
        for m in 0:s2-1
        for n in 0:s3-1
            recipSpace[i+1,j+1,k+1] += rho[l+1,m+1,n+1] * exp(-1im * (ux[l+1,m+1,n+1]*G[1] + uy[l+1,m+1,n+1]*G[2] + uz[l+1,m+1,n+1]*G[3])) * exp(-1im * 2 * pi * (i*l/s1+j*m/s2+k*n/s3))
        end
        end
        end
    end
    end
    end
    recipSpace .*= recSupport
    intens = intens .* recSupport
    c = reduce(+, intens)/mapreduce(abs2, +, recipSpace)
    return mapreduce((i,r) -> c*abs2(r) - LogExpFunctions.xlogy(i,c*abs2(r)) - i + LogExpFunctions.xlogx(i), +, intens, recipSpace)/length(intens)
end

function mesoLikelihoodWithoutScaling(rho, ux, uy, uz, G, intens, recSupport)
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
    s1 = size(intens,1)
    s2 = size(intens,2)
    s3 = size(intens,3)
    for i in 0:s1-1
    for j in 0:s2-1
    for k in 0:s3-1
        for l in 0:s1-1
        for m in 0:s2-1
        for n in 0:s3-1
            recipSpace[i+1,j+1,k+1] += rho[l+1,m+1,n+1] * exp(-1im * (ux[l+1,m+1,n+1]*G[1] + uy[l+1,m+1,n+1]*G[2] + uz[l+1,m+1,n+1]*G[3])) * exp(-1im * 2 * pi * (i*l/s1+j*m/s2+k*n/s3))
        end
        end
        end
    end
    end
    end
    recipSpace .*= recSupport
    intens = intens .* recSupport
    return mapreduce((i,r) -> abs2(r) - LogExpFunctions.xlogy(i,abs2(r)) - i + LogExpFunctions.xlogx(i), +, intens, recipSpace)/length(intens)
end

function mesoL2WithScaling(rho, ux, uy, uz, G, intens, recSupport)
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
    s1 = size(intens,1)
    s2 = size(intens,2)
    s3 = size(intens,3)
    for i in 0:s1-1
    for j in 0:s2-1
    for k in 0:s3-1
        for l in 0:s1-1
        for m in 0:s2-1
        for n in 0:s3-1
            recipSpace[i+1,j+1,k+1] += rho[l+1,m+1,n+1] * exp(-1im * (ux[l+1,m+1,n+1]*G[1] + uy[l+1,m+1,n+1]*G[2] + uz[l+1,m+1,n+1]*G[3])) * exp(-1im * 2 * pi * (i*l/s1+j*m/s2+k*n/s3))
        end
        end
        end
    end
    end
    end
    absRecipSpace = abs.(recipSpace) .* recSupport
    sqIntens = sqrt.(intens) .* recSupport
    c = mapreduce((sqi,absr)-> sqi*absr, +, sqIntens, absRecipSpace)/mapreduce(x -> x^2, +, absRecipSpace)
    return mapreduce((sqi,absr) -> (c*absr - sqi)^2, +, sqIntens, absRecipSpace)/length(intens)
end

function mesoL2WithoutScaling(rho, ux, uy, uz, G, intens, recSupport)
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
    s1 = size(intens,1)
    s2 = size(intens,2)
    s3 = size(intens,3)
    for i in 0:s1-1
    for j in 0:s2-1
    for k in 0:s3-1
        for l in 0:s1-1
        for m in 0:s2-1
        for n in 0:s3-1
            recipSpace[i+1,j+1,k+1] += rho[l+1,m+1,n+1] * exp(-1im * (ux[l+1,m+1,n+1]*G[1] + uy[l+1,m+1,n+1]*G[2] + uz[l+1,m+1,n+1]*G[3])) * exp(-1im * 2 * pi * (i*l/s1+j*m/s2+k*n/s3))
        end
        end
        end
    end
    end
    end
    absRecipSpace = abs.(recipSpace) .* recSupport
    sqIntens = sqrt.(intens) .* recSupport
    return mapreduce((sqi,absr) -> (absr - sqi)^2, +, sqIntens, absRecipSpace)/length(intens)

end
