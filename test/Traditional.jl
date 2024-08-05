function tradLikelihoodWithScaling(realSpace, intens, recSupport)
    recipSpace = zeros(typeof(realSpace[1]), 4,4,4)
    s1 = size(realSpace,1)
    s2 = size(realSpace,2)
    s3 = size(realSpace,3)
    for i in 0:s1-1
    for j in 0:s2-1
    for k in 0:s3-1
        for l in 0:s1-1
        for m in 0:s2-1
        for n in 0:s3-1
            recipSpace[i+1,j+1,k+1] += realSpace[l+1,m+1,n+1] * exp(-1im * 2 * pi * (i*l/s1+j*m/s2+k*n/s3))
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

function tradLikelihoodWithoutScaling(realSpace, intens, recSupport)
    recipSpace = zeros(typeof(realSpace[1]), 4,4,4)
    s1 = size(realSpace,1)
    s2 = size(realSpace,2)
    s3 = size(realSpace,3)
    for i in 0:s1-1
    for j in 0:s2-1
    for k in 0:s3-1
        for l in 0:s1-1
        for m in 0:s2-1
        for n in 0:s3-1
            recipSpace[i+1,j+1,k+1] += realSpace[l+1,m+1,n+1] * exp(-1im * 2 * pi * (i*l/s1+j*m/s2+k*n/s3))
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

function tradL2WithScaling(realSpace, intens, recSupport)
    recipSpace = zeros(typeof(realSpace[1]), 4,4,4)
    s1 = size(realSpace,1)
    s2 = size(realSpace,2)
    s3 = size(realSpace,3)
    for i in 0:s1-1
    for j in 0:s2-1
    for k in 0:s3-1
        for l in 0:s1-1
        for m in 0:s2-1
        for n in 0:s3-1
            recipSpace[i+1,j+1,k+1] += realSpace[l+1,m+1,n+1] * exp(-1im * 2 * pi * (i*l/s1+j*m/s2+k*n/s3))
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

function tradL2WithoutScaling(realSpace, intens, recSupport)
    recipSpace = zeros(typeof(realSpace[1]), 4,4,4)
    s1 = size(realSpace,1)
    s2 = size(realSpace,2)
    s3 = size(realSpace,3)
    for i in 0:s1-1
    for j in 0:s2-1
    for k in 0:s3-1
        for l in 0:s1-1
        for m in 0:s2-1
        for n in 0:s3-1
            recipSpace[i+1,j+1,k+1] += realSpace[l+1,m+1,n+1] * exp(-1im * 2 * pi * (i*l/s1+j*m/s2+k*n/s3))
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
