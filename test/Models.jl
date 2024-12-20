function atomicModel(x, y, z, h, k, l, intens, recSupport)
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

function tradModel(realSpace, intens, recSupport)
    recipSpace = zeros(typeof(realSpace[1]), 4,4,4)
    s1 = size(intens,1)
    s2 = size(intens,2)
    s3 = size(intens,3)
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

function tradModel(realSpace, intens, recSupport, x, y, z)
    recipSpace = zeros(typeof(realSpace[1]), 4,4,4)
    s1 = size(intens,1)
    s2 = size(intens,2)
    s3 = size(intens,3)
    for i in 1:s1
    for j in 1:s2
    for k in 1:s3
        for l in 1:length(x)
            a = i-div(s1,2)-1
            b = j-div(s2,2)-1
            c = k-div(s3,2)-1
            recipSpace[i,j,k] += realSpace[l] * exp(-1im * (a*x[l]+b*y[l]+c*z[l]))
        end
    end
    end
    end
    absRecipSpace = abs.(recipSpace) .* recSupport
    sqIntens = sqrt.(intens) .* recSupport
    return mapreduce((sqi,absr) -> (absr - sqi)^2, +, sqIntens, absRecipSpace)/length(intens)
end

function mesoModel(x, y, z, rho, ux, uy, uz, h, k, l, G, intens, recSupport)
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

function mesoModel(x, y, z, rho, ux, uy, uz, disux, disuy, disuz, h, k, l, G, intens, recSupport)
    diffType = Float64
    if typeof(rho[1]) != Float64
        diffType = typeof(rho[1])
    elseif typeof(ux[1]) != Float64
        diffType = typeof(ux[1])
    elseif typeof(uy[1]) != Float64
        diffType = typeof(uy[1])
    elseif typeof(uz[1]) != Float64
        diffType = typeof(uz[1])
    elseif typeof(disux[1]) != Float64
        diffType = typeof(disux[1])
    elseif typeof(disuy[1]) != Float64
        diffType = typeof(disuy[1])
    elseif typeof(disuz[1]) != Float64
        diffType = typeof(disuz[1])
    end
    recipSpace = zeros(Complex{diffType}, 4,4,4)
    for i in 1:length(h)
        for j in 1:length(x)
            recipSpace[i] += rho[j] * exp(-1im * (
                x[j] * h[i] + y[j] * k[i] + z[j] * l[i] + ux[j] * G[1] + uy[j] * G[2] + uz[j] * G[3] + disux[j] * h[i] + disuy[j] * k[i] + disuz[j] * l[i]
            ))
        end
    end
    absRecipSpace = abs.(recipSpace) .* recSupport
    sqIntens = sqrt.(intens) .* recSupport
    return mapreduce((sqi,absr) -> (absr - sqi)^2, +, sqIntens, absRecipSpace)/length(intens)
end

function mesoModel(rho, ux, uy, uz, G, intens, recSupport)
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

function multiModel(x, y, z, mx, my, mz, rho, ux, uy, uz, h, k, l, hp, kp, lp, intens, recSupport)
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
