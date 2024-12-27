using BcdiCore
using Test
using CUDA
using LogExpFunctions
using ForwardDiff

include("Models.jl")
include("Losses.jl")
include("Regs.jl")
include("Atomic.jl")

@testset verbose=true "BcdiCore.jl" begin
    n = 20
    x = 1.8 .* pi .* rand(n) .+ 0.1 .* pi
    y = 1.8 .* pi .* rand(n) .+ 0.1 .* pi
    z = 1.8 .* pi .* rand(n) .+ 0.1 .* pi
    G = 20 .* rand(3)
    h = zeros(Int64,4,4,4)
    k = zeros(Int64,4,4,4)
    l = zeros(Int64,4,4,4)
    for i in 1:4
        for j in 1:4
            for m in 1:4
                h[i,j,m] = i-3
                k[i,j,m] = j-3
                l[i,j,m] = m-3
            end
        end
    end    
    intens = rand(0:30,4,4,4)
    recSupport = ones(Bool,4,4,4)
    recSupport[rand(1:4),rand(1:4),rand(1:4)] = false

    cuX = CuArray{Float64}(x)
    cuY = CuArray{Float64}(y)
    cuZ = CuArray{Float64}(z)
    cuH = CuArray{Int64}(h)
    cuK = CuArray{Int64}(k)
    cuL = CuArray{Int64}(l)
    cuIntens = CuArray{Int64}(intens)
    cuRecSupport = CuArray{Bool}(recSupport)
 
    losses = zeros(Float64, n)
    manyX = 1.8 .* pi .* rand(n) .+ 0.1 .* pi
    manyY = 1.8 .* pi .* rand(n) .+ 0.1 .* pi
    manyZ = 1.8 .* pi .* rand(n) .+ 0.1 .* pi
    adds = ones(Bool, n)

    cuLosses = CuArray{Float64}(losses)
    cuManyX = CuArray{Float64}(manyX)
    cuManyY = CuArray{Float64}(manyY)
    cuManyZ = CuArray{Float64}(manyZ)
    cuAdds = CuArray{Bool}(adds)
      
    realSpace = 100 .* rand(4,4,4) .+ 1im .* 100 .* rand(4,4,4)
    normSpace = realSpace ./ maximum(abs.(realSpace))
    flatSpace = 100 .* rand(n) .+ 1im .* 100 .* rand(n)

    cuRealSpace = CuArray{ComplexF64}(realSpace)
    cuNormSpace = CuArray{ComplexF64}(normSpace)
    cuFlatSpace = CuArray{ComplexF64}(flatSpace)

    rho = 100 .* rand(n)
    ux = 0.2 .* pi .* rand(n) .- 0.1 .* pi
    uy = 0.2 .* pi .* rand(n) .- 0.1 .* pi
    uz = 0.2 .* pi .* rand(n) .- 0.1 .* pi
    disux = 0.2 .* pi .* rand(n) .- 0.1 .* pi
    disuy = 0.2 .* pi .* rand(n) .- 0.1 .* pi
    disuz = 0.2 .* pi .* rand(n) .- 0.1 .* pi

    rhoB = 100 .* rand(4,4,4)
    normRhoB = rhoB ./ maximum(rhoB)
    uxB = 0.2 .* pi .* rand(4,4,4) .- 0.1 .* pi
    uyB = 0.2 .* pi .* rand(4,4,4) .- 0.1 .* pi
    uzB = 0.2 .* pi .* rand(4,4,4) .- 0.1 .* pi

    cuRho = CuArray{Float64}(rho)
    cuUx = CuArray{Float64}(ux)
    cuUy = CuArray{Float64}(uy)
    cuUz = CuArray{Float64}(uz)
    cuDisux = CuArray{Float64}(disux)
    cuDisuy = CuArray{Float64}(disuy)
    cuDisuz = CuArray{Float64}(disuz)

    cuRhoB = CuArray{Float64}(rhoB)
    cuNormRhoB = CuArray{Float64}(normRhoB)
    cuUxB = CuArray{Float64}(uxB)
    cuUyB = CuArray{Float64}(uyB)
    cuUzB = CuArray{Float64}(uzB)

    # Setup of Multi Tests
    mx = 1.8 .* pi .* rand(n) .+ 0.1 .* pi
    my = 1.8 .* pi .* rand(n) .+ 0.1 .* pi
    mz = 1.8 .* pi .* rand(n) .+ 0.1 .* pi

    cuMx = CuArray{Float64}(mx)
    cuMy = CuArray{Float64}(my)
    cuMz = CuArray{Float64}(mz)

    neighbors = zeros(Int64, 6,4^3)
    neighbors[1,:] .= mod.(1:(4^3), 4^3) .+ 1
    neighbors[2,:] .= mod.((1:(4^3)) .- 2, 4^3) .+ 1
    neighbors[3,:] .= mod.((1:(4^3)) .+ 1, 4^3) .+ 1
    neighbors[4,:] .= mod.((1:(4^3)) .- 3, 4^3) .+ 1
    neighbors[5,:] .= mod.((1:(4^3)) .+ 2, 4^3) .+ 1
    neighbors[6,:] .= mod.((1:(4^3)) .- 4, 4^3) .+ 1
    flatNeighbors = zeros(Int64, 6,n)
    flatNeighbors[1,:] .= mod.((1:n), n) .+ 1
    flatNeighbors[2,:] .= mod.((1:n) .- 2, n) .+ 1
    flatNeighbors[3,:] .= mod.((1:n) .+ 1, n) .+ 1
    flatNeighbors[4,:] .= mod.((1:n) .- 3, n) .+ 1
    flatNeighbors[5,:] .= mod.((1:n) .+ 2, n) .+ 1
    flatNeighbors[6,:] .= mod.((1:n) .- 4, n) .+ 1

    cuNeighbors = CuArray{Int64}(neighbors)

    lambda = 0.1
    tvmagreg = BcdiCore.TVMagReg(lambda, cuNeighbors)
    tvreg = BcdiCore.TVReg(lambda, cuNeighbors)
    a = 10*rand()
    b = 10*rand()
    c = 10*rand()-5
    betareg = BcdiCore.BetaReg(lambda, a, b, c)
    l2reg = BcdiCore.L2Reg(lambda, trues(size(realSpace)))
    delta = 0.1*sqrt(maximum(intens))

    # Test of Forward Models
    @testset verbose=true "Models" begin
        # Test of atomic model
        tester = atomicModel(x, y, z, h.+G[1], k.+G[2], l.+G[3], intens, recSupport)
        xDeriv = ForwardDiff.gradient(xp -> atomicModel(xp, y, z, h.+G[1], k.+G[2], l.+G[3], intens, recSupport), x)
        yDeriv = ForwardDiff.gradient(yp -> atomicModel(x, yp, z, h.+G[1], k.+G[2], l.+G[3], intens, recSupport), y)
        zDeriv = ForwardDiff.gradient(zp -> atomicModel(x, y, zp, h.+G[1], k.+G[2], l.+G[3], intens, recSupport), z)

        state = BcdiCore.AtomicState("L2", false, cuIntens, G, cuH, cuK, cuL, cuRecSupport)
        BcdiCore.setpts!(state, cuX, cuY, cuZ, true)
        testee = BcdiCore.loss(state, true, true, false)

        BcdiCore.slowForwardProp(state, x, y, z, adds, true)
        testee2 = BcdiCore.emptyLoss(state)

        @test @CUDA.allowscalar isapprox(testee[1], tester, rtol=1e-6)
        @test all(isapprox.(Array(state.xDeriv), xDeriv, rtol=1e-6))
        @test all(isapprox.(Array(state.yDeriv), yDeriv, rtol=1e-6))
        @test all(isapprox.(Array(state.zDeriv), zDeriv, rtol=1e-6))
        @test @CUDA.allowscalar isapprox(testee2[1], tester, rtol=1e-6)

        # Test of traditional model
        tester = tradModel(realSpace, intens, recSupport)
        rDeriv = ForwardDiff.gradient(rsp -> tradModel(rsp .+ 1im .* imag.(realSpace), intens, recSupport), real.(realSpace))
        iDeriv = ForwardDiff.gradient(isp -> tradModel(real.(realSpace) .+ 1im .* isp, intens, recSupport), imag.(realSpace))

        state = BcdiCore.TradState("L2", false, cuRealSpace, cuIntens, cuRecSupport)
        testee = BcdiCore.loss(state, true, true, false)

        @test @CUDA.allowscalar isapprox(testee[1], tester, rtol=1e-6)
        @test all(isapprox.(Array(real.(state.deriv)), rDeriv, rtol=1e-6))
        @test all(isapprox.(Array(imag.(state.deriv)), iDeriv, rtol=1e-6))

        tester = tradModel(flatSpace, intens, recSupport, x, y, z)
        rDeriv = ForwardDiff.gradient(rsp -> tradModel(rsp .+ 1im .* imag.(flatSpace), intens, recSupport, x, y, z), real.(flatSpace))
        iDeriv = ForwardDiff.gradient(isp -> tradModel(real.(flatSpace) .+ 1im .* isp, intens, recSupport, x, y, z), imag.(flatSpace))

        state = BcdiCore.TradState("L2", false, cuFlatSpace, cuIntens, cuRecSupport)
        BcdiCore.setpts!(state, cuX, cuY, cuZ, true)
        testee = BcdiCore.loss(state, true, true, false)

        @test @CUDA.allowscalar isapprox(testee[1], tester, rtol=1e-6)
        @test all(isapprox.(Array(real.(state.deriv)), rDeriv, rtol=1e-6))
        @test all(isapprox.(Array(imag.(state.deriv)), iDeriv, rtol=1e-6))

        # Test of meso model
        nonuniform = true
        highstrain = true
        disjoint = false
        tester = mesoModel(x, y, z, rho, ux, uy, uz, h, k, l, G, intens, recSupport)
        rhoDeriv = ForwardDiff.gradient(rhop -> mesoModel(x, y, z, rhop, ux, uy, uz, h, k, l, G, intens, recSupport), rho)
        uxDeriv = ForwardDiff.gradient(uxp -> mesoModel(x, y, z, rho, uxp, uy, uz, h, k, l, G, intens, recSupport), ux)
        uyDeriv = ForwardDiff.gradient(uyp -> mesoModel(x, y, z, rho, ux, uyp, uz, h, k, l, G, intens, recSupport), uy)
        uzDeriv = ForwardDiff.gradient(uzp -> mesoModel(x, y, z, rho, ux, uy, uzp, h, k, l, G, intens, recSupport), uz)

        state = BcdiCore.MesoState("L2", false, cuIntens, G, cuRecSupport, nonuniform, highstrain, disjoint)
        BcdiCore.setpts!(state, cuX, cuY, cuZ, cuRho, cuUx, cuUy, cuUz, true)
        testee = BcdiCore.loss(state, true, true, false)

        @test @CUDA.allowscalar isapprox(testee[1], tester, rtol=1e-6)
        @test all(isapprox.(Array(state.rhoDeriv), rhoDeriv, rtol=1e-6))
        @test all(isapprox.(Array(state.uxDeriv), uxDeriv, rtol=1e-6))
        @test all(isapprox.(Array(state.uyDeriv), uyDeriv, rtol=1e-6))
        @test all(isapprox.(Array(state.uzDeriv), uzDeriv, rtol=1e-6))

        nonuniform = true
        highstrain = true
        disjoint = true
        tester = mesoModel(x, y, z, rho, ux, uy, uz, disux, disuy, disuz, h, k, l, G, intens, recSupport)
        rhoDeriv = ForwardDiff.gradient(rhop -> mesoModel(x, y, z, rhop, ux, uy, uz, disux, disuy, disuz, h, k, l, G, intens, recSupport), rho)
        uxDeriv = ForwardDiff.gradient(uxp -> mesoModel(x, y, z, rho, uxp, uy, uz, disux, disuy, disuz, h, k, l, G, intens, recSupport), ux)
        uyDeriv = ForwardDiff.gradient(uyp -> mesoModel(x, y, z, rho, ux, uyp, uz, disux, disuy, disuz, h, k, l, G, intens, recSupport), uy)
        uzDeriv = ForwardDiff.gradient(uzp -> mesoModel(x, y, z, rho, ux, uy, uzp, disux, disuy, disuz, h, k, l, G, intens, recSupport), uz)
        disuxDeriv = ForwardDiff.gradient(disuxp -> mesoModel(x, y, z, rho, ux, uy, uz, disuxp, disuy, disuz, h, k, l, G, intens, recSupport), disux)
        disuyDeriv = ForwardDiff.gradient(disuyp -> mesoModel(x, y, z, rho, ux, uy, uz, disux, disuyp, disuz, h, k, l, G, intens, recSupport), disuy)
        disuzDeriv = ForwardDiff.gradient(disuzp -> mesoModel(x, y, z, rho, ux, uy, uz, disux, disuy, disuzp, h, k, l, G, intens, recSupport), disuz)

        state = BcdiCore.MesoState("L2", false, cuIntens, G, cuRecSupport, nonuniform, highstrain, disjoint)
        BcdiCore.setpts!(state, cuX, cuY, cuZ, cuRho, cuUx, cuUy, cuUz, cuDisux, cuDisuy, cuDisuz, true)
        testee = BcdiCore.loss(state, true, true, false)

        @test @CUDA.allowscalar isapprox(testee[1], tester, rtol=1e-6)
        @test all(isapprox.(Array(state.rhoDeriv), rhoDeriv, rtol=1e-6))
        @test all(isapprox.(Array(state.uxDeriv), uxDeriv, rtol=1e-6))
        @test all(isapprox.(Array(state.uyDeriv), uyDeriv, rtol=1e-6))
        @test all(isapprox.(Array(state.uzDeriv), uzDeriv, rtol=1e-6))
        @test all(isapprox.(Array(state.disuxDeriv), disuxDeriv, rtol=1e-6))
        @test all(isapprox.(Array(state.disuyDeriv), disuyDeriv, rtol=1e-6))
        @test all(isapprox.(Array(state.disuzDeriv), disuzDeriv, rtol=1e-6))

        nonuniform = false
        highstrain = false
        disjoint = false
        tester = mesoModel(rhoB, uxB, uyB, uzB, G, intens, recSupport)
        rhoDeriv = ForwardDiff.gradient(rhop -> mesoModel(rhop, uxB, uyB, uzB, G, intens, recSupport), rhoB)
        uxDeriv = ForwardDiff.gradient(uxp -> mesoModel(rhoB, uxp, uyB, uzB, G, intens, recSupport), uxB)
        uyDeriv = ForwardDiff.gradient(uyp -> mesoModel(rhoB, uxB, uyp, uzB, G, intens, recSupport), uyB)
        uzDeriv = ForwardDiff.gradient(uzp -> mesoModel(rhoB, uxB, uyB, uzp, G, intens, recSupport), uzB)

        state = BcdiCore.MesoState("L2", false, cuIntens, G, cuRecSupport, nonuniform, highstrain, disjoint)
        BcdiCore.setpts!(state, cuRhoB, cuUxB, cuUyB, cuUzB, true)
        testee = BcdiCore.loss(state, true, true, false)

        @test @CUDA.allowscalar isapprox(testee[1], tester, rtol=1e-6)
        @test all(isapprox.(Array(state.rhoDeriv), rhoDeriv, rtol=1e-6))
        @test all(isapprox.(Array(state.uxDeriv), uxDeriv, rtol=1e-6))
        @test all(isapprox.(Array(state.uyDeriv), uyDeriv, rtol=1e-6))
        @test all(isapprox.(Array(state.uzDeriv), uzDeriv, rtol=1e-6))

        # Test of multi model
        tester = multiModel(x, y, z, mx, my, mz, rho, ux, uy, uz, h, k, l, h.+G[1], k.+G[2], l.+G[3], intens, recSupport)
        xDeriv = ForwardDiff.gradient(xp -> multiModel(xp, y, z, mx, my, mz, rho, ux, uy, uz, h, k, l, h.+G[1], k.+G[2], l.+G[3], intens, recSupport), x)
        yDeriv = ForwardDiff.gradient(yp -> multiModel(x, yp, z, mx, my, mz, rho, ux, uy, uz, h, k, l, h.+G[1], k.+G[2], l.+G[3], intens, recSupport), y)
        zDeriv = ForwardDiff.gradient(zp -> multiModel(x, y, zp, mx, my, mz, rho, ux, uy, uz, h, k, l, h.+G[1], k.+G[2], l.+G[3], intens, recSupport), z)
        rhoDeriv = ForwardDiff.gradient(rhop -> multiModel(x, y, z, mx, my, mz, rhop, ux, uy, uz, h, k, l, h.+G[1], k.+G[2], l.+G[3], intens, recSupport), rho)
        uxDeriv = ForwardDiff.gradient(uxp -> multiModel(x, y, z, mx, my, mz, rho, uxp, uy, uz, h, k, l, h.+G[1], k.+G[2], l.+G[3], intens, recSupport), ux)
        uyDeriv = ForwardDiff.gradient(uyp -> multiModel(x, y, z, mx, my, mz, rho, ux, uyp, uz, h, k, l, h.+G[1], k.+G[2], l.+G[3], intens, recSupport), uy)
        uzDeriv = ForwardDiff.gradient(uzp -> multiModel(x, y, z, mx, my, mz, rho, ux, uy, uzp, h, k, l, h.+G[1], k.+G[2], l.+G[3], intens, recSupport), uz)

        state = BcdiCore.MultiState("L2", false, cuIntens, G, cuH, cuK, cuL, cuRecSupport)
        BcdiCore.setpts!(state, cuX, cuY, cuZ, cuMx, cuMy, cuMz, cuRho, cuUx, cuUy, cuUz, true)
        testee = BcdiCore.loss(state, true, true, false)

        @test @CUDA.allowscalar isapprox(testee[1], tester, rtol=1e-6)
        @test all(isapprox.(Array(state.xDeriv), xDeriv, rtol=1e-6))
        @test all(isapprox.(Array(state.yDeriv), yDeriv, rtol=1e-6))
        @test all(isapprox.(Array(state.zDeriv), zDeriv, rtol=1e-6))
        @test all(isapprox.(Array(state.rhoDeriv), rhoDeriv, rtol=1e-6))
        @test all(isapprox.(Array(state.uxDeriv), uxDeriv, rtol=1e-6))
        @test all(isapprox.(Array(state.uyDeriv), uyDeriv, rtol=1e-6))
        @test all(isapprox.(Array(state.uzDeriv), uzDeriv, rtol=1e-6))
    end

    # Test of Loss Functions
    @testset verbose=true "Losses" begin
        # Test of likelihood with scaling
        tester = likelihoodWithScaling(realSpace, intens, recSupport)
        rDeriv = ForwardDiff.gradient(rsp -> likelihoodWithScaling(rsp .+ 1im .* imag.(realSpace), intens, recSupport), real.(realSpace))
        iDeriv = ForwardDiff.gradient(isp -> likelihoodWithScaling(real.(realSpace) .+ 1im .* isp, intens, recSupport), imag.(realSpace))

        state = BcdiCore.TradState("likelihood", true, cuRealSpace, cuIntens, cuRecSupport)
        testee = BcdiCore.loss(state, true, true, false)

        @test @CUDA.allowscalar isapprox(testee[1], tester, rtol=1e-6)
        @test all(isapprox.(Array(real.(state.deriv)), rDeriv, rtol=1e-6))
        @test all(isapprox.(Array(imag.(state.deriv)), iDeriv, rtol=1e-6))

        # Test of likelihood without scaling
        tester = likelihoodWithoutScaling(realSpace, intens, recSupport)
        rDeriv = ForwardDiff.gradient(rsp -> likelihoodWithoutScaling(rsp .+ 1im .* imag.(realSpace), intens, recSupport), real.(realSpace))
        iDeriv = ForwardDiff.gradient(isp -> likelihoodWithoutScaling(real.(realSpace) .+ 1im .* isp, intens, recSupport), imag.(realSpace))

        state = BcdiCore.TradState("likelihood", false, cuRealSpace, cuIntens, cuRecSupport)
        testee = BcdiCore.loss(state, true, true, false)

        @test @CUDA.allowscalar isapprox(testee[1], tester, rtol=1e-6)
        @test all(isapprox.(Array(real.(state.deriv)), rDeriv, rtol=1e-6))
        @test all(isapprox.(Array(imag.(state.deriv)), iDeriv, rtol=1e-6))

        # Test of L2 norm with scaling
        tester = l2WithScaling(realSpace, intens, recSupport)
        rDeriv = ForwardDiff.gradient(rsp -> l2WithScaling(rsp .+ 1im .* imag.(realSpace), intens, recSupport), real.(realSpace))
        iDeriv = ForwardDiff.gradient(isp -> l2WithScaling(real.(realSpace) .+ 1im .* isp, intens, recSupport), imag.(realSpace))

        state = BcdiCore.TradState("L2", true, cuRealSpace, cuIntens, cuRecSupport)
        testee = BcdiCore.loss(state, true, true, false)

        @test @CUDA.allowscalar isapprox(testee[1], tester, rtol=1e-6)
        @test all(isapprox.(Array(real.(state.deriv)), rDeriv, rtol=1e-6))
        @test all(isapprox.(Array(imag.(state.deriv)), iDeriv, rtol=1e-6))

        # Test of L2 norm without scaling
        tester = l2WithoutScaling(realSpace, intens, recSupport)
        rDeriv = ForwardDiff.gradient(rsp -> l2WithoutScaling(rsp .+ 1im .* imag.(realSpace), intens, recSupport), real.(realSpace))
        iDeriv = ForwardDiff.gradient(isp -> l2WithoutScaling(real.(realSpace) .+ 1im .* isp, intens, recSupport), imag.(realSpace))

        state = BcdiCore.TradState("L2", false, cuRealSpace, cuIntens, cuRecSupport)
        testee = BcdiCore.loss(state, true, true, false)

        @test @CUDA.allowscalar isapprox(testee[1], tester, rtol=1e-6)
        @test all(isapprox.(Array(real.(state.deriv)), rDeriv, rtol=1e-6))
        @test all(isapprox.(Array(imag.(state.deriv)), iDeriv, rtol=1e-6))

        # Test of Huber norm with scaling
        tester = huberWithScaling(realSpace, intens, recSupport, delta)
        rDeriv = ForwardDiff.gradient(rsp -> huberWithScaling(rsp .+ 1im .* imag.(realSpace), intens, recSupport, delta), real.(realSpace))
        iDeriv = ForwardDiff.gradient(isp -> huberWithScaling(real.(realSpace) .+ 1im .* isp, intens, recSupport, delta), imag.(realSpace))

        state = BcdiCore.TradState("Huber", true, cuRealSpace, cuIntens, cuRecSupport)
        testee = BcdiCore.loss(state, true, true, false)

        @test @CUDA.allowscalar isapprox(testee[1], tester, rtol=1e-6)
        @test all(isapprox.(Array(real.(state.deriv)), rDeriv, rtol=1e-6))
        @test all(isapprox.(Array(imag.(state.deriv)), iDeriv, rtol=1e-6))
  
        # Test of Huber norm without scaling
        tester = huberWithoutScaling(realSpace, intens, recSupport, delta)
        rDeriv = ForwardDiff.gradient(rsp -> huberWithoutScaling(rsp .+ 1im .* imag.(realSpace), intens, recSupport, delta), real.(realSpace))
        iDeriv = ForwardDiff.gradient(isp -> huberWithoutScaling(real.(realSpace) .+ 1im .* isp, intens, recSupport, delta), imag.(realSpace))

        state = BcdiCore.TradState("Huber", false, cuRealSpace, cuIntens, cuRecSupport)
        testee = BcdiCore.loss(state, true, true, false)

        @test @CUDA.allowscalar isapprox(testee[1], tester, rtol=1e-6)
        @test all(isapprox.(Array(real.(state.deriv)), rDeriv, rtol=1e-6))
        @test all(isapprox.(Array(imag.(state.deriv)), iDeriv, rtol=1e-6))
    end

    # Test of Regularizers
    @testset verbose=true "Regularizers" begin
        nonuniform = false
        highstrain = false
        disjoint = false

        # Test of total variation (magnitude) regularization
        tester = TVRMag(realSpace, lambda, neighbors)
        rDeriv = ForwardDiff.gradient(rsp -> TVRMag(rsp .+ 1im .* imag.(realSpace), lambda, neighbors), real.(realSpace))
        iDeriv = ForwardDiff.gradient(isp -> TVRMag(real.(realSpace) .+ 1im .* isp, lambda, neighbors), imag.(realSpace))

        state = BcdiCore.TradState("likelihood", true, cuRealSpace, cuIntens, cuRecSupport)
        testee = BcdiCore.modifyLoss(state, tvmagreg)
        BcdiCore.modifyDeriv(state, tvmagreg)

        @test @CUDA.allowscalar isapprox(testee[1], tester, atol=1e-6)
        @test all(isapprox.(Array(real.(state.deriv)), rDeriv, atol=1e-6))
        @test all(isapprox.(Array(imag.(state.deriv)), iDeriv, atol=1e-6))

        tester = TVRMag(rhoB, lambda, neighbors)
        rhoDeriv = ForwardDiff.gradient(rhop -> TVRMag(rhop, lambda, neighbors), rhoB)

        state = BcdiCore.MesoState("likelihood", true, cuIntens, G, cuRecSupport, nonuniform, highstrain, disjoint)
        BcdiCore.setpts!(state, cuRhoB, cuUxB, cuUyB, cuUzB, true)
        testee = BcdiCore.modifyLoss(state, tvmagreg)
        BcdiCore.modifyDeriv(state, tvmagreg)

        @test @CUDA.allowscalar isapprox(testee[1], tester, atol=1e-6)
        @test all(isapprox.(Array(state.rhoDeriv), rhoDeriv, atol=1e-6))

        # Test of total variation regularization
        tester = TVR(realSpace, lambda, neighbors)
        rDeriv = ForwardDiff.gradient(rsp -> TVR(rsp .+ 1im .* imag.(realSpace), lambda, neighbors), real.(realSpace))
        iDeriv = ForwardDiff.gradient(isp -> TVR(real.(realSpace) .+ 1im .* isp, lambda, neighbors), imag.(realSpace))

        state = BcdiCore.TradState("likelihood", true, cuRealSpace, cuIntens, cuRecSupport)
        testee = BcdiCore.modifyLoss(state, tvreg)
        BcdiCore.modifyDeriv(state, tvreg)

        @test @CUDA.allowscalar isapprox(testee[1], tester, atol=1e-6)
        @test all(isapprox.(Array(real.(state.deriv)), rDeriv, atol=1e-6))
        @test all(isapprox.(Array(imag.(state.deriv)), iDeriv, atol=1e-6))

        tester = TVR(rhoB, uxB, uyB, uzB, G, lambda, neighbors)
        rhoDeriv = ForwardDiff.gradient(rhop -> TVR(rhop, uxB, uyB, uzB, G, lambda, neighbors), rhoB)
        uxDeriv = ForwardDiff.gradient(uxp -> TVR(rhoB, uxp, uyB, uzB, G, lambda, neighbors), uxB)
        uyDeriv = ForwardDiff.gradient(uyp -> TVR(rhoB, uxB, uyp, uzB, G, lambda, neighbors), uyB)
        uzDeriv = ForwardDiff.gradient(uzp -> TVR(rhoB, uxB, uyB, uzp, G, lambda, neighbors), uzB)

        state = BcdiCore.MesoState("likelihood", true, cuIntens, G, cuRecSupport, nonuniform, highstrain, disjoint)
        BcdiCore.setpts!(state, cuRhoB, cuUxB, cuUyB, cuUzB, true)
        testee = BcdiCore.modifyLoss(state, tvreg)
        BcdiCore.modifyDeriv(state, tvreg)

        @test @CUDA.allowscalar isapprox(testee[1], tester, atol=1e-6)
        @test all(isapprox.(Array(state.rhoDeriv), rhoDeriv, atol=1e-6))
        @test all(isapprox.(Array(state.uxDeriv), uxDeriv, atol=1e-6))
        @test all(isapprox.(Array(state.uyDeriv), uyDeriv, atol=1e-6))
        @test all(isapprox.(Array(state.uzDeriv), uzDeriv, atol=1e-6))

        # Test of beta regularization
        tester = BetaR(normSpace, lambda, a, b, c)
        rDeriv = ForwardDiff.gradient(rsp -> BetaR(rsp .+ 1im .* imag.(normSpace), lambda, a, b, c), real.(normSpace))
        iDeriv = ForwardDiff.gradient(isp -> BetaR(real.(normSpace) .+ 1im .* isp, lambda, a, b, c), imag.(normSpace))

        state = BcdiCore.TradState("likelihood", true, cuNormSpace, cuIntens, cuRecSupport)
        testee = BcdiCore.modifyLoss(state, betareg)
        BcdiCore.modifyDeriv(state, betareg)

        @test @CUDA.allowscalar isapprox(testee[1], tester, atol=1e-6)
        @test all(isapprox.(Array(real.(state.deriv)), rDeriv, atol=1e-6))
        @test all(isapprox.(Array(imag.(state.deriv)), iDeriv, atol=1e-6))

        tester = BetaR(normRhoB, lambda, a, b, c)
        rhoDeriv = ForwardDiff.gradient(rhop -> BetaR(rhop, lambda, a, b, c), normRhoB)

        state = BcdiCore.MesoState("likelihood", true, cuIntens, G, cuRecSupport, nonuniform, highstrain, disjoint)
        BcdiCore.setpts!(state, cuNormRhoB, cuUxB, cuUyB, cuUzB, true)
        testee = BcdiCore.modifyLoss(state, betareg)
        BcdiCore.modifyDeriv(state, betareg)

        @test @CUDA.allowscalar isapprox(testee[1], tester, atol=1e-6)
        @test all(isapprox.(Array(state.rhoDeriv), rhoDeriv, atol=1e-6))

        # Test of L2 regularization
        tester = L2R(realSpace, lambda)
        rDeriv = ForwardDiff.gradient(rsp -> L2R(rsp .+ 1im .* imag.(realSpace), lambda), real.(realSpace))
        iDeriv = ForwardDiff.gradient(isp -> L2R(real.(realSpace) .+ 1im .* isp, lambda), imag.(realSpace))

        state = BcdiCore.TradState("likelihood", true, cuRealSpace, cuIntens, cuRecSupport)
        testee = BcdiCore.modifyLoss(state, l2reg)
        BcdiCore.modifyDeriv(state, l2reg)

        @test @CUDA.allowscalar isapprox(testee[1], tester, atol=1e-6)
        @test all(isapprox.(Array(real.(state.deriv)), rDeriv, atol=1e-6))
        @test all(isapprox.(Array(imag.(state.deriv)), iDeriv, atol=1e-6))

        tester = L2R(rhoB, lambda)
        rhoDeriv = ForwardDiff.gradient(rhop -> L2R(rhop, lambda), rhoB)

        state = BcdiCore.MesoState("likelihood", true, cuIntens, G, cuRecSupport, nonuniform, highstrain, disjoint)
        BcdiCore.setpts!(state, cuRhoB, cuUxB, cuUyB, cuUzB, true)
        testee = BcdiCore.modifyLoss(state, l2reg)
        BcdiCore.modifyDeriv(state, l2reg)

        @test @CUDA.allowscalar isapprox(testee[1], tester, atol=1e-6)
        @test all(isapprox.(Array(state.rhoDeriv), rhoDeriv, atol=1e-6))
    end

    # Aditional Atomic tests
    @testset verbose=true "Atomic" begin
        # Test of likelihood with scaling
        state = BcdiCore.AtomicState("likelihood", true, cuIntens, G, cuH, cuK, cuL, cuRecSupport)
        BcdiCore.setpts!(state, cuX, cuY, cuZ, true)
        BcdiCore.loss(state, false, false, true)

        for i in 1:length(manyX)
            losses[i] = atomicLikelihoodWithScaling(vcat(x,[manyX[i]]), vcat(y,[manyY[i]]), vcat(z,[manyZ[i]]), h.+G[1], k.+G[2], l.+G[3], intens, recSupport)
        end
        BcdiCore.lossManyAtomic!(cuLosses, state, cuManyX, cuManyY, cuManyZ, cuAdds, false)

        @test all(isapprox.(Array(cuLosses), losses, rtol=1e-6))

        # Test of likelihood without scaling
        state = BcdiCore.AtomicState("likelihood", false, cuIntens, G, cuH, cuK, cuL, cuRecSupport)
        BcdiCore.setpts!(state, cuX, cuY, cuZ, true)
        BcdiCore.loss(state, false, false, true)

        for i in 1:length(manyX)
            losses[i] = atomicLikelihoodWithoutScaling(vcat(x,[manyX[i]]), vcat(y,[manyY[i]]), vcat(z,[manyZ[i]]), h.+G[1], k.+G[2], l.+G[3], intens, recSupport)
        end
        BcdiCore.lossManyAtomic!(cuLosses, state, cuManyX, cuManyY, cuManyZ, cuAdds, false)

        @test all(isapprox.(Array(cuLosses), losses, rtol=1e-6))

        # Test of L2 with scaling
        state = BcdiCore.AtomicState("L2", true, cuIntens, G, cuH, cuK, cuL, cuRecSupport)
        BcdiCore.setpts!(state, cuX, cuY, cuZ, true)
        BcdiCore.loss(state, false, false, true)

        for i in 1:length(manyX)
            losses[i] = atomicL2WithScaling(vcat(x,[manyX[i]]), vcat(y,[manyY[i]]), vcat(z,[manyZ[i]]), h.+G[1], k.+G[2], l.+G[3], intens, recSupport)
        end
        BcdiCore.lossManyAtomic!(cuLosses, state, cuManyX, cuManyY, cuManyZ, cuAdds, false)

        @test all(isapprox.(Array(cuLosses), losses, rtol=1e-6))

        # Test of L2 without scaling
        state = BcdiCore.AtomicState("L2", false, cuIntens, G, cuH, cuK, cuL, cuRecSupport)
        BcdiCore.setpts!(state, cuX, cuY, cuZ, true)
        BcdiCore.loss(state, false, false, true)

        for i in 1:length(manyX)
            losses[i] = atomicL2WithoutScaling(vcat(x,[manyX[i]]), vcat(y,[manyY[i]]), vcat(z,[manyZ[i]]), h.+G[1], k.+G[2], l.+G[3], intens, recSupport)
        end
        BcdiCore.lossManyAtomic!(cuLosses, state, cuManyX, cuManyY, cuManyZ, cuAdds, false)

        @test all(isapprox.(Array(cuLosses), losses, rtol=1e-6))
    end
end
