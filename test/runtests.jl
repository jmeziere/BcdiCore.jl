using BcdiCore
using Test
using CUDA
using LogExpFunctions
using ForwardDiff

include("Atomic.jl")
include("Traditional.jl")
include("Meso.jl")
include("Multi.jl")

@testset verbose=true "BcdiCore.jl" begin
    # Setup for Atomic Tests
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
        
    # Atomic Test of Likelihood With Scaling
    @testset verbose=true "Atomic-Likelihood-Scaling" begin
        tester = atomicLikelihoodWithScaling(x, y, z, h.+G[1], k.+G[2], l.+G[3], intens, recSupport)
        xDeriv = ForwardDiff.gradient(xp -> atomicLikelihoodWithScaling(xp, y, z, h.+G[1], k.+G[2], l.+G[3], intens, recSupport), x)
        yDeriv = ForwardDiff.gradient(yp -> atomicLikelihoodWithScaling(x, yp, z, h.+G[1], k.+G[2], l.+G[3], intens, recSupport), y)
        zDeriv = ForwardDiff.gradient(zp -> atomicLikelihoodWithScaling(x, y, zp, h.+G[1], k.+G[2], l.+G[3], intens, recSupport), z)

        state = BcdiCore.AtomicState("likelihood", true, cuIntens, G, cuH, cuK, cuL, cuRecSupport)
        BcdiCore.setpts!(state, cuX, cuY, cuZ, true)
        testee = BcdiCore.loss(state, true, true, false)

        BcdiCore.slowForwardProp(state, x, y, z, adds, true)
        testee2 = BcdiCore.emptyLoss(state)

        for i in 1:length(manyX)
            losses[i] = atomicLikelihoodWithScaling(vcat(x,[manyX[i]]), vcat(y,[manyY[i]]), vcat(z,[manyZ[i]]), h.+G[1], k.+G[2], l.+G[3], intens, recSupport)
        end
        BcdiCore.lossManyAtomic!(cuLosses, state, cuManyX, cuManyY, cuManyZ, cuAdds, false)

        @test isapprox(testee, tester, rtol=1e-6)
        @test all(isapprox.(Array(state.xDeriv), xDeriv, rtol=1e-6))
        @test all(isapprox.(Array(state.yDeriv), yDeriv, rtol=1e-6))
        @test all(isapprox.(Array(state.zDeriv), zDeriv, rtol=1e-6))
        @test isapprox(testee2, tester, rtol=1e-6)
        @test all(isapprox.(Array(cuLosses), losses, rtol=1e-6))
    end

    # Atomic Test of Likelihood Without Scaling
    @testset verbose=true "Atomic-Likelihood-NoScaling" begin
        tester = atomicLikelihoodWithoutScaling(x, y, z, h.+G[1], k.+G[2], l.+G[3], intens, recSupport)
        xDeriv = ForwardDiff.gradient(xp -> atomicLikelihoodWithoutScaling(xp, y, z, h.+G[1], k.+G[2], l.+G[3], intens, recSupport), x)
        yDeriv = ForwardDiff.gradient(yp -> atomicLikelihoodWithoutScaling(x, yp, z, h.+G[1], k.+G[2], l.+G[3], intens, recSupport), y)
        zDeriv = ForwardDiff.gradient(zp -> atomicLikelihoodWithoutScaling(x, y, zp, h.+G[1], k.+G[2], l.+G[3], intens, recSupport), z)

        state = BcdiCore.AtomicState("likelihood", false, cuIntens, G, cuH, cuK, cuL, cuRecSupport)
        BcdiCore.setpts!(state, cuX, cuY, cuZ, true)
        testee = BcdiCore.loss(state, true, true, false)

        BcdiCore.slowForwardProp(state, x, y, z, adds, true)
        testee2 = BcdiCore.emptyLoss(state)

        for i in 1:length(manyX)
            losses[i] = atomicLikelihoodWithoutScaling(vcat(x,[manyX[i]]), vcat(y,[manyY[i]]), vcat(z,[manyZ[i]]), h.+G[1], k.+G[2], l.+G[3], intens, recSupport)
        end
        BcdiCore.lossManyAtomic!(cuLosses, state, cuManyX, cuManyY, cuManyZ, cuAdds, false)

        @test isapprox(testee, tester, rtol=1e-6)
        @test all(isapprox.(Array(state.xDeriv), xDeriv, rtol=1e-6))
        @test all(isapprox.(Array(state.yDeriv), yDeriv, rtol=1e-6))
        @test all(isapprox.(Array(state.zDeriv), zDeriv, rtol=1e-6))
        @test isapprox(testee2, tester, rtol=1e-6)
        @test all(isapprox.(Array(cuLosses), losses, rtol=1e-6))
    end

    # Atomic Test of L2 Norm With Scaling
    @testset verbose=true "Atomic-L2-Scaling" begin
        tester = atomicL2WithScaling(x, y, z, h.+G[1], k.+G[2], l.+G[3], intens, recSupport)
        xDeriv = ForwardDiff.gradient(xp -> atomicL2WithScaling(xp, y, z, h.+G[1], k.+G[2], l.+G[3], intens, recSupport), x)
        yDeriv = ForwardDiff.gradient(yp -> atomicL2WithScaling(x, yp, z, h.+G[1], k.+G[2], l.+G[3], intens, recSupport), y)
        zDeriv = ForwardDiff.gradient(zp -> atomicL2WithScaling(x, y, zp, h.+G[1], k.+G[2], l.+G[3], intens, recSupport), z)

        state = BcdiCore.AtomicState("L2", true, cuIntens, G, cuH, cuK, cuL, cuRecSupport)
        BcdiCore.setpts!(state, cuX, cuY, cuZ, true)
        testee = BcdiCore.loss(state, true, true, false)

        BcdiCore.slowForwardProp(state, x, y, z, adds, true)
        testee2 = BcdiCore.emptyLoss(state)

        for i in 1:length(manyX)
            losses[i] = atomicL2WithScaling(vcat(x,[manyX[i]]), vcat(y,[manyY[i]]), vcat(z,[manyZ[i]]), h.+G[1], k.+G[2], l.+G[3], intens, recSupport)
        end
        BcdiCore.lossManyAtomic!(cuLosses, state, cuManyX, cuManyY, cuManyZ, cuAdds, false)

        @test isapprox(testee, tester, rtol=1e-6)
        @test all(isapprox.(Array(state.xDeriv), xDeriv, rtol=1e-6))
        @test all(isapprox.(Array(state.yDeriv), yDeriv, rtol=1e-6))
        @test all(isapprox.(Array(state.zDeriv), zDeriv, rtol=1e-6))
        @test isapprox(testee2, tester, rtol=1e-6)
        @test all(isapprox.(Array(cuLosses), losses, rtol=1e-6))
    end

    # Atomic Test of L2 Norm Without Scaling
    @testset verbose=true "Atomic-L2-NoScaling" begin
        tester = atomicL2WithoutScaling(x, y, z, h.+G[1], k.+G[2], l.+G[3], intens, recSupport)
        xDeriv = ForwardDiff.gradient(xp -> atomicL2WithoutScaling(xp, y, z, h.+G[1], k.+G[2], l.+G[3], intens, recSupport), x)
        yDeriv = ForwardDiff.gradient(yp -> atomicL2WithoutScaling(x, yp, z, h.+G[1], k.+G[2], l.+G[3], intens, recSupport), y)
        zDeriv = ForwardDiff.gradient(zp -> atomicL2WithoutScaling(x, y, zp, h.+G[1], k.+G[2], l.+G[3], intens, recSupport), z)

        state = BcdiCore.AtomicState("L2", false, cuIntens, G, cuH, cuK, cuL, cuRecSupport)
        BcdiCore.setpts!(state, cuX, cuY, cuZ, true)
        testee = BcdiCore.loss(state, true, true, false)

        BcdiCore.slowForwardProp(state, x, y, z, adds, true)
        testee2 = BcdiCore.emptyLoss(state)

        for i in 1:length(manyX)
            losses[i] = atomicL2WithoutScaling(vcat(x,[manyX[i]]), vcat(y,[manyY[i]]), vcat(z,[manyZ[i]]), h.+G[1], k.+G[2], l.+G[3], intens, recSupport)
        end
        BcdiCore.lossManyAtomic!(cuLosses, state, cuManyX, cuManyY, cuManyZ, cuAdds, false)

        @test isapprox(testee, tester, rtol=1e-6)
        @test all(isapprox.(Array(state.xDeriv), xDeriv, rtol=1e-6))
        @test all(isapprox.(Array(state.yDeriv), yDeriv, rtol=1e-6))
        @test all(isapprox.(Array(state.zDeriv), zDeriv, rtol=1e-6))
        @test isapprox(testee2, tester, rtol=1e-6)
        @test all(isapprox.(Array(cuLosses), losses, rtol=1e-6))
    end

    # Setup for Traditional Tests
    realSpace = 100 .* rand(4,4,4) .+ 1im .* 100 .* rand(4,4,4)

    cuRealSpace = CuArray{ComplexF64}(realSpace)

    # Traditional Test of Likelihood With Scaling
    @testset verbose=true "Traditional-Likelihood-Scaling" begin
        tester = tradLikelihoodWithScaling(realSpace, intens, recSupport)
        rDeriv = ForwardDiff.gradient(rsp -> tradLikelihoodWithScaling(rsp .+ 1im .* imag.(realSpace), intens, recSupport), real.(realSpace))
        iDeriv = ForwardDiff.gradient(isp -> tradLikelihoodWithScaling(real.(realSpace) .+ 1im .* isp, intens, recSupport), imag.(realSpace))

        state = BcdiCore.TradState("likelihood", true, cuRealSpace, cuIntens, cuRecSupport)
        testee = BcdiCore.loss(state, true, true, false)

        @test isapprox(testee, tester, rtol=1e-6)
        @test all(isapprox.(Array(real.(state.deriv)), rDeriv, rtol=1e-6))
        @test all(isapprox.(Array(imag.(state.deriv)), iDeriv, rtol=1e-6))
    end

    # Traditional Test of Likelihood Without Scaling
    @testset verbose=true "Traditional-Likelihood-NoScaling" begin
        tester = tradLikelihoodWithoutScaling(realSpace, intens, recSupport)
        rDeriv = ForwardDiff.gradient(rsp -> tradLikelihoodWithoutScaling(rsp .+ 1im .* imag.(realSpace), intens, recSupport), real.(realSpace))
        iDeriv = ForwardDiff.gradient(isp -> tradLikelihoodWithoutScaling(real.(realSpace) .+ 1im .* isp, intens, recSupport), imag.(realSpace))

        state = BcdiCore.TradState("likelihood", false, cuRealSpace, cuIntens, cuRecSupport)
        testee = BcdiCore.loss(state, true, true, false)

        @test isapprox(testee, tester, rtol=1e-6)
        @test all(isapprox.(Array(real.(state.deriv)), rDeriv, rtol=1e-6))
        @test all(isapprox.(Array(imag.(state.deriv)), iDeriv, rtol=1e-6))
    end

    # Traditional Test of L2 Norm With Scaling
    @testset verbose=true "Traditional-L2-Scaling" begin
        tester = tradL2WithScaling(realSpace, intens, recSupport)
        rDeriv = ForwardDiff.gradient(rsp -> tradL2WithScaling(rsp .+ 1im .* imag.(realSpace), intens, recSupport), real.(realSpace))
        iDeriv = ForwardDiff.gradient(isp -> tradL2WithScaling(real.(realSpace) .+ 1im .* isp, intens, recSupport), imag.(realSpace))

        state = BcdiCore.TradState("L2", true, cuRealSpace, cuIntens, cuRecSupport)
        testee = BcdiCore.loss(state, true, true, false)

        @test isapprox(testee, tester, rtol=1e-6)
        @test all(isapprox.(Array(real.(state.deriv)), rDeriv, rtol=1e-6))
        @test all(isapprox.(Array(imag.(state.deriv)), iDeriv, rtol=1e-6))
    end

    # Traditional Test of L2 Norm Without Scaling
    @testset verbose=true "Traditional-L2-NoScaling" begin
        tester = tradL2WithoutScaling(realSpace, intens, recSupport)
        rDeriv = ForwardDiff.gradient(rsp -> tradL2WithoutScaling(rsp .+ 1im .* imag.(realSpace), intens, recSupport), real.(realSpace))
        iDeriv = ForwardDiff.gradient(isp -> tradL2WithoutScaling(real.(realSpace) .+ 1im .* isp, intens, recSupport), imag.(realSpace))

        state = BcdiCore.TradState("L2", false, cuRealSpace, cuIntens, cuRecSupport)
        testee = BcdiCore.loss(state, true, true, false)

        @test isapprox(testee, tester, rtol=1e-6)
        @test all(isapprox.(Array(real.(state.deriv)), rDeriv, rtol=1e-6))
        @test all(isapprox.(Array(imag.(state.deriv)), iDeriv, rtol=1e-6))
    end

    # Setup of Meso Tests
    rho = 100 .* rand(n)
    ux = 0.2 .* pi .* rand(n) .- 0.1 .* pi
    uy = 0.2 .* pi .* rand(n) .- 0.1 .* pi
    uz = 0.2 .* pi .* rand(n) .- 0.1 .* pi

    cuRho = CuArray{Float64}(rho)
    cuUx = CuArray{Float64}(ux)
    cuUy = CuArray{Float64}(uy)
    cuUz = CuArray{Float64}(uz)

    # Meso Test of Likelihood With Scaling
    @testset verbose=true "Meso-Likelihood-Scaling" begin
        tester = mesoLikelihoodWithScaling(x, y, z, rho, ux, uy, uz, h, k, l, h.+G[1], k.+G[2], l.+G[3], intens, recSupport)
        rhoDeriv = ForwardDiff.gradient(rhop -> mesoLikelihoodWithScaling(x, y, z, rhop, ux, uy, uz, h, k, l, h.+G[1], k.+G[2], l.+G[3], intens, recSupport), rho)
        uxDeriv = ForwardDiff.gradient(uxp -> mesoLikelihoodWithScaling(x, y, z, rho, uxp, uy, uz, h, k, l, h.+G[1], k.+G[2], l.+G[3], intens, recSupport), ux)
        uyDeriv = ForwardDiff.gradient(uyp -> mesoLikelihoodWithScaling(x, y, z, rho, ux, uyp, uz, h, k, l, h.+G[1], k.+G[2], l.+G[3], intens, recSupport), uy)
        uzDeriv = ForwardDiff.gradient(uzp -> mesoLikelihoodWithScaling(x, y, z, rho, ux, uy, uzp, h, k, l, h.+G[1], k.+G[2], l.+G[3], intens, recSupport), uz)

        state = BcdiCore.MesoState("likelihood", true, cuIntens, G, cuH, cuK, cuL, cuRecSupport)
        BcdiCore.setpts!(state, cuX, cuY, cuZ, cuRho, cuUx, cuUy, cuUz, true)
        testee = BcdiCore.loss(state, true, true, false)

        @test isapprox(testee, tester, rtol=1e-6)
        @test all(isapprox.(Array(state.rhoDeriv), rhoDeriv, rtol=1e-6))
        @test all(isapprox.(Array(state.uxDeriv), uxDeriv, rtol=1e-6))
        @test all(isapprox.(Array(state.uyDeriv), uyDeriv, rtol=1e-6))
        @test all(isapprox.(Array(state.uzDeriv), uzDeriv, rtol=1e-6))
    end

    # Meso Test of Likelihood Without Scaling
    @testset verbose=true "Meso-Likelihood-NoScaling" begin
        tester = mesoLikelihoodWithoutScaling(x, y, z, rho, ux, uy, uz, h, k, l, h.+G[1], k.+G[2], l.+G[3], intens, recSupport)
        rhoDeriv = ForwardDiff.gradient(rhop -> mesoLikelihoodWithoutScaling(x, y, z, rhop, ux, uy, uz, h, k, l, h.+G[1], k.+G[2], l.+G[3], intens, recSupport), rho)
        uxDeriv = ForwardDiff.gradient(uxp -> mesoLikelihoodWithoutScaling(x, y, z, rho, uxp, uy, uz, h, k, l, h.+G[1], k.+G[2], l.+G[3], intens, recSupport), ux)
        uyDeriv = ForwardDiff.gradient(uyp -> mesoLikelihoodWithoutScaling(x, y, z, rho, ux, uyp, uz, h, k, l, h.+G[1], k.+G[2], l.+G[3], intens, recSupport), uy)
        uzDeriv = ForwardDiff.gradient(uzp -> mesoLikelihoodWithoutScaling(x, y, z, rho, ux, uy, uzp, h, k, l, h.+G[1], k.+G[2], l.+G[3], intens, recSupport), uz)

        state = BcdiCore.MesoState("likelihood", false, cuIntens, G, cuH, cuK, cuL, cuRecSupport)
        BcdiCore.setpts!(state, cuX, cuY, cuZ, cuRho, cuUx, cuUy, cuUz, true)
        testee = BcdiCore.loss(state, true, true, false)

        @test isapprox(testee, tester, rtol=1e-6)
        @test all(isapprox.(Array(state.rhoDeriv), rhoDeriv, rtol=1e-6))
        @test all(isapprox.(Array(state.uxDeriv), uxDeriv, rtol=1e-6))
        @test all(isapprox.(Array(state.uyDeriv), uyDeriv, rtol=1e-6))
        @test all(isapprox.(Array(state.uzDeriv), uzDeriv, rtol=1e-6))
    end

    # Meso Test of L2 Norm With Scaling
    @testset verbose=true "Meso-L2-Scaling" begin
        tester = mesoL2WithScaling(x, y, z, rho, ux, uy, uz, h, k, l, h.+G[1], k.+G[2], l.+G[3], intens, recSupport)
        rhoDeriv = ForwardDiff.gradient(rhop -> mesoL2WithScaling(x, y, z, rhop, ux, uy, uz, h, k, l, h.+G[1], k.+G[2], l.+G[3], intens, recSupport), rho)
        uxDeriv = ForwardDiff.gradient(uxp -> mesoL2WithScaling(x, y, z, rho, uxp, uy, uz, h, k, l, h.+G[1], k.+G[2], l.+G[3], intens, recSupport), ux)
        uyDeriv = ForwardDiff.gradient(uyp -> mesoL2WithScaling(x, y, z, rho, ux, uyp, uz, h, k, l, h.+G[1], k.+G[2], l.+G[3], intens, recSupport), uy)
        uzDeriv = ForwardDiff.gradient(uzp -> mesoL2WithScaling(x, y, z, rho, ux, uy, uzp, h, k, l, h.+G[1], k.+G[2], l.+G[3], intens, recSupport), uz)

        state = BcdiCore.MesoState("L2", true, cuIntens, G, cuH, cuK, cuL, cuRecSupport)
        BcdiCore.setpts!(state, cuX, cuY, cuZ, cuRho, cuUx, cuUy, cuUz, true)
        testee = BcdiCore.loss(state, true, true, false)

        @test isapprox(testee, tester, rtol=1e-6)
        @test all(isapprox.(Array(state.rhoDeriv), rhoDeriv, rtol=1e-6))
        @test all(isapprox.(Array(state.uxDeriv), uxDeriv, rtol=1e-6))
        @test all(isapprox.(Array(state.uyDeriv), uyDeriv, rtol=1e-6))
        @test all(isapprox.(Array(state.uzDeriv), uzDeriv, rtol=1e-6))
    end

    # Atomic Test of L2 Norm Without Scaling
    @testset verbose=true "Meso-L2-NoScaling" begin
        tester = mesoL2WithoutScaling(x, y, z, rho, ux, uy, uz, h, k, l, h.+G[1], k.+G[2], l.+G[3], intens, recSupport)
        rhoDeriv = ForwardDiff.gradient(rhop -> mesoL2WithoutScaling(x, y, z, rhop, ux, uy, uz, h, k, l, h.+G[1], k.+G[2], l.+G[3], intens, recSupport), rho)
        uxDeriv = ForwardDiff.gradient(uxp -> mesoL2WithoutScaling(x, y, z, rho, uxp, uy, uz, h, k, l, h.+G[1], k.+G[2], l.+G[3], intens, recSupport), ux)
        uyDeriv = ForwardDiff.gradient(uyp -> mesoL2WithoutScaling(x, y, z, rho, ux, uyp, uz, h, k, l, h.+G[1], k.+G[2], l.+G[3], intens, recSupport), uy)
        uzDeriv = ForwardDiff.gradient(uzp -> mesoL2WithoutScaling(x, y, z, rho, ux, uy, uzp, h, k, l, h.+G[1], k.+G[2], l.+G[3], intens, recSupport), uz)

        state = BcdiCore.MesoState("L2", false, cuIntens, G, cuH, cuK, cuL, cuRecSupport)
        BcdiCore.setpts!(state, cuX, cuY, cuZ, cuRho, cuUx, cuUy, cuUz, true)
        testee = BcdiCore.loss(state, true, true, false)

        @test isapprox(testee, tester, rtol=1e-6)
        @test all(isapprox.(Array(state.rhoDeriv), rhoDeriv, rtol=1e-6))
        @test all(isapprox.(Array(state.uxDeriv), uxDeriv, rtol=1e-6))
        @test all(isapprox.(Array(state.uyDeriv), uyDeriv, rtol=1e-6))
        @test all(isapprox.(Array(state.uzDeriv), uzDeriv, rtol=1e-6))
    end

    # Setup of Multi Tests
    mx = 1.8 .* pi .* rand(n) .+ 0.1 .* pi
    my = 1.8 .* pi .* rand(n) .+ 0.1 .* pi
    mz = 1.8 .* pi .* rand(n) .+ 0.1 .* pi

    cuMx = CuArray{Float64}(mx)
    cuMy = CuArray{Float64}(my)
    cuMz = CuArray{Float64}(mz)

    # Multi Test of Likelihood With Scaling
    @testset verbose=true "Multi-Likelihood-Scaling" begin
        tester = multiLikelihoodWithScaling(x, y, z, mx, my, mz, rho, ux, uy, uz, h, k, l, h.+G[1], k.+G[2], l.+G[3], intens, recSupport)
        xDeriv = ForwardDiff.gradient(xp -> multiLikelihoodWithScaling(xp, y, z, mx, my, mz, rho, ux, uy, uz, h, k, l, h.+G[1], k.+G[2], l.+G[3], intens, recSupport), x)
        yDeriv = ForwardDiff.gradient(yp -> multiLikelihoodWithScaling(x, yp, z, mx, my, mz, rho, ux, uy, uz, h, k, l, h.+G[1], k.+G[2], l.+G[3], intens, recSupport), y)
        zDeriv = ForwardDiff.gradient(zp -> multiLikelihoodWithScaling(x, y, zp, mx, my, mz, rho, ux, uy, uz, h, k, l, h.+G[1], k.+G[2], l.+G[3], intens, recSupport), z)
        rhoDeriv = ForwardDiff.gradient(rhop -> multiLikelihoodWithScaling(x, y, z, mx, my, mz, rhop, ux, uy, uz, h, k, l, h.+G[1], k.+G[2], l.+G[3], intens, recSupport), rho)
        uxDeriv = ForwardDiff.gradient(uxp -> multiLikelihoodWithScaling(x, y, z, mx, my, mz, rho, uxp, uy, uz, h, k, l, h.+G[1], k.+G[2], l.+G[3], intens, recSupport), ux)
        uyDeriv = ForwardDiff.gradient(uyp -> multiLikelihoodWithScaling(x, y, z, mx, my, mz, rho, ux, uyp, uz, h, k, l, h.+G[1], k.+G[2], l.+G[3], intens, recSupport), uy)
        uzDeriv = ForwardDiff.gradient(uzp -> multiLikelihoodWithScaling(x, y, z, mx, my, mz, rho, ux, uy, uzp, h, k, l, h.+G[1], k.+G[2], l.+G[3], intens, recSupport), uz)

        state = BcdiCore.MultiState("likelihood", true, cuIntens, G, cuH, cuK, cuL, cuRecSupport)
        BcdiCore.setpts!(state, cuX, cuY, cuZ, cuMx, cuMy, cuMz, cuRho, cuUx, cuUy, cuUz, true)
        testee = BcdiCore.loss(state, true, true, false)

        @test isapprox(testee, tester, rtol=1e-6)
        @test all(isapprox.(Array(state.xDeriv), xDeriv, rtol=1e-6))
        @test all(isapprox.(Array(state.yDeriv), yDeriv, rtol=1e-6))
        @test all(isapprox.(Array(state.zDeriv), zDeriv, rtol=1e-6))
        @test all(isapprox.(Array(state.rhoDeriv), rhoDeriv, rtol=1e-6))
        @test all(isapprox.(Array(state.uxDeriv), uxDeriv, rtol=1e-6))
        @test all(isapprox.(Array(state.uyDeriv), uyDeriv, rtol=1e-6))
        @test all(isapprox.(Array(state.uzDeriv), uzDeriv, rtol=1e-6))
    end

    # Multi Test of Likelihood Without Scaling
    @testset verbose=true "Multi-Likelihood-NoScaling" begin
        tester = multiLikelihoodWithoutScaling(x, y, z, mx, my, mz, rho, ux, uy, uz, h, k, l, h.+G[1], k.+G[2], l.+G[3], intens, recSupport)
        xDeriv = ForwardDiff.gradient(xp -> multiLikelihoodWithoutScaling(xp, y, z, mx, my, mz, rho, ux, uy, uz, h, k, l, h.+G[1], k.+G[2], l.+G[3], intens, recSupport), x)
        yDeriv = ForwardDiff.gradient(yp -> multiLikelihoodWithoutScaling(x, yp, z, mx, my, mz, rho, ux, uy, uz, h, k, l, h.+G[1], k.+G[2], l.+G[3], intens, recSupport), y)
        zDeriv = ForwardDiff.gradient(zp -> multiLikelihoodWithoutScaling(x, y, zp, mx, my, mz, rho, ux, uy, uz, h, k, l, h.+G[1], k.+G[2], l.+G[3], intens, recSupport), z)
        rhoDeriv = ForwardDiff.gradient(rhop -> multiLikelihoodWithoutScaling(x, y, z, mx, my, mz, rhop, ux, uy, uz, h, k, l, h.+G[1], k.+G[2], l.+G[3], intens, recSupport), rho)
        uxDeriv = ForwardDiff.gradient(uxp -> multiLikelihoodWithoutScaling(x, y, z, mx, my, mz, rho, uxp, uy, uz, h, k, l, h.+G[1], k.+G[2], l.+G[3], intens, recSupport), ux)
        uyDeriv = ForwardDiff.gradient(uyp -> multiLikelihoodWithoutScaling(x, y, z, mx, my, mz, rho, ux, uyp, uz, h, k, l, h.+G[1], k.+G[2], l.+G[3], intens, recSupport), uy)
        uzDeriv = ForwardDiff.gradient(uzp -> multiLikelihoodWithoutScaling(x, y, z, mx, my, mz, rho, ux, uy, uzp, h, k, l, h.+G[1], k.+G[2], l.+G[3], intens, recSupport), uz)

        state = BcdiCore.MultiState("likelihood", false, cuIntens, G, cuH, cuK, cuL, cuRecSupport)
        BcdiCore.setpts!(state, cuX, cuY, cuZ, cuMx, cuMy, cuMz, cuRho, cuUx, cuUy, cuUz, true)
        testee = BcdiCore.loss(state, true, true, false)

        @test isapprox(testee, tester, rtol=1e-6)
        @test all(isapprox.(Array(state.xDeriv), xDeriv, rtol=1e-6))
        @test all(isapprox.(Array(state.yDeriv), yDeriv, rtol=1e-6))
        @test all(isapprox.(Array(state.zDeriv), zDeriv, rtol=1e-6))
        @test all(isapprox.(Array(state.rhoDeriv), rhoDeriv, rtol=1e-6))
        @test all(isapprox.(Array(state.uxDeriv), uxDeriv, rtol=1e-6))
        @test all(isapprox.(Array(state.uyDeriv), uyDeriv, rtol=1e-6))
        @test all(isapprox.(Array(state.uzDeriv), uzDeriv, rtol=1e-6))
    end

    # Multi Test of L2 Norm With Scaling
    @testset verbose=true "Multi-L2-Scaling" begin
        tester = multiL2WithScaling(x, y, z, mx, my, mz, rho, ux, uy, uz, h, k, l, h.+G[1], k.+G[2], l.+G[3], intens, recSupport)
        xDeriv = ForwardDiff.gradient(xp -> multiL2WithScaling(xp, y, z, mx, my, mz, rho, ux, uy, uz, h, k, l, h.+G[1], k.+G[2], l.+G[3], intens, recSupport), x)
        yDeriv = ForwardDiff.gradient(yp -> multiL2WithScaling(x, yp, z, mx, my, mz, rho, ux, uy, uz, h, k, l, h.+G[1], k.+G[2], l.+G[3], intens, recSupport), y)
        zDeriv = ForwardDiff.gradient(zp -> multiL2WithScaling(x, y, zp, mx, my, mz, rho, ux, uy, uz, h, k, l, h.+G[1], k.+G[2], l.+G[3], intens, recSupport), z)
        rhoDeriv = ForwardDiff.gradient(rhop -> multiL2WithScaling(x, y, z, mx, my, mz, rhop, ux, uy, uz, h, k, l, h.+G[1], k.+G[2], l.+G[3], intens, recSupport), rho)
        uxDeriv = ForwardDiff.gradient(uxp -> multiL2WithScaling(x, y, z, mx, my, mz, rho, uxp, uy, uz, h, k, l, h.+G[1], k.+G[2], l.+G[3], intens, recSupport), ux)
        uyDeriv = ForwardDiff.gradient(uyp -> multiL2WithScaling(x, y, z, mx, my, mz, rho, ux, uyp, uz, h, k, l, h.+G[1], k.+G[2], l.+G[3], intens, recSupport), uy)
        uzDeriv = ForwardDiff.gradient(uzp -> multiL2WithScaling(x, y, z, mx, my, mz, rho, ux, uy, uzp, h, k, l, h.+G[1], k.+G[2], l.+G[3], intens, recSupport), uz)

        state = BcdiCore.MultiState("L2", true, cuIntens, G, cuH, cuK, cuL, cuRecSupport)
        BcdiCore.setpts!(state, cuX, cuY, cuZ, cuMx, cuMy, cuMz, cuRho, cuUx, cuUy, cuUz, true)
        testee = BcdiCore.loss(state, true, true, false)

        @test isapprox(testee, tester, rtol=1e-6)
        @test all(isapprox.(Array(state.xDeriv), xDeriv, rtol=1e-6))
        @test all(isapprox.(Array(state.yDeriv), yDeriv, rtol=1e-6))
        @test all(isapprox.(Array(state.zDeriv), zDeriv, rtol=1e-6))
        @test all(isapprox.(Array(state.rhoDeriv), rhoDeriv, rtol=1e-6))
        @test all(isapprox.(Array(state.uxDeriv), uxDeriv, rtol=1e-6))
        @test all(isapprox.(Array(state.uyDeriv), uyDeriv, rtol=1e-6))
        @test all(isapprox.(Array(state.uzDeriv), uzDeriv, rtol=1e-6))
    end

    # Atomic Test of L2 Norm Without Scaling
    @testset verbose=true "Multi-L2-NoScaling" begin
        tester = multiL2WithoutScaling(x, y, z, mx, my, mz, rho, ux, uy, uz, h, k, l, h.+G[1], k.+G[2], l.+G[3], intens, recSupport)
        xDeriv = ForwardDiff.gradient(xp -> multiL2WithoutScaling(xp, y, z, mx, my, mz, rho, ux, uy, uz, h, k, l, h.+G[1], k.+G[2], l.+G[3], intens, recSupport), x)
        yDeriv = ForwardDiff.gradient(yp -> multiL2WithoutScaling(x, yp, z, mx, my, mz, rho, ux, uy, uz, h, k, l, h.+G[1], k.+G[2], l.+G[3], intens, recSupport), y)
        zDeriv = ForwardDiff.gradient(zp -> multiL2WithoutScaling(x, y, zp, mx, my, mz, rho, ux, uy, uz, h, k, l, h.+G[1], k.+G[2], l.+G[3], intens, recSupport), z)
        rhoDeriv = ForwardDiff.gradient(rhop -> multiL2WithoutScaling(x, y, z, mx, my, mz, rhop, ux, uy, uz, h, k, l, h.+G[1], k.+G[2], l.+G[3], intens, recSupport), rho)
        uxDeriv = ForwardDiff.gradient(uxp -> multiL2WithoutScaling(x, y, z, mx, my, mz, rho, uxp, uy, uz, h, k, l, h.+G[1], k.+G[2], l.+G[3], intens, recSupport), ux)
        uyDeriv = ForwardDiff.gradient(uyp -> multiL2WithoutScaling(x, y, z, mx, my, mz, rho, ux, uyp, uz, h, k, l, h.+G[1], k.+G[2], l.+G[3], intens, recSupport), uy)
        uzDeriv = ForwardDiff.gradient(uzp -> multiL2WithoutScaling(x, y, z, mx, my, mz, rho, ux, uy, uzp, h, k, l, h.+G[1], k.+G[2], l.+G[3], intens, recSupport), uz)

        state = BcdiCore.MultiState("L2", false, cuIntens, G, cuH, cuK, cuL, cuRecSupport)
        BcdiCore.setpts!(state, cuX, cuY, cuZ, cuMx, cuMy, cuMz, cuRho, cuUx, cuUy, cuUz, true)
        testee = BcdiCore.loss(state, true, true, false)

        @test isapprox(testee, tester, rtol=1e-6)
        @test all(isapprox.(Array(state.xDeriv), xDeriv, rtol=1e-6))
        @test all(isapprox.(Array(state.yDeriv), yDeriv, rtol=1e-6))
        @test all(isapprox.(Array(state.zDeriv), zDeriv, rtol=1e-6))
        @test all(isapprox.(Array(state.rhoDeriv), rhoDeriv, rtol=1e-6))
        @test all(isapprox.(Array(state.uxDeriv), uxDeriv, rtol=1e-6))
        @test all(isapprox.(Array(state.uyDeriv), uyDeriv, rtol=1e-6))
        @test all(isapprox.(Array(state.uzDeriv), uzDeriv, rtol=1e-6))
    end
end
