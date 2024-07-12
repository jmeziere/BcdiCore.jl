struct NUGpuPlan{T1,T2,T3}
    forPlan::T1
    revPlan::T2
    cupy::T3
    realSpace::CuArray{ComplexF64, 1, CUDA.Mem.DeviceBuffer}
    recipSpace::CuArray{ComplexF64, 3, CUDA.Mem.DeviceBuffer}
    tempSpace::CuArray{ComplexF64, 3, CUDA.Mem.DeviceBuffer}

    function NUGpuPlan(s)
        realSpace = CUDA.zeros(ComplexF64, 0)
        recipSpace = CUDA.zeros(ComplexF64, s)
        tempSpace = CUDA.zeros(ComplexF64, s)

        cufinufft = PyCall.pyimport("cufinufft")
        cupy = PyCall.pyimport("cupy")
        forPlan = cufinufft.Plan(1, s, isign=-1, eps=1e-12, dtype="complex128")
        CUDA.synchronize()
        revPlan = cufinufft.Plan(2, s, isign=1, eps=1e-12, dtype="complex128")
        CUDA.synchronize()

        new{typeof(forPlan), typeof(revPlan), typeof(cupy)}(forPlan, revPlan, cupy, realSpace, recipSpace, tempSpace)
    end
end

function Base.:*(plan::NUGpuPlan, realSpace)
    plan.realSpace .= realSpace
    CUDA.synchronize()
    plan.forPlan.execute(
         DLPack.share(plan.realSpace, plan.cupy.from_dlpack),
         DLPack.share(plan.recipSpace, plan.cupy.from_dlpack)
    )
    CUDA.synchronize()
    plan.recipSpace .= permutedims(plan.recipSpace, (3,2,1))
end

function Base.:\(plan::NUGpuPlan, recipSpace)
    permutedims!(plan.recipSpace, recipSpace, (3,2,1))
    CUDA.synchronize()
    plan.revPlan.execute(
        DLPack.share(plan.recipSpace, plan.cupy.from_dlpack),
        DLPack.share(plan.realSpace, plan.cupy.from_dlpack)
    )
    CUDA.synchronize()
    plan.realSpace ./= length(recipSpace)
end

struct UGpuPlan{T}
    plan::T
    realSpace::CuArray{ComplexF64, 3, CUDA.Mem.DeviceBuffer}
    recipSpace::CuArray{ComplexF64, 3, CUDA.Mem.DeviceBuffer}
    tempSpace::CuArray{ComplexF64, 3, CUDA.Mem.DeviceBuffer}

    function UGpuPlan(s)
        realSpace = CUDA.zeros(ComplexF64, s)
        recipSpace = CUDA.zeros(ComplexF64, s)
        tempSpace = CUDA.zeros(ComplexF64, s)
        plan = CUFFT.plan_fft!(realSpace)

        new{typeof(plan)}(plan, realSpace, recipSpace, tempSpace)
    end
end

function Base.:*(plan::UGpuPlan, realSpace)
    plan.recipSpace .= realSpace
    plan.plan * plan.recipSpace
end

function Base.:\(plan::UGpuPlan, recipSpace)
    plan.realSpace .= recipSpace
    plan.plan \ plan.realSpace
end
