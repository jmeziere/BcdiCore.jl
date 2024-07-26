struct NUGpuPlan{T1,T2}
    forPlan::T1
    revPlan::T2
    realSpace::CuArray{ComplexF64, 1, CUDA.Mem.DeviceBuffer}
    recipSpace::CuArray{ComplexF64, 3, CUDA.Mem.DeviceBuffer}
    tempSpace::CuArray{ComplexF64, 3, CUDA.Mem.DeviceBuffer}

    function NUGpuPlan(s)
        realSpace = CUDA.zeros(ComplexF64, 0)
        recipSpace = CUDA.zeros(ComplexF64, s)
        tempSpace = CUDA.zeros(ComplexF64, s)

        forPlan = FINUFFT.cufinufft_makeplan(1, collect(s), -1, 1, 1e-12)
        revPlan = FINUFFT.cufinufft_makeplan(2, collect(s), 1, 1, 1e-12)

        new{typeof(forPlan), typeof(revPlan)}(forPlan, revPlan, realSpace, recipSpace, tempSpace)
    end
end

function Base.:*(plan::NUGpuPlan, realSpace)
    plan.realSpace .= realSpace
    FINUFFT.cufinufft_exec!(plan.forPlan, plan.realSpace, plan.recipSpace)
end

function Base.:\(plan::NUGpuPlan, recipSpace)
    plan.recipSpace .= recipSpace
    FINUFFT.cufinufft_exec!(plan.revPlan, plan.recipSpace, plan.realSpace)
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
