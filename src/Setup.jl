function centerPeak(intens, recSupport, loc)
    s = size(intens)
    shift = [0.,0,0]
    supShift = [0,0,0]
    for i in 1:s[1]
        for j in 1:s[2]
            for k in 1:s[3]
                shift[1] += i*sqrt(intens[i,j,k])
                shift[2] += j*sqrt(intens[i,j,k])
                shift[3] += k*sqrt(intens[i,j,k])
            end
        end
    end
    if loc == "center"
        shift .= [s[1]//2+1,s[2]//2+1,s[3]//2+1] .- round.(Int64, shift ./ mapreduce(sqrt, +, intens))
        supShift .= Int64.(shift)
    elseif loc == "corner"
        shift .= [1,1,1] .- round.(Int64, shift ./ mapreduce(sqrt, +, intens))
        supShift = Int64.([s[1]//2,s[2]//2,s[3]//2] .+ shift)
    else
        println("Put a warning here, invalid variable")
    end
    intens = CuArray{Float64, 3, CUDA.Mem.DeviceBuffer}(circshift(intens,shift))

    if supShift[1] < 0
        recSupport[s[1]+1-supShift[1]:end,:,:] .= false
    elseif supShift[1] > 0
        recSupport[1:supShift[1],:,:] .= false
    end     
    if supShift[2] < 0
        recSupport[:,s[2]+1-supShift[2]:end,:] .= false
    elseif supShift[2] > 0
        recSupport[:,1:supShift[2],:] .= false
    end     
    if supShift[3] < 0
        recSupport[:,:,s[3]+1-supShift[3]:end] .= false
    elseif supShift[3] > 0
        recSupport[:,:,1:supShift[3]] .= false
    end     

    recSupport = CuArray{Float64, 3, CUDA.Mem.DeviceBuffer}(circshift(recSupport,shift))
    shift .*= -1
    return intens, recSupport, shift
end

function generateRecSpace(s)
    h = CUDA.zeros(Int64, s)
    k = CUDA.zeros(Int64, s)
    l = CUDA.zeros(Int64, s)
    for i in 1:s[1]
        h[i,:,:] .= i-1-s[1]//2
        k[:,i,:] .= i-1-s[1]//2
        l[:,:,i] .= i-1-s[1]//2
    end
    return h,k,l
end
