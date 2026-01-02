# Tensor_trains 
using TensorTrains, TensorCast, Tullio, LogarithmicNumbers, ProgressMeter, LinearAlgebra
using TensorTrains: compress!, TruncBondThresh  


function ConstantSwap(swap::Float64)

    function (B)
        swap_idx = [1, 3, 2, 4]
        tensors_swap = [T[:, :, swap_idx] for T in B.tensors]
        B_swap = TensorTrain(tensors_swap; z = B.z)        
        multiply_by_constant!(B_swap, swap)
        multiply_by_constant!(B, (1 - swap))
        return B + B_swap  
    end
end

function EnergySwap(swap::TensorTrain)
    function (B)
        swap_idx = [1, 3, 2, 4]
        tensors_swap = [T[:, :, swap_idx] for T in B.tensors]
        B_swap = TensorTrain(tensors_swap; z = B.z)
        B *(identity_tensor_train(swap) - swap)  + B_swap * swap
    end
end

function no_swap()
    function (B)
        return B
    end
end

function distribution_b_tt(
    A::TensorTrain, 
    P0::Vector{Vector{Float64}}, 
    t::Int, 
    bond::Int,
    swapfun = (B,s)->B,
#    swap::Union{TensorTrain, Float64, Nothing}=nothing,
    save::Bool=true
    )
    
    # compress!(A; svd_trunc=TruncBond(bond))
            
    B = TensorTrain([(@tullio _[1,1,x] := pi[x]) for pi in P0])
    lista_B_T = save ? [B] : nothing
    
    @showprogress for _ in 1:t
        # Evolución temporal
        B = map(zip(A.tensors, B.tensors)) do (A_i, B_i)
            @tullio new_[m1,m2,n1,n2,σ_next] := A_i[m1,n1,σ,σ_next] * B_i[m2,n2,σ]
            @cast _[(m1,m2),(n1,n2),σ_next] := new_[m1,m2,n1,n2,σ_next]
        end |> TensorTrain

        normalize_eachmatrix!(B)
        compress!(B; svd_trunc=TruncBond(bond))
        
        B = swapfun(B)
        # In case of swap
        # if typeof(swap) == Float64
            
        #     tensors_swap = [T[:, :, swap_idx] for T in B.tensors]
        #     B_swap = TensorTrain(tensors_swap; z = B.z)
            
        #     multiply_by_constant!(B_swap, swap)
        #     multiply_by_constant!(B, (1 - swap))

        #     B = B + B_swap

        # elseif typeof(swap) == TensorTrain
        #     tensors_swap = [T[:, :, swap_idx] for T in B.tensors]
        #     B_swap = TensorTrain(tensors_swap; z = B.z)

        #     B = B *(identity_tensor_train(swap) - swap)  + B_swap * swap
        # end 

        normalize_eachmatrix!(B)
        compress!(B; svd_trunc=TruncBond(bond))
        normalize!(B)

        save && push!(lista_B_T, B)
    end
    
    return save ? lista_B_T : B
end


