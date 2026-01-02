using TensorTrains, TensorCast, Tullio, LogarithmicNumbers, ProgressMeter, LinearAlgebra
using TensorTrains: compress!, TruncBondThresh  



# Auxiliary function to create an identity tensor train of given length and physical dimensions
identity_tensor_train(N, qs) = [ones(1,1,qs...) for _ in 1:N] |> TensorTrain
identity_tensor_train(N,qs...) = identity_tensor_train(N,qs)
identity_tensor_train(A::AbstractTensorTrain) = identity_tensor_train(length(A), size(A[1])[3:end])



# Auxiliary function to estimate the norm of a tensor train
function estimate_norm_tt(B)
    B1 = (reshape(b,size(b,1),size(b,2),prod(size(b)[3:end])) for b in B)
    return (1/abs(B.z))*only(prod([maximum(abs(b[i,j,x]) for x in axes(b,3)) for i in axes(b,1), j in axes(b,2)] for b in B1))
end


# Auxiliary functions to multiply and divide a TT by a constant
function divide_by_constant!(B, constant)
    B.z *= constant
    return B
end

function multiply_by_constant!(B, constant)
    B.z /= constant
    return B
end




#function inverse_tt_standard(B, bond)
#    steps = 5  #log( 1+estimate_norm_tt(B)) |> floor |> Int
#    return inverse_tt(B, steps, bond)
#end


"""
* Kronecker multiplication of two tensors in Tensor Train format with matching physical dimensions.
"""
function Base.:*(A::T, B::T) where T<:AbstractTensorTrain
    C = map(zip(A.tensors, B.tensors)) do (a, b)  
        @assert size(a)[3:end] == size(b)[3:end]

        ar = reshape(a, size(a,1), size(a,2), prod(size(a)[3:end]))
        br = reshape(b, size(b,1), size(b,2), prod(size(b)[3:end]))
        
        @cast c[(ia,ib),(ja,jb),x] := ar[ia,ja,x] * br[ib,jb,x]

        reshape(c, size(c,1), size(c,2), size(a)[3:end]...)
    end

    T(C; z = A.z * B.z)
end



function absorb_z_into_matrices!(tt::AbstractTensorTrain)
    L = length(tt)
    
    if tt.z != 1 && tt.z != 0
        # Para Logarithmic numbers
        z_val = Float64(tt.z)  # Convierte a Float64 si es necesario
        factor = exp(log(abs(z_val)) / L)
        sign_factor = sign(z_val)
        
        for idx in 1:L
            if idx == 1
                tt[idx] *= sign_factor / factor
            else
                tt[idx] *= 1/factor
            end
        end
        
        tt.z = one(typeof(tt.z))  # Resetea a 1 del tipo correcto
    end
end











###############################################################################
###############################################################################
# Revisar 

"""
    mult_sep(A, B): Multiplies two TensorTrains A and B by separating the physical dimensions.
"""
function mult_sep_4(A, B)
    d = map(zip(A.tensors,B.tensors)) do (a,b)
        @tullio c[m1,m2,n1,n2,x,y,x1,y1] := a[m1,n1,x,x1] * b[m2,n2,y,y1]
        @cast _[(m1,m2),(n1,n2),(x,y),(x1,y1)] := c[m1,m2,n1,n2,x,y,x1,y1]
    end
    return TensorTrain(d; z = A.z * B.z)   
end


function mult_sep_3(A, B)
    d = map(zip(A.tensors,B.tensors)) do (a,b)
        @tullio c[m1,m2,n1,n2,x,y] := a[m1,n1,x] * b[m2,n2,y]
        @cast _[(m1,m2),(n1,n2),(x,y)] := c[m1,m2,n1,n2,x,y]
    end
    return TensorTrain(d; z = A.z * B.z)   
end

###############################################################################
###############################################################################









######################################################
# Revisar
######################################################


# Auxiliary function to calculate the inverse of a tensor train using the Newton method
function inverse_tt(B, bond; steps = 150)
    normalize_eachmatrix!(B)
    B0 = 1 / (2*estimate_norm_tt(B))
    @show B0

    # Bn = B0 * (2I - B0 * B)
    temp = multiply_by_constant!(deepcopy(B), B0)
    two = multiply_by_constant!(identity_tensor_train(B), 2)
    Bn = multiply_by_constant!( two - temp, B0)
    
    #Bn = Bn - temp

    @showprogress for t in 1:steps
        # X_{n+1} = X_n * (2I - B * X_n)
        temp1 = B * Bn
        normalize_eachmatrix!(temp1)
        compress!(temp1; svd_trunc=TruncBond(bond))

        # normalize_eachmatrix!(temp1)
        # absorb_z_into_matrices!(temp1)
        Bnn = two - temp1

        # normalize_eachmatrix!(Bnn)
        Bn = Bnn * Bn
    
        normalize_eachmatrix!(Bn)
        compress!(Bn; svd_trunc=TruncBond(bond))
        
        # normalize_eachmatrix!(Bn)
        # if estimate_norm_tt(Bn) < B0
        #     Bn.z *= B0 / estimate_norm_tt(Bn) 
        # end
        @show estimate_norm_tt(Bn) Bn.z
    end
    normalize_eachmatrix!(Bn)
    return Bn
end



# Auxiliary function to calculate the inverse of a tensor train using the Newton method
function inverse_tt_improve(B, bond; steps = 150)
    normalize_eachmatrix!(B)
    B0 = 1 / estimate_norm_tt(B)
    @show B0

    # Bn = B0 * (2I - B0 * B)
    temp = multiply_by_constant!(deepcopy(B), B0)
    two = multiply_by_constant!(identity_tensor_train(B), 2)
    Bn = multiply_by_constant!( two - temp, B0)
    
    #Bn = Bn - temp

    z_ = 1.0
    z_acc = Logarithmic(1.0)
    @showprogress for t in 1:steps
        # X_{n+1} = X_n * (2I - B * X_n)
        temp1 = B * Bn
        normalize_eachmatrix!(temp1)
        compress!(temp1; svd_trunc=TruncBond(bond))

        # normalize_eachmatrix!(temp1)
        # absorb_z_into_matrices!(temp1)
        two_ = deepcopy(two)
        two_ = multiply_by_constant!(two_, z_)
        Bnn = two_ - temp1

        # normalize_eachmatrix!(Bnn)
        Bn = Bnn * Bn
    
        normalize_eachmatrix!(Bn)
        z_ = normalize!(Bn)
        compress!(Bn; svd_trunc=TruncBond(bond))
        z_acc *= z_^2

        @show estimate_norm_tt(Bn) Bn.z
    end
    normalize_eachmatrix!(Bn)
    return Bn, z_acc
end
