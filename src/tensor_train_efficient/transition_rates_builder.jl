# Tensor_trains 
using TensorTrains, TensorCast, Tullio, LogarithmicNumbers, ProgressMeter, LinearAlgebra
using TensorTrains: compress!, TruncBondThresh  


###########################################################
# TENSOR TRAINS BUILDERS
###########################################################

###
# Build the tensor train of transition per step of a parallel update dynamics
###

function build_parallel_transition_tt(transition_rate, params, Q::Int = 2, σ = x -> 2x - 3)
    N = length(params.h_vector)  # Número de sitios en la cadena

    function delta_vector(spin_value)
        return [spin_value == σ(q) ? 1.0 : 0.0 for q in 1:Q]'
    end

    tensors = Array{Float64, 4}[]  # Vector de arrays 4D
    
    # ============================================
    # SITIO 1: A₁[1, Q², σ₁ᵗ, σ₁ᵗ⁺¹]
    # Forma: (f₁(σ₁ᵗ, σ₁ᵗ⁺¹, ωⱼ))ⱼ · (Iᵣ ⊗ v_σ₁ᵗ)
    # f₁ es un vector fila 1×Q
    # v_σ₁ᵗ es un vector fila 1×Q
    # (Iᵣ ⊗ v_σ₁ᵗ) es una matriz Q×Q²
    # Resultado: (1×Q) · (Q×Q²) = 1×Q²
    # ============================================
    A1 = zeros(1, Q^2, Q, Q)
    
    for sigma_t in 1:Q          # σ₁ᵗ
        for sigma_t_plus in 1:Q  # σ₁ᵗ⁺¹
            
            # Vector fila f₁(σ₁ᵗ, σ₁ᵗ⁺¹, ωⱼ) de dimensión 1×Q
            # ωⱼ representa los posibles valores de σ₂ᵗ
            f_vector = zeros(1, Q)
            for j in 1:Q
                omega_j = σ(j)  # Valor de σ₂ᵗ
                neighbors = [σ(sigma_t), omega_j]  # [σ₁ᵗ, σ₂ᵗ]
                f_vector[1, j] = transition_rate(neighbors, σ(sigma_t_plus), 1, params)
            end
            
            # v_σ₁ᵗ es un vector fila 1×Q
            v_sigma = delta_vector(σ(sigma_t))
            
            # (Iᵣ ⊗ v_σ₁ᵗ) es una matriz Q×Q²
            I_kron_v = kron(Matrix(I, Q, Q), v_sigma)  # Q×Q²
            
            # f₁ · (Iᵣ ⊗ v_σ₁ᵗ) = (1×Q) · (Q×Q²) = 1×Q²
            A1[1, :, sigma_t, sigma_t_plus] = f_vector * I_kron_v
        end
    end
    
    push!(tensors, A1)
    
    # ============================================
    # SITIOS INTERMEDIOS: Aᵢ[Q², Q², σᵢᵗ, σᵢᵗ⁺¹]
    # Forma: (v_σᵢᵗᵀ ⊗ Iᵣ) · Fᵢ(σᵢᵗ, σᵢᵗ⁺¹) · (Iᵣ ⊗ v_σᵢᵗ)
    # v_σᵢᵗ es un vector fila 1×Q, entonces v_σᵢᵗᵀ es un vector columna Q×1
    # (v_σᵢᵗᵀ ⊗ Iᵣ) es una matriz Q²×Q
    # Fᵢ es una matriz Q×Q
    # (Iᵣ ⊗ v_σᵢᵗ) es una matriz Q×Q²
    # Resultado: (Q²×Q) · (Q×Q) · (Q×Q²) = Q²×Q²
    # ============================================
    for i in 2:N-1
        Ai = zeros(Q^2, Q^2, Q, Q)
        
        for sigma_t in 1:Q         # σᵢᵗ
            for sigma_t_plus in 1:Q # σᵢᵗ⁺¹
                
                # Fᵢ(σᵢᵗ, σᵢᵗ⁺¹) es una matriz Q×Q
                # F_i[k,l] = fᵢ(ωₖ, σᵢᵗ, σᵢᵗ⁺¹, ωₗ)
                # donde ωₖ = σᵢ₋₁ᵗ y ωₗ = σᵢ₊₁ᵗ
                
                F_matrix = zeros(Q, Q)
                
                for k in 1:Q  # ωₖ = σᵢ₋₁ᵗ (vecino izquierdo)
                    for l in 1:Q  # ωₗ = σᵢ₊₁ᵗ (vecino derecho)
                        omega_k = σ(k)
                        omega_l = σ(l)
                        neighbors = [omega_k, σ(sigma_t), omega_l]  # [σᵢ₋₁ᵗ, σᵢᵗ, σᵢ₊₁ᵗ]
                        F_matrix[k, l] = transition_rate(neighbors, σ(sigma_t_plus), i, params)
                    end
                end
                
                # v_σᵢᵗ es un vector fila 1×Q
                v_sigma = delta_vector(σ(sigma_t))
                
                # v_σᵢᵗᵀ es un vector columna Q×1
                v_sigma_T = transpose(v_sigma)
                
                # (v_σᵢᵗᵀ ⊗ Iᵣ) es una matriz Q²×Q
                vT_kron_I = kron(v_sigma_T, Matrix(I, Q, Q))
                
                # (Iᵣ ⊗ v_σᵢᵗ) es una matriz Q×Q²
                I_kron_v = kron(Matrix(I, Q, Q), v_sigma)
                
                # Producto: (Q²×Q) · (Q×Q) · (Q×Q²) = Q²×Q²
                Ai[:, :, sigma_t, sigma_t_plus] = vT_kron_I * F_matrix * I_kron_v
            end
        end
        
        push!(tensors, Ai)
    end
    
    # ============================================
    # SITIO N: Aₙ[Q², 1, σₙᵗ, σₙᵗ⁺¹]
    # Forma: (v_σₙᵗᵀ ⊗ Iᵣ) · (fₙ(σₙᵗ, σₙᵗ⁺¹, ωⱼ))ⱼ
    # v_σₙᵗ es un vector fila 1×Q, entonces v_σₙᵗᵀ es un vector columna Q×1
    # (v_σₙᵗᵀ ⊗ Iᵣ) es una matriz Q²×Q
    # fₙ es un vector columna Q×1
    # Resultado: (Q²×Q) · (Q×1) = Q²×1
    # ============================================
    AN = zeros(Q^2, 1, Q, Q)
    
    for sigma_t in 1:Q         # σₙᵗ
        for sigma_t_plus in 1:Q # σₙᵗ⁺¹
            
            # Vector columna fₙ(σₙᵗ, σₙᵗ⁺¹, ωⱼ) de dimensión Q×1
            # ωⱼ representa los posibles valores de σₙ₋₁ᵗ
            f_vector = zeros(Q, 1)
            for j in 1:Q
                omega_j = σ(j)  # Valor de σₙ₋₁ᵗ
                neighbors = [omega_j, σ(sigma_t)]  # [σₙ₋₁ᵗ, σₙᵗ]
                f_vector[j, 1] = transition_rate(neighbors, σ(sigma_t_plus), N, params)
            end
            
            # v_σₙᵗ es un vector fila 1×Q
            v_sigma = delta_vector(σ(sigma_t))
            
            # v_σₙᵗᵀ es un vector columna Q×1
            v_sigma_T = transpose(v_sigma)
            
            # (v_σₙᵗᵀ ⊗ Iᵣ) es una matriz Q²×Q
            vT_kron_I = kron(v_sigma_T, Matrix(I, Q, Q))
            
            # (v_σₙᵗᵀ ⊗ Iᵣ) · fₙ = (Q²×Q) · (Q×1) = Q²×1
            AN[:, 1, sigma_t, sigma_t_plus] = vT_kron_I * f_vector
        end
    end
    
    push!(tensors, AN)
    
    # Crear y retornar el TensorTrain
    return TensorTrain(tensors)
end


###
# Build the tensor train of a single step transition rate of a sequential update dynamics 
###
function build_sequential_transition_tt(transition_rate, params, bond = 5, Q::Int = 2, σ = x -> 2x - 3)
    N = length(params.h_vector)
    
    function transition_event(i, params, transition_rate)
        if i == 1
            tt_i = Array{Float64, 4}[]
            A1 = zeros(1, Q, Q, Q)
            for sigma in 1:Q
                for sigma_plus in 1:Q
                    f_vector = zeros(1, Q)
                    for j in 1:Q
                        omega_j = σ(j)
                        neighbors = [σ(sigma), omega_j]
                        f_vector[1, j] = transition_rate(neighbors, σ(sigma_plus), 1, params)
                    end
                    A1[1, :, sigma, sigma_plus] = f_vector
                end
            end
            push!(tt_i, A1)
            push!(tt_i, [1 ; 0 ;;; 0 ; 0 ;;;; 0 ; 0 ;;; 0 ; 1])
            foreach(_ -> push!(tt_i, [1 ;;; 0 ;;;; 0 ;;; 1]) , 3:N)
        
        elseif i == N
            tt_i = Array{Float64, 4}[]
            foreach(_ -> push!(tt_i, [1 ;;; 0 ;;;; 0 ;;; 1]) , 1:N-2)
            push!(tt_i, [1  0;;; 0 0 ;;;; 0 0 ;;; 0 1])
            AN = zeros(Q, 1, Q, Q)
            for sigma in 1:Q
                for sigma_plus in 1:Q
                    f_vector = zeros(Q, 1)
                    for j in 1:Q
                        omega_j = σ(j)
                        neighbors = [omega_j, σ(sigma)]
                        f_vector[j, 1] = transition_rate(neighbors, σ(sigma_plus), N, params)
                    end
                    AN[:, 1, sigma, sigma_plus] = f_vector
                end
            end
            push!(tt_i, AN)

        else
            tt_i = Array{Float64, 4}[]
            foreach(_ -> push!(tt_i, [1 ;;; 0 ;;;; 0 ;;; 1]) , 1:i-2)
            push!(tt_i, [1  0;;; 0 0 ;;;;0 0 ;;; 0 1])

            Ai = zeros(Q, Q, Q, Q)
            for sigma in 1:Q
                for sigma_plus in 1:Q
                    F_matrix = zeros(Q, Q)
                    for k in 1:Q
                        for l in 1:Q
                            omega_k = σ(k)
                            omega_l = σ(l)
                            neighbors = [omega_k, σ(sigma), omega_l]
                            F_matrix[k, l] = transition_rate(neighbors, σ(sigma_plus), i, params)
                        end
                    end
                    Ai[:, :, sigma, sigma_plus] = F_matrix
                end
            end
            push!(tt_i, Ai)
            push!(tt_i, [1 ; 0 ;;; 0 ; 0 ;;;; 0 ; 0 ;;; 0 ; 1])
            foreach(_ -> push!(tt_i, [1 ;;; 0 ;;;; 0 ;;; 1]) , i+2:N)
        end
        return TensorTrain(tt_i)
    end

    tensors = transition_event(1, params, transition_rate)
    
    # Comprimir inmediatamente el primer evento
    compress!(tensors, svd_trunc=TruncBond(bond))

    for i in 2:N
        tt_i = transition_event(i, params, transition_rate)
        
        # Comprimir el nuevo evento ANTES de sumarlo
        compress!(tt_i, svd_trunc=TruncBond(bond))
        
        # Sumar
        tensors += tt_i
        
        # Comprimir después de cada suma
        compress!(tensors, svd_trunc=TruncBond(bond))
    end
    
    divide_by_constant!(tensors, N)
    return tensors
end



function build_transition_tensortrain(
    params;
    update_rule::Symbol,
    bond::Int = 5,
    Q::Int = 2, 
    σ::Function = x -> 2x - 3
    )
    if update_rule == :sequential
        return build_sequential_transition_tt(glauber_transition_rate, params, bond, Q, σ)
    elseif update_rule == :parallel
        return build_parallel_transition_tt(transition_rate_inertia, params, Q, σ)
    elseif update_rule == :metropolis
        return build_sequential_transition_tt(metropolis_transition_rate, params, bond, Q, σ)
    else
        error("Unknown update rule: $update_rule. Use :sequential, :parallel, or :metropolis.")
    end
end
    


###
# Build the tensor train for parallel chains
###
"""
    parallel_transition_tensorchain(update_rule, params)

Construye un TensorTrain que representa la matriz de transición A((x,y)ᵗ, (x,y)ᵗ⁺¹).

# Argumentos
- `update_rule`: Símbolo que indica la regla de actualización (:sequential, :parallel, :metropolis)
- `params`: Parámetros del modelo (j_vector, h_vector, beta_1, beta_2, N)
- `N`: Número de sitios en la cadena
- `Q`: Cantidad de valores posibles de cada espin
- `σ`: Función que asigna a cada índice 1,2,...,Q el valor real del espín (for example (1,2) -> (-1,1))

# Retorna
- `TensorTrain`: Representación compacta de la matriz de transición mixta
"""

function parallel_transition_tensor_train(params; update_rule, bond=5, Q = 2, σ = x -> 2x - 3)
    params_1 = (N = params.N, betas = params.betas[1], j_vector = params.j_vector, h_vector = params.h_vector, p0 = params.p0)
    params_2 = (N = params.N, betas = params.betas[2], j_vector = params.j_vector, h_vector = params.h_vector, p0 = params.p0)
    A1 = build_transition_tensortrain(params_1; update_rule, bond, Q, σ)
    A2 = build_transition_tensortrain(params_2; update_rule, bond, Q, σ)
    return mult_sep_4(A1, A2) 
end




"""
    mult_sep(A, B): Multiplies two TensorTrains A and B by separating the physical dimensions.
"""
function mult_sep_transition(A, B)
    d = map(zip(A.tensors,B.tensors)) do (a,b)
        @tullio c[m1,m2,n1,n2,x,x2] := a[m1,n1,x,x1] * b[m2,n2,x1,x2]
        @cast _[(m1,m2),(n1,n2),(x),(x2)] := c[m1,m2,n1,n2,x,x2]
    end
    return TensorTrain(d; z = A.z * B.z)   
end

###
# Build a tensor train of number of steps k
### 

function k_step_transition_tt(A::TensorTrain, steps::Int, bond::Int)
    A_step = A
    for _ in 1:steps-1
        A_step = mult_sep_transition(A_step, A)
        compress!(A_step; svd_trunc=TruncBond(bond)) 
    end
    return A_step
end


##########################################################
# Particular transition rates
###########################################################

"""
    GLAUBER DYNAMIC WITH INERTIA PARAMETER p0

Calculate the probability of transition P(σᵢᵗ⁺¹ = sigma_new | current configuration). 

# Arguments
- `sigma_neighbors`: Relevant neighboring spin values
  * Site 1: [σ₁ᵗ, σ₂ᵗ]
  * Site i (intermediate): [σᵢ₋₁ᵗ, σᵢᵗ, σᵢ₊₁ᵗ]
  * Site N: [σₙ₋₁ᵗ, σₙᵗ]
  * `sigma_new`: New value of the espin (±1)
  * `site_index`: Site index (1 to N)
  * `params`: Parameters (betas, j_vector, h_vector)

# Returns
- Probability according to Glauber dynamics
"""
function transition_rate_inertia(sigma_neighbors, sigma_new, site_index, params)
    N = length(params.h_vector)
    betas = params.betas !== nothing ? params.betas[1] : params.betas
    if site_index == 1
        # Site 1: only right neighbor
        # sigma_neighbors = [σ₁ᵗ, σ₂ᵗ]
        h_eff = params.j_vector[1] * sigma_neighbors[2] + params.h_vector[1]
        
    elseif site_index == N
        # Site N: only left neighbor
        # sigma_neighbors = [σₙ₋₁ᵗ, σₙᵗ]
        h_eff = params.j_vector[end] * sigma_neighbors[1] + params.h_vector[end]
        
    else
        # Intermediate site: left and right neighbors
        # sigma_neighbors = [σᵢ₋₁ᵗ, σᵢᵗ, σᵢ₊₁ᵗ]
        h_eff = params.j_vector[site_index - 1] * sigma_neighbors[1] + 
                params.j_vector[site_index] * sigma_neighbors[3] + 
                params.h_vector[site_index]
    end
    
    # Glauber dynamics with inertia
    return (exp(betas * sigma_new * h_eff) / (2 * cosh(betas * h_eff)))*(1-params.p0) + params.p0* (sigma_new == sigma_neighbors[site_index == 1 ? 1 : site_index == N ? 2 : 2] ? 1.0 : 0.0)
end


# Same as above but without inertia

function glauber_transition_rate(sigma_neighbors, sigma_new, site_index, params)
    N = length(params.h_vector)

    betas = params.betas !== nothing ? params.betas[1] : params.betas
    
    if site_index == 1
        # Site 1: only right neighbor
        # sigma_neighbors = [σ₁ᵗ, σ₂ᵗ]
        h_eff = params.j_vector[1] * sigma_neighbors[2] + params.h_vector[1]
        
    elseif site_index == N
        # Site N: only left neighbor
        # sigma_neighbors = [σₙ₋₁ᵗ, σₙᵗ]
        h_eff = params.j_vector[end] * sigma_neighbors[1] + params.h_vector[end]
        
    else
        # Intermediate site: left and right neighbors
        # sigma_neighbors = [σᵢ₋₁ᵗ, σᵢᵗ, σᵢ₊₁ᵗ]
        h_eff = params.j_vector[site_index - 1] * sigma_neighbors[1] + 
                params.j_vector[site_index] * sigma_neighbors[3] + 
                params.h_vector[site_index]
    end
    
    # Glauber dynamics
    return (exp(betas * sigma_new * h_eff) / (2 * cosh(betas * h_eff))) 
end


# Metropolis transition rate

# function metropolis_transition_rate(sigma_neighbors, sigma_new, site_index, params)
#     N = length(params.h_vector)


#     betas = params.betas !== nothing ? params.betas[1] : params.betas
    
#     if site_index == 1
#         # Site 1: only right neighbor
#         # sigma_neighbors = [σ₁ᵗ, σ₂ᵗ]
#         delta_E = params.j_vector[1] * sigma_neighbors[2] + params.h_vector[1]
#         delta_E *= -sigma_neighbors[1]
        
#     elseif site_index == N
#         # Site N: only left neighbor
#         # sigma_neighbors = [σₙ₋₁ᵗ, σₙᵗ]
#         delta_E = params.j_vector[end] * sigma_neighbors[1] + params.h_vector[end] 
#         delta_E *= -sigma_neighbors[2]
#     else
#         # Intermediate site: left and right neighbors
#         # sigma_neighbors = [σᵢ₋₁ᵗ, σᵢᵗ, σᵢ₊₁ᵗ]
#         delta_E = params.j_vector[site_index - 1] * sigma_neighbors[1] + 
#                 params.j_vector[site_index] * sigma_neighbors[3] + 
#                 params.h_vector[site_index]
#         delta_E *= -sigma_neighbors[2]
#     end

#     # Glauber dynamics
#     return min(exp(-2 * betas * delta_E), 1.0)
# end


function metropolis_transition_rate(sigma_neighbors, sigma_new, site_index, params)
    N = length(params.h_vector)
    betas = params.betas !== nothing ? params.betas[1] : params.betas
    
    if site_index == 1
        # Site 1: only right neighbor
        # sigma_neighbors = [σ₁ᵗ, σ₂ᵗ]
        h_eff = params.j_vector[1] * sigma_neighbors[2] + params.h_vector[1]
        delta_E = 2 * sigma_neighbors[1] * h_eff  # ← Factor 2 explícito
        
    elseif site_index == N
        # Site N: only left neighbor
        # sigma_neighbors = [σₙ₋₁ᵗ, σₙᵗ]
        h_eff = params.j_vector[end] * sigma_neighbors[1] + params.h_vector[end] 
        delta_E = 2 * sigma_neighbors[2] * h_eff  # ← Factor 2 explícito
    else
        # Intermediate site: left and right neighbors
        # sigma_neighbors = [σᵢ₋₁ᵗ, σᵢᵗ, σᵢ₊₁ᵗ]
        h_eff = params.j_vector[site_index - 1] * sigma_neighbors[1] + 
                params.j_vector[site_index] * sigma_neighbors[3] + 
                params.h_vector[site_index]
        delta_E = 2 * sigma_neighbors[2] * h_eff  # ← Factor 2 explícito
    end

    if sigma_new == sigma_neighbors[site_index == 1 ? 1 : site_index == N ? 2 : 2]
        return 1 - min(exp(-betas * delta_E), 1.0)
    else
        return min(exp(-betas * delta_E), 1.0)  # ← Ahora solo -beta (sin el 2)
    end
end
