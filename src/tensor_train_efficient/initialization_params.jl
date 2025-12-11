using Random


# ============================================
# SELECCIÓN DE PARÁMETROS ALEATORIOS
# ============================================

"""
    random_params(N): Genera parámetros aleatorios para el modelo de Ising.
"""
function random_params(N, a::Float64 = -1.0, b::Float64 = 1.0)
    params = (
        N = N, 
        beta = rand(),                              # Inversa de la temperatura (β = 1/kT)
        j_vector = a .+ (b - a) .* rand(N-1) ,        # Acoplamientos J_{i,i+1} (N-1 elementos)
        h_vector = a .+ (b - a) .* rand(N) ,    # Campos externos h_i (N elementos)
        p0 = rand(),                                 # Probabilidad de mantener configuración,
    )
    return params
end

"""
    random_params(N): Genera parámetros aleatorios para el modelo de Ising paralelo.
"""
function parallel_random_params(N)
    a, b = -1.0, 1.0
    params = (
        N = N, 
        beta_1 = rand(),                              # Inversa de la temperatura (β = 1/kT)
        beta_2 = rand(),                              # Inversa de la temperatura (β = 1/kT)
        j_vector = a .+ (b - a) .* rand(N-1) ,        # Acoplamientos J_{i,i+1} (N-1 elementos)
        h_vector = a .+ (b - a) .* rand(N) ,          # Campos externos h_i (N elementos)
        p0 = rand()                                   # Probabilidad de mantener configuración,
    )
    return params
end

"""
    random_P0(N, Q): Genera una distribución de probabilidad inicial aleatoria normalizada para N sitios 
    y Q estados por espín.
"""
function random_P0(N, Q = 2)
    P0 = [rand(Q) for _ in 1:N]
    for i in 1:N
        P0[i] /= sum(P0[i])  # Normaliza cada vector de probabilidad
    end
    return P0
end


"""
    parallel_random_P0_fixed(N): Genera una distribución de probabilidad inicial fija para N sitios 
    en el modelo paralelo.
"""
function parallel_random_P0_fixed(N)
    P0 = [Float64[rand(), 0.0, 0.0, rand()] for _ in 1:N]
    for i in 1:N
        P0[i] ./= sum(P0[i])  # Normaliza cada vector de probabilidad
    end
    return P0
end
    

