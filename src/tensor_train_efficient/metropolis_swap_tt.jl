

function tt_swap(params, n, bond)
    tt = boltzman_swap_tt(params)    # x

    for i in 2:n
        tt_i = boltzman_swap_n_tt(params, i)        # x^i
        # absorb_z_into_matrices!(tt_i)
        # absorb_z_into_matrices!(tt)  # Normalizar antes de sumar
        tt += tt_i                                  # Sumar x + x^2 + ... + x^n     
        normalize_eachmatrix!(tt)    # Normalizar la suma
        compress!(tt; svd_trunc=TruncBond(bond))
    end

    # absorb_z_into_matrices!(tt)
    tt_plus = tt + identity_tensor_train(tt)           # x + x^2 + ... + x^n + I
    normalize_eachmatrix!(tt_plus)

    tt_inverse = inverse_tt(tt_plus, bond)              # (x + x^2 + ... + x^n + I)^(-1)
    return tt * tt_inverse                       # (x + x^2 + ... + x^n ) * (x + x^2 + ... + x^n + I)^(-1)
end



