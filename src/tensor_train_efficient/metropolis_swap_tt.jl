

function tt_swap(params, n, bond)
    tt = boltzman_swap_tt(params)

    for i in 2:n
        tt_i = boltzman_swap_n_tt(params, i)
        normalize!(tt_i)
        normalize!(tt)  # Normalizar antes de sumar
        tt += tt_i
        normalize_eachmatrix!(tt)    # Normalizar la suma
        compress!(tt; svd_trunc=TruncBond(bond))
    end

    normalize!(tt)
    tt_plus = tt + identity_tensor_train(tt)
    normalize_eachmatrix!(tt_plus)

    tt_inverse = inverse_tt(tt_plus, bond)
    return tt * tt_inverse
end



