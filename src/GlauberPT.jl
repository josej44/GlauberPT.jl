module GlauberPT


using LogarithmicNumbers,  LinearAlgebra
using TensorTrains, TensorCast, Tullio
using TensorTrains: compress, TruncBondThresh
using ProgressMeter, Revise
using Random, Statistics, Distributions
using Plots, Colors


export  # Monte Carlo functions
        run_monte_carlo_general, initialize_spins, 

        glauber_transition_rate, compute_local_energy_change, apply_update!,
        parallel_update!, sequential_update!, sequential_update_with_energy!,
        metropolis_update!, metropolis_update_with_energy!,

        compute_total_energy, apply_fixed_rate_swap!, 
        apply_metropolis_swap!, apply_metropolis_swap_with_energies!,

        compute_swap_acceptance_rate, suggest_beta_schedule,
        MCParameters, validate, 

        initialize_observable_accumulators, accumulate_observables!, finalize_observables,
        compute_total_energy, 

        # Tensor train functions    
        identity_tensor_train,  estimate_norm_tt, divide_by_constant!, multiply_by_constant!,
        inverse_tt, *, mult_sep, mult_sep_transition, k_step_transition_tt

        random_params, parallel_random_params, random_P0, parallel_random_P0_fixed,

        full_subproducts_system, full_marginals_system, full_simple_ev_system, marginal_ev_system,
        full_second_order_marginals_system, correlation_next_pairs_simple, second_moment_system,
        correlation_between_spins_system, energy_function, energy_function_simple, 
        system_description,system_description_over_time,

        boltzman_tt, boltzman_n_tt, boltzman_swap_tt, boltzman_swap_n_tt,

        build_parallel_transition_tt, build_sequential_transition_tt,
        build_transition_tensortrain, parallel_transition_tensor_train, 
        transition_rate_inertia, glauber_transition_rate, metropolis_transition_rate,

        ConstantSwap, EnergySwap, distribution_b_tt, 
        tt_swap







include("monte_carlo_efficient/parameters.jl")
include("monte_carlo_efficient/swap_criteria.jl")
include("monte_carlo_efficient/update_rules.jl")
include("monte_carlo_efficient/observables.jl")
include("monte_carlo_efficient/monte_carlo_general.jl")

include("tensor_train_efficient/auxiliary_tensor_train_tools.jl")
include("tensor_train_efficient/boltzmann_distribution_tt.jl")
include("tensor_train_efficient/transition_rates_builder.jl")
include("tensor_train_efficient/initialization_params.jl")
include("tensor_train_efficient/efficient_observables.jl")
include("tensor_train_efficient/tensor_train_evolution.jl")
include("tensor_train_efficient/metropolis_swap_tt.jl")

end
