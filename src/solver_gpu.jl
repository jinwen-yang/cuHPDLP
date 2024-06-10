
"""
Logging while the algorithm is running.
"""
function pdhg_specific_log(
    # problem::QuadraticProgrammingProblem,
    iteration::Int64,
    current_primal_solution::CuVector{Float64},
    current_dual_solution::CuVector{Float64},
    step_size::Float64,
    required_ratio::Union{Float64,Nothing},
    primal_weight::Float64,
)
    Printf.@printf(
        # "   %5d inv_step_size=%9g ",
        "   %5d norms=(%9g, %9g) inv_step_size=%9g ",
        iteration,
        CUDA.norm(current_primal_solution),
        CUDA.norm(current_dual_solution),
        1 / step_size,
    )
    if !isnothing(required_ratio)
        Printf.@printf(
        "   primal_weight=%18g  inverse_ss=%18g\n",
        primal_weight,
        required_ratio
        )
    else
        Printf.@printf(
        "   primal_weight=%18g \n",
        primal_weight,
        )
    end
end



"""
Logging for when the algorithm terminates.
"""
function pdhg_final_log(
    problem::QuadraticProgrammingProblem,
    avg_primal_solution::Vector{Float64},
    avg_dual_solution::Vector{Float64},
    verbosity::Int64,
    iteration::Int64,
    termination_reason::TerminationReason,
    last_iteration_stats::IterationStats,
)

    if verbosity >= 2
        
        # println("Solution:")
        Printf.@printf(
            "  pr_infeas=%12g pr_obj=%15.10g dual_infeas=%12g dual_obj=%15.10g\n",
            last_iteration_stats.convergence_information[1].l_inf_primal_residual,
            last_iteration_stats.convergence_information[1].primal_objective,
            last_iteration_stats.convergence_information[1].l_inf_dual_residual,
            last_iteration_stats.convergence_information[1].dual_objective
        )
        Printf.@printf(
            "  primal norms: L1=%15.10g, L2=%15.10g, Linf=%15.10g\n",
            CUDA.norm(avg_primal_solution, 1),
            CUDA.norm(avg_primal_solution),
            CUDA.norm(avg_primal_solution, Inf)
        )
        Printf.@printf(
            "  dual norms:   L1=%15.10g, L2=%15.10g, Linf=%15.10g\n",
            CUDA.norm(avg_dual_solution, 1),
            CUDA.norm(avg_dual_solution),
            CUDA.norm(avg_dual_solution, Inf)
        )
    end

    generic_final_log(
        problem,
        avg_primal_solution,
        avg_dual_solution,
        last_iteration_stats,
        verbosity,
        iteration,
        termination_reason,
    )
end

function power_method_failure_probability(
    dimension::Int64,
    epsilon::Float64,
    k::Int64,
)
    if k < 2 || epsilon <= 0.0
        return 1.0
    end
    return min(0.824, 0.354 / sqrt(epsilon * (k - 1))) * sqrt(dimension) * (1.0 - epsilon)^(k - 1 / 2) # FirstOrderLp.jl old version (epsilon * (k - 1)) instead of sqrt(epsilon * (k - 1)))
end

function estimate_maximum_singular_value(
    matrix::SparseMatrixCSC{Float64,Int64};
    probability_of_failure = 0.01::Float64,
    desired_relative_error = 0.1::Float64,
    seed::Int64 = 1,
)
    epsilon = 1.0 - (1.0 - desired_relative_error)^2
    x = randn(Random.MersenneTwister(seed), size(matrix, 2))

    number_of_power_iterations = 0
    while power_method_failure_probability(
        size(matrix, 2),
        epsilon,
        number_of_power_iterations,
    ) > probability_of_failure
        x = x / norm(x, 2)
        x = matrix' * (matrix * x)
        number_of_power_iterations += 1
    end
    
    return sqrt(dot(x, matrix' * (matrix * x)) / norm(x, 2)^2),
    number_of_power_iterations
end


function compute_next_primal_solution_kernel!(
    objective_vector::CuDeviceVector{Float64},
    variable_lower_bound::CuDeviceVector{Float64},
    variable_upper_bound::CuDeviceVector{Float64},
    current_primal_solution::CuDeviceVector{Float64},
    current_dual_product::CuDeviceVector{Float64},
    step_size::Float64,
    primal_weight::Float64,
    num_variables::Int64,
    delta_primal::CuDeviceVector{Float64},
)
    tx = threadIdx().x + (blockDim().x * (blockIdx().x - 0x1))
    if tx <= num_variables
        @inbounds begin
            delta_primal[tx] = current_primal_solution[tx] - (step_size / primal_weight) * (objective_vector[tx] - current_dual_product[tx])
            delta_primal[tx] = min(variable_upper_bound[tx], max(variable_lower_bound[tx], delta_primal[tx]))
            delta_primal[tx] -= current_primal_solution[tx]
        end
    end
    return 
end

function compute_next_primal_solution!(
    problem::CuLinearProgrammingProblem,
    current_primal_solution::CuVector{Float64},
    current_dual_product::CuVector{Float64},
    step_size::Float64,
    primal_weight::Float64,
    delta_primal::CuVector{Float64},
    delta_primal_product::CuVector{Float64},
)
    NumBlockPrimal = ceil(Int64, problem.num_variables/ThreadPerBlock)

    CUDA.@sync @cuda threads = ThreadPerBlock blocks = NumBlockPrimal compute_next_primal_solution_kernel!(
        problem.objective_vector,
        problem.variable_lower_bound,
        problem.variable_upper_bound,
        current_primal_solution,
        current_dual_product,
        step_size,
        primal_weight,
        problem.num_variables,
        delta_primal,
    )

    CUDA.CUSPARSE.mv!('N', 1, problem.constraint_matrix, delta_primal, 0, delta_primal_product, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
    
end

function compute_next_dual_solution_kernel!(
    right_hand_side::CuDeviceVector{Float64},
    current_dual_solution::CuDeviceVector{Float64},
    current_primal_product::CuDeviceVector{Float64},
    delta_primal_product::CuDeviceVector{Float64},
    step_size::Float64,
    primal_weight::Float64,
    extrapolation_coefficient::Float64,
    num_equalities::Int64,
    num_constraints::Int64,
    delta_dual::CuDeviceVector{Float64},
)
    tx = threadIdx().x + (blockDim().x * (blockIdx().x - 0x1))
    if tx <= num_equalities
        @inbounds begin
            delta_dual[tx] = current_dual_solution[tx] + (primal_weight * step_size) * (right_hand_side[tx] - (1 + extrapolation_coefficient) * delta_primal_product[tx] - extrapolation_coefficient * current_primal_product[tx])

            delta_dual[tx] -= current_dual_solution[tx]
        end
    elseif num_equalities + 1 <= tx <= num_constraints
        @inbounds begin
            delta_dual[tx] = current_dual_solution[tx] + (primal_weight * step_size) * (right_hand_side[tx] - (1 + extrapolation_coefficient) * delta_primal_product[tx] - extrapolation_coefficient * current_primal_product[tx])
            delta_dual[tx] = max(delta_dual[tx], 0.0)

            delta_dual[tx] -= current_dual_solution[tx]
        end
    end
    return 
end

function compute_next_dual_solution!(
    problem::CuLinearProgrammingProblem,
    current_dual_solution::CuVector{Float64},
    step_size::Float64,
    primal_weight::Float64,
    delta_primal_product::CuVector{Float64},
    current_primal_product::CuVector{Float64},
    delta_dual::CuVector{Float64};
    extrapolation_coefficient::Float64 = 1.0,
)
    NumBlockDual = ceil(Int64, problem.num_constraints/ThreadPerBlock)

    CUDA.@sync @cuda threads = ThreadPerBlock blocks = NumBlockDual compute_next_dual_solution_kernel!(
        problem.right_hand_side,
        current_dual_solution,
        current_primal_product,
        delta_primal_product,
        step_size,
        primal_weight,
        extrapolation_coefficient,
        problem.num_equalities,
        problem.num_constraints,
        delta_dual,
    )
    # CUDA.CUSPARSE.mv!('N', 1, problem.constraint_matrix_t, next_dual, 0, next_dual_product, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
end

function update_solution_in_solver_state!(
    problem::CuLinearProgrammingProblem,
    solver_state::CuPdhgSolverState,
    buffer_state::CuBufferState,
)
    weight = (solver_state.inner_count + 1) / (solver_state.inner_count + 2)

    # primal iterates
    solver_state.current_primal_solution .*= weight
    solver_state.current_primal_solution .+= weight .* buffer_state.delta_primal
    solver_state.current_primal_solution .+= (1 - weight) .* solver_state.initial_primal_solution
    # primal product
    solver_state.current_primal_product .*= weight
    solver_state.current_primal_product .+= weight .* buffer_state.delta_primal_product
    solver_state.current_primal_product .+= (1 - weight) .* solver_state.initial_primal_product

    # dual iterates
    solver_state.current_dual_solution .*= weight
    solver_state.current_dual_solution .+= weight .* buffer_state.delta_dual
    solver_state.current_dual_solution .+= (1 - weight) .* solver_state.initial_dual_solution
    # dual product
    CUDA.CUSPARSE.mv!('N', 1, problem.constraint_matrix_t, solver_state.current_dual_solution, 0, solver_state.current_dual_product, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2)
 
    solver_state.inner_count += 1   
end


function compute_interaction_and_movement(
    problem::CuLinearProgrammingProblem,
    solver_state::CuPdhgSolverState,
    buffer_state::CuBufferState,
)    
    primal_dual_interaction = CUDA.dot(buffer_state.delta_primal_product, buffer_state.delta_dual) 
    interaction = abs(primal_dual_interaction) 

    norm_delta_primal = CUDA.norm(buffer_state.delta_primal)
    norm_delta_dual = CUDA.norm(buffer_state.delta_dual)

    movement = 0.5 * solver_state.primal_weight * norm_delta_primal^2 + (0.5 / solver_state.primal_weight) * norm_delta_dual^2

    return interaction, movement
end

function take_step!(
    step_params::AdaptiveStepsizeParams,
    problem::CuLinearProgrammingProblem,
    solver_state::CuPdhgSolverState,
    buffer_state::CuBufferState,
)
    step_size = solver_state.step_size
    done = false

    while !done
        solver_state.total_number_iterations += 1

        compute_next_primal_solution!(
            problem,
            solver_state.current_primal_solution,
            solver_state.current_dual_product,
            step_size,
            solver_state.primal_weight,
            buffer_state.delta_primal,
            buffer_state.delta_primal_product,
        )

        compute_next_dual_solution!(
            problem,
            solver_state.current_dual_solution,
            step_size,
            solver_state.primal_weight,
            buffer_state.delta_primal_product,
            solver_state.current_primal_product,
            buffer_state.delta_dual,
        )

        interaction, movement = compute_interaction_and_movement(
            problem,
            solver_state,
            buffer_state,
        )

        solver_state.cumulative_kkt_passes += 1

        if interaction > 0
            step_size_limit = movement / interaction
            if movement == 0.0
                solver_state.numerical_error = true
                break
            end
        else
            step_size_limit = Inf
        end

        if step_size <= step_size_limit
            update_solution_in_solver_state!(
                problem,
                solver_state, 
                buffer_state,
            )
            done = true
        end

        first_term = (1 - 1/(solver_state.total_number_iterations + 1)^(step_params.reduction_exponent)) * step_size_limit

        second_term = (1 + 1/(solver_state.total_number_iterations + 1)^(step_params.growth_exponent)) * step_size

        step_size = min(first_term, second_term)
        
    end  
    solver_state.step_size = step_size
end

function take_step!(
    step_params::ConstantStepsizeParams,
    problem::CuLinearProgrammingProblem,
    solver_state::CuPdhgSolverState,
    buffer_state::CuBufferState,
)
    compute_next_primal_solution!(
        problem,
        solver_state.current_primal_solution,
        solver_state.current_dual_product,
        step_size,
        solver_state.primal_weight,
        buffer_state.delta_primal,
        buffer_state.delta_primal_product,
    )

    compute_next_dual_solution!(
        problem,
        solver_state.current_dual_solution,
        step_size,
        solver_state.primal_weight,
        buffer_state.delta_primal_product,
        solver_state.current_primal_product,
        buffer_state.delta_dual,
    )

    solver_state.cumulative_kkt_passes += 1

    update_solution_in_solver_state!(
        problem,
        solver_state, 
        buffer_state,
    )
end


function optimize(
    params::PdhgParameters,
    original_problem::QuadraticProgrammingProblem,
)
    validate(original_problem)
    qp_cache = cached_quadratic_program_info(original_problem)

    start_rescaling_time = time()
    scaled_problem = rescale_problem(
        params.l_inf_ruiz_iterations,
        params.l2_norm_rescaling,
        params.pock_chambolle_alpha,
        params.verbosity,
        original_problem,
    )
    rescaling_time = time() - start_rescaling_time
    Printf.@printf(
        "preconditioning time (seconds): %.2e\n",
        rescaling_time,
    )

    primal_size = length(scaled_problem.scaled_qp.variable_lower_bound)
    dual_size = length(scaled_problem.scaled_qp.right_hand_side)
    num_eq = scaled_problem.scaled_qp.num_equalities
    if params.primal_importance <= 0 || !isfinite(params.primal_importance)
        error("primal_importance must be positive and finite")
    end

    # transfer from cpu to gpu
    d_scaled_problem = scaledqp_cpu_to_gpu(scaled_problem)
    d_problem = d_scaled_problem.scaled_qp
    buffer_lp = qp_cpu_to_gpu(original_problem)


    # initialization
    solver_state = CuPdhgSolverState(
        CUDA.zeros(Float64, primal_size),    # current_primal_solution
        CUDA.zeros(Float64, dual_size),      # current_dual_solution
        CUDA.zeros(Float64, dual_size),      # current_primal_product
        CUDA.zeros(Float64, primal_size),    # current_dual_product
        CUDA.zeros(Float64, primal_size),    # initial_primal_solution
        CUDA.zeros(Float64, dual_size),      # initial_dual_solution
        CUDA.zeros(Float64, dual_size),      # initial_primal_product
        CUDA.zeros(Float64, primal_size),    # initial_dual_product
        0,                   # inner_count
        0.0,                 # step_size
        1.0,                 # primal_weight
        false,               # numerical_error
        0.0,                 # cumulative_kkt_passes
        0,                   # total_number_iterations
        nothing,
        nothing,
    )

    buffer_state = CuBufferState(
        CUDA.zeros(Float64, primal_size),      # delta_primal
        CUDA.zeros(Float64, dual_size),        # delta_dual
        CUDA.zeros(Float64, dual_size),        # delta_primal_product
    )

    buffer_original = BufferOriginalSol(
        CUDA.zeros(Float64, primal_size),      # primal
        CUDA.zeros(Float64, dual_size),        # dual
        CUDA.zeros(Float64, dual_size),        # primal_product
        CUDA.zeros(Float64, primal_size),      # primal_gradient
    )

    buffer_primal_gradient = CUDA.zeros(Float64, primal_size)

    buffer_kkt = BufferKKTState(
        buffer_original.original_primal_solution,      # primal
        buffer_original.original_dual_solution,        # dual
        buffer_original.original_primal_product,        # primal_product
        buffer_original.original_primal_gradient,      # primal_gradient
        CUDA.zeros(Float64, primal_size),      # lower_variable_violation
        CUDA.zeros(Float64, primal_size),      # upper_variable_violation
        CUDA.zeros(Float64, dual_size),        # constraint_violation
        CUDA.zeros(Float64, primal_size),      # dual_objective_contribution_array
        CUDA.zeros(Float64, primal_size),      # reduced_costs_violations
        CuDualStats(
            0.0,
            CUDA.zeros(Float64, dual_size - num_eq),
            CUDA.zeros(Float64, primal_size),
        ),
        0.0,                                   # dual_res_inf
    )
    
    buffer_kkt_infeas = BufferKKTState(
        buffer_original.original_primal_solution,      # primal
        buffer_original.original_dual_solution,        # dual
        buffer_original.original_primal_product,        # primal_product
        buffer_original.original_primal_gradient,      # primal_gradient
        CUDA.zeros(Float64, primal_size),      # lower_variable_violation
        CUDA.zeros(Float64, primal_size),      # upper_variable_violation
        CUDA.zeros(Float64, dual_size),        # constraint_violation
        CUDA.zeros(Float64, primal_size),      # dual_objective_contribution_array
        CUDA.zeros(Float64, primal_size),      # reduced_costs_violations
        CuDualStats(
            0.0,
            CUDA.zeros(Float64, dual_size - num_eq),
            CUDA.zeros(Float64, primal_size),
        ),
        0.0,                                   # dual_res_inf
    )

    # stepsize
    if params.step_size_policy_params isa AdaptiveStepsizeParams
        solver_state.cumulative_kkt_passes += 0.5
        solver_state.step_size = 1.0 / norm(scaled_problem.scaled_qp.constraint_matrix, Inf)
    else
        desired_relative_error = 0.2
        maximum_singular_value, number_of_power_iterations =
            estimate_maximum_singular_value(
                scaled_problem.scaled_qp.constraint_matrix,
                probability_of_failure = 0.001,
                desired_relative_error = desired_relative_error,
            )
        solver_state.step_size =
            (1 - desired_relative_error) / maximum_singular_value
        solver_state.cumulative_kkt_passes += number_of_power_iterations
    end

    KKT_PASSES_PER_TERMINATION_EVALUATION = 2.0

    if params.scale_invariant_initial_primal_weight
        solver_state.primal_weight = select_initial_primal_weight(
            d_problem,
            1.0,
            1.0,
            params.primal_importance,
            params.verbosity,
        )
    else
        solver_state.primal_weight = params.primal_importance
    end

    primal_weight_update_smoothing = params.restart_params.primal_weight_update_smoothing 

    iteration_stats = IterationStats[]
    start_time = time()
    time_spent_doing_basic_algorithm = 0.0

    last_restart_info = create_last_restart_info(
        d_problem,
        solver_state.current_primal_solution,
        solver_state.current_dual_solution,
        solver_state.current_primal_product,
    )

    # For termination criteria:
    termination_criteria = params.termination_criteria
    iteration_limit = termination_criteria.iteration_limit
    termination_evaluation_frequency = params.termination_evaluation_frequency

    # This flag represents whether a numerical error occurred during the algorithm
    # if it is set to true it will trigger the algorithm to terminate.
    solver_state.numerical_error = false
    display_iteration_stats_heading(params.verbosity)

    iteration = 0
    while true
        iteration += 1

        if mod(iteration - 1, termination_evaluation_frequency) == 0 ||
            iteration == iteration_limit + 1 ||
            iteration <= 10 ||
            solver_state.numerical_error
            
            solver_state.cumulative_kkt_passes += KKT_PASSES_PER_TERMINATION_EVALUATION

            buffer_primal_gradient .= d_problem.objective_vector .- solver_state.current_dual_product

            ### KKT ###
            current_iteration_stats = evaluate_unscaled_iteration_stats(
                d_scaled_problem,
                qp_cache,
                params.termination_criteria,
                params.record_iteration_stats,
                solver_state.current_primal_solution,
                solver_state.current_dual_solution,
                iteration,
                time() - start_time,
                solver_state.cumulative_kkt_passes,
                termination_criteria.eps_optimal_absolute,
                termination_criteria.eps_optimal_relative,
                solver_state.step_size,
                solver_state.primal_weight,
                POINT_TYPE_UNSPECIFIED, 
                solver_state.current_primal_product,
                buffer_primal_gradient,
                buffer_original,
                buffer_kkt,
                buffer_kkt_infeas,
                buffer_lp,
            )
            method_specific_stats = current_iteration_stats.method_specific_stats
            method_specific_stats["time_spent_doing_basic_algorithm"] =
                time_spent_doing_basic_algorithm

            primal_norm_params, dual_norm_params = define_norms(
                solver_state.step_size,
                solver_state.primal_weight,
            )

            ### check termination criteria ###
            termination_reason = check_termination_criteria(
                termination_criteria,
                qp_cache,
                current_iteration_stats,
            )
            if solver_state.numerical_error && termination_reason == false
                termination_reason = TERMINATION_REASON_NUMERICAL_ERROR
            end

            # If we're terminating, record the iteration stats to provide final
            # solution stats.
            if params.record_iteration_stats || termination_reason != false
                push!(iteration_stats, current_iteration_stats)
            end

            # Print table.
            if print_to_screen_this_iteration(
                termination_reason,
                iteration,
                params.verbosity,
                termination_evaluation_frequency,
            )
                display_iteration_stats(current_iteration_stats, params.verbosity)
            end

            if termination_reason != false
                primal_solution = zeros(primal_size)
                dual_solution = zeros(dual_size)
                gpu_to_cpu!(
                    solver_state.current_primal_solution,
                    solver_state.current_dual_solution,
                    primal_solution,
                    dual_solution,
                )

                pdhg_final_log(
                    scaled_problem.scaled_qp,
                    primal_solution,
                    dual_solution,
                    params.verbosity,
                    iteration,
                    termination_reason,
                    current_iteration_stats,
                )

                return unscaled_saddle_point_output(
                    scaled_problem,
                    primal_solution,
                    dual_solution,
                    termination_reason,
                    iteration - 1,
                    iteration_stats,
                )
            end

            do_restart = run_restart_scheme(
                d_problem,
                solver_state,
                buffer_state,
                last_restart_info,
                iteration - 1,
                params.verbosity,
                params.restart_params,
            )
            
            if do_restart
                solver_state.primal_weight = compute_new_primal_weight(
                    last_restart_info,
                    solver_state.primal_weight,
                    primal_weight_update_smoothing,
                    params.verbosity,
                )
                solver_state.ratio_step_sizes = 1.0
            end
        end

        time_spent_doing_basic_algorithm_checkpoint = time()
      
        ### no dual_objective
        if params.verbosity >= 6 && print_to_screen_this_iteration(
            false, # termination_reason
            iteration,
            params.verbosity,
            termination_evaluation_frequency,
        )
            pdhg_specific_log(
                # problem,
                iteration,
                solver_state.current_primal_solution,
                solver_state.current_dual_solution,
                solver_state.step_size,
                solver_state.required_ratio,
                solver_state.primal_weight,
            )
          end

        take_step!(params.step_size_policy_params, d_problem, solver_state, buffer_state)

        time_spent_doing_basic_algorithm += time() - time_spent_doing_basic_algorithm_checkpoint
    end
end