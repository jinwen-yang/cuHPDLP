struct AdaptiveStepsizeParams
    reduction_exponent::Float64
    growth_exponent::Float64
end

struct ConstantStepsizeParams end

@enum RestartScheme NO_RESTARTS FIXED_FREQUENCY ADAPTIVE

mutable struct RestartParameters
    """
    Specifies what type of restart scheme is used.
    """
    restart_scheme::RestartScheme
    """
    If `restart_scheme` = `FIXED_FREQUENCY` then this number determines the frequency that the algorithm is restarted.
    """
    restart_frequency_if_fixed::Int64
    """
    If in the past `artificial_restart_threshold` fraction of iterations no restart has occurred then a restart will be artificially triggered. The value should be between zero and one. Smaller values will have more frequent artificial restarts than larger values.
    """
    artificial_restart_threshold::Float64
    """
    Only applies when `restart_scheme` = `ADAPTIVE`. It is the threshold improvement in the quality of the current/average iterate compared with that  of the last restart that will trigger a restart. The value of this parameter should be between zero and one. Smaller values make restarts less frequent, larger values make restarts more frequent.
    """
    sufficient_reduction_for_restart::Float64
    """
    Only applies when `restart_scheme` = `ADAPTIVE`. It is the threshold
    improvement in the quality of the current/average iterate compared with that of the last restart that is neccessary for a restart to be triggered. If this thrshold is met and the quality of the iterates appear to be getting worse then a restart is triggered. The value of this parameter should be between zero and one, and greater than sufficient_reduction_for_restart. Smaller values make restarts less frequent, larger values make restarts more frequent.
    """
    necessary_reduction_for_restart::Float64
    """
    Controls the exponential smoothing of log(primal_weight) when the primal weight is updated (i.e., on every restart). Must be between 0.0 and 1.0 inclusive. At 0.0 the primal weight remains frozen at its initial value.
    """
    primal_weight_update_smoothing::Float64
end

struct PdhgParameters
    l_inf_ruiz_iterations::Int
    l2_norm_rescaling::Bool
    pock_chambolle_alpha::Union{Float64,Nothing}
    primal_importance::Float64
    scale_invariant_initial_primal_weight::Bool
    verbosity::Int64
    record_iteration_stats::Bool
    termination_evaluation_frequency::Int32
    termination_criteria::TerminationCriteria
    restart_params::RestartParameters
    step_size_policy_params::Union{
        AdaptiveStepsizeParams,
        ConstantStepsizeParams,
    }
end

mutable struct CuPdhgSolverState
    current_primal_solution::CuVector{Float64}
    current_dual_solution::CuVector{Float64}
    current_primal_product::CuVector{Float64}
    current_dual_product::CuVector{Float64}
    initial_primal_solution::CuVector{Float64}
    initial_dual_solution::CuVector{Float64}
    initial_primal_product::CuVector{Float64}
    initial_dual_product::CuVector{Float64}
    inner_count::Int64
    step_size::Float64
    primal_weight::Float64
    numerical_error::Bool
    cumulative_kkt_passes::Float64
    total_number_iterations::Int64
    required_ratio::Union{Float64,Nothing}
    ratio_step_sizes::Union{Float64,Nothing}
end


mutable struct CuBufferState
    delta_primal::CuVector{Float64}
    delta_dual::CuVector{Float64}
    delta_primal_product::CuVector{Float64}
end


function define_norms(
    primal_size::Int64,
    dual_size::Int64,
    step_size::Float64,
    primal_weight::Float64,
)
    return 1 / step_size * primal_weight, 1 / step_size / primal_weight
end



struct SaddlePointOutput
    primal_solution::Vector{Float64}
    dual_solution::Vector{Float64}
    termination_reason::TerminationReason
    termination_string::String
    iteration_count::Int32
    iteration_stats::Vector{IterationStats}
end

function unscaled_saddle_point_output(
    scaled_problem::ScaledQpProblem,
    primal_solution::AbstractVector{Float64},
    dual_solution::AbstractVector{Float64},
    termination_reason::TerminationReason,
    iterations_completed::Int64,
    iteration_stats::Vector{IterationStats},
)
    original_primal_solution =
        primal_solution ./ scaled_problem.variable_rescaling
    original_dual_solution = dual_solution ./ scaled_problem.constraint_rescaling
  
    return SaddlePointOutput(
        original_primal_solution,
        original_dual_solution,
        termination_reason,
        termination_reason_to_string(termination_reason),
        iterations_completed,
        iteration_stats,
    )
end


function weighted_norm(
    vec::CuVector{Float64},
    weights::Float64,
)
    tmp = CUDA.norm(vec)
    return sqrt(weights) * tmp
end

function define_norms(
    step_size::Float64,
    primal_weight::Float64,
)
    return 1 / step_size * primal_weight, 1 / step_size / primal_weight
end

function compute_fixed_point_residual(
    primal_diff::CuVector{Float64},
    dual_diff::CuVector{Float64},
    primal_diff_product::CuVector{Float64},
    primal_norm_params::Float64,
    dual_norm_params::Float64,
)
    primal_dual_interaction = CUDA.dot(primal_diff_product, dual_diff) 
    interaction = abs(primal_dual_interaction) 

    norm_delta_primal = weighted_norm(primal_diff, primal_norm_params)
    norm_delta_dual = weighted_norm(dual_diff, dual_norm_params)

    movement = 0.5 * norm_delta_primal^2 + 0.5 * norm_delta_dual^2

    return movement + interaction
end

function construct_restart_parameters(
    restart_scheme::RestartScheme,
    restart_frequency_if_fixed::Int64,
    artificial_restart_threshold::Float64,
    sufficient_reduction_for_restart::Float64,
    necessary_reduction_for_restart::Float64,
    primal_weight_update_smoothing::Float64,
)
    @assert restart_frequency_if_fixed > 1
    @assert 0.0 < artificial_restart_threshold <= 1.0
    @assert 0.0 <
            sufficient_reduction_for_restart <=
            necessary_reduction_for_restart <=
            1.0
    @assert 0.0 <= primal_weight_update_smoothing <= 1.0
  
    return RestartParameters(
        restart_scheme,
        restart_frequency_if_fixed,
        artificial_restart_threshold,
        sufficient_reduction_for_restart,
        necessary_reduction_for_restart,
        primal_weight_update_smoothing,
    )
end

mutable struct CuRestartInfo
    primal_solution::CuVector{Float64}
    dual_solution::CuVector{Float64}
    primal_diff::CuVector{Float64}
    dual_diff::CuVector{Float64}
    primal_diff_product::CuVector{Float64}
    primal_distance_moved_last_restart_period::Float64
    dual_distance_moved_last_restart_period::Float64
    reduction_ratio_last_trial::Float64
end

function create_last_restart_info(
    problem::CuLinearProgrammingProblem,
    primal_solution::CuVector{Float64},
    dual_solution::CuVector{Float64},
    primal_product::CuVector{Float64},
)
    dual_size, primal_size = size(problem.constraint_matrix)
    return CuRestartInfo(
        copy(primal_solution),
        copy(dual_solution),
        CUDA.zeros(Float64, primal_size),      # delta_primal
        CUDA.zeros(Float64, dual_size),        # delta_dual
        CUDA.zeros(Float64, dual_size),        # delta_primal_product
        0.0,
        0.0,
        1.0,
    )
end

function should_do_adaptive_restart(
    problem::CuLinearProgrammingProblem, 
    restart_params::RestartParameters,
    solver_state::CuPdhgSolverState,
    buffer_state::CuBufferState,
    last_restart_info::CuRestartInfo,
)
    primal_norm_params, dual_norm_params = define_norms(solver_state.step_size, solver_state.primal_weight)

    last_restart_fixed_point_residual = compute_fixed_point_residual(last_restart_info.primal_diff, last_restart_info.dual_diff, last_restart_info.primal_diff_product, primal_norm_params, dual_norm_params)

    current_fixed_point_residual = compute_fixed_point_residual(buffer_state.delta_primal, buffer_state.delta_dual, buffer_state.delta_primal_product, primal_norm_params, dual_norm_params)

    do_restart = false
   
    reduction_ratio = current_fixed_point_residual / last_restart_fixed_point_residual

    if reduction_ratio < restart_params.necessary_reduction_for_restart
        if reduction_ratio < restart_params.sufficient_reduction_for_restart
            do_restart = true
        elseif reduction_ratio > last_restart_info.reduction_ratio_last_trial
            do_restart = true
        end
    end
    last_restart_info.reduction_ratio_last_trial = reduction_ratio
  
    return do_restart
end


function run_restart_scheme(
    problem::CuLinearProgrammingProblem,
    solver_state::CuPdhgSolverState,
    buffer_state::CuBufferState,
    last_restart_info::CuRestartInfo,
    iterations_completed::Int64,
    verbosity::Int64,
    restart_params::RestartParameters,
)
    if solver_state.inner_count == 0
        return false
    end

    restart_length = solver_state.inner_count
    artificial_restart = false
    do_restart = false
    
    if restart_length >= restart_params.artificial_restart_threshold * iterations_completed
        do_restart = true
        artificial_restart = true
    end

    if !do_restart
        if restart_params.restart_scheme == ADAPTIVE
            do_restart = should_do_adaptive_restart(
                problem,
                restart_params,
                solver_state,
                buffer_state,
                last_restart_info,
            )
        elseif restart_params.restart_scheme == FIXED_FREQUENCY &&
            restart_params.restart_frequency_if_fixed <= restart_length
            do_restart = true
        end
    end

    if do_restart
        if verbosity >= 4
            print("  Restart")
            print(" after ", rpad(restart_length, 4), " iterations")
            if artificial_restart
                println("*")
            else
                println("")
            end
        end
        reset_solver_state!(solver_state, buffer_state, problem)
        update_last_restart_info!(last_restart_info, solver_state, buffer_state)
    end
    return do_restart
end


function reset_solver_state!(
    solver_state::CuPdhgSolverState,
    buffer_state::CuBufferState,
    problem::CuLinearProgrammingProblem,
)
    solver_state.initial_primal_solution .= solver_state.current_primal_solution .+ buffer_state.delta_primal
    solver_state.initial_dual_solution .= solver_state.current_dual_solution .+ buffer_state.delta_dual
    solver_state.initial_primal_product .= solver_state.current_primal_product .+ buffer_state.delta_primal_product
    CUDA.CUSPARSE.mv!('N', 1, problem.constraint_matrix_t, solver_state.initial_dual_solution, 0, solver_state.initial_dual_product, 'O', CUDA.CUSPARSE.CUSPARSE_SPMV_CSR_ALG2) 
    solver_state.inner_count = 0

    solver_state.current_primal_solution .= solver_state.initial_primal_solution
    solver_state.current_primal_product .= solver_state.initial_primal_product
    solver_state.current_dual_solution .= solver_state.initial_dual_solution
    solver_state.current_dual_product .= solver_state.initial_dual_product
    return
end

function update_last_restart_info!(
    last_restart_info::CuRestartInfo,
    solver_state::CuPdhgSolverState,
    buffer_state::CuBufferState,
)
    primal_norm_params, dual_norm_params = define_norms(solver_state.step_size, solver_state.primal_weight)

    last_restart_info.primal_distance_moved_last_restart_period =
        weighted_norm(
            solver_state.initial_primal_solution - last_restart_info.primal_solution,
            primal_norm_params,
        ) / sqrt(solver_state.primal_weight)
    last_restart_info.dual_distance_moved_last_restart_period =
        weighted_norm(
            solver_state.initial_dual_solution - last_restart_info.dual_solution,
            dual_norm_params,
        ) * sqrt(solver_state.primal_weight)

    last_restart_info.primal_diff .= buffer_state.delta_primal
    last_restart_info.dual_diff .= buffer_state.delta_dual
    last_restart_info.primal_diff_product .= buffer_state.delta_primal_product

    last_restart_info.primal_solution .= solver_state.initial_primal_solution
    last_restart_info.dual_solution .= solver_state.initial_dual_solution
end


function compute_new_primal_weight(
    last_restart_info::CuRestartInfo,
    primal_weight::Float64,
    primal_weight_update_smoothing::Float64,
    verbosity::Int64,
)
    primal_distance = last_restart_info.primal_distance_moved_last_restart_period
    dual_distance = last_restart_info.dual_distance_moved_last_restart_period
    
    if primal_distance > eps() && dual_distance > eps()
        new_primal_weight_estimate = dual_distance / primal_distance
        log_primal_weight =
            primal_weight_update_smoothing * log(new_primal_weight_estimate) +
            (1 - primal_weight_update_smoothing) * log(primal_weight)

        primal_weight = exp(log_primal_weight)
        if verbosity >= 4
            Printf.@printf "  New computed primal weight is %.2e\n" primal_weight
        end

        return primal_weight
    else
        return primal_weight
    end
end

#################################################
"""
A simple string name for a PointType.
"""
function point_type_label(point_type::PointType)
    if point_type == POINT_TYPE_CURRENT_ITERATE
        return "current"
    elseif point_type == POINT_TYPE_AVERAGE_ITERATE
        return "average"
    elseif point_type == POINT_TYPE_ITERATE_DIFFERENCE
        return "difference"
    else
        return "unknown PointType"
    end
end

"""
Logging for when the algorithm terminates.
"""
function generic_final_log(
    problem::QuadraticProgrammingProblem,
    current_primal_solution::Vector{Float64},
    current_dual_solution::Vector{Float64},
    last_iteration_stats::IterationStats,
    verbosity::Int64,
    iteration::Int64,
    termination_reason::TerminationReason,
)
    if verbosity >= 2 && verbosity <=3
        Printf.@printf(
            "total time (seconds): %.2e\n",
            last_iteration_stats.cumulative_time_sec,
        )
    end

    if verbosity >= 1
        print("Terminated after $iteration iterations: ")
        println(termination_reason_to_string(termination_reason))
    end

    method_specific_stats = last_iteration_stats.method_specific_stats
    if verbosity >= 3
        for convergence_information in last_iteration_stats.convergence_information
            Printf.@printf(
                "Primal objective: %f, ",
                convergence_information.primal_objective
            )
            Printf.@printf(
                "dual objective: %f, ",
                convergence_information.dual_objective
            )
            Printf.@printf(
                "corrected dual objective: %f \n",
                convergence_information.corrected_dual_objective
            )
        end
    end

    if verbosity >= 7
        for convergence_information in last_iteration_stats.convergence_information
            print_infinity_norms(convergence_information)
        end
        print_variable_and_constraint_hardness(
            problem,
            current_primal_solution,
            current_dual_solution,
        )
    end
end


function select_initial_primal_weight(
    problem::CuLinearProgrammingProblem,
    primal_norm_params::Float64,
    dual_norm_params::Float64,
    primal_importance::Float64,
    verbosity::Int64,
)
    rhs_vec_norm = weighted_norm(problem.right_hand_side, dual_norm_params)
    obj_vec_norm = weighted_norm(problem.objective_vector, primal_norm_params)
    if obj_vec_norm > 0.0 && rhs_vec_norm > 0.0
        primal_weight = primal_importance * (obj_vec_norm / rhs_vec_norm)
    else
        primal_weight = primal_importance
    end
    if verbosity >= 6
        println("Initial primal weight = $primal_weight")
    end
    return primal_weight
end

