import ArgParse
import GZip
import JSON3

import cuHPDLP

function write_vector_to_file(filename, vector)
    open(filename, "w") do io
      for x in vector
        println(io, x)
      end
    end
end

function solve_instance_and_output(
    parameters::cuHPDLP.PdhgParameters,
    output_dir::String,
    instance_path::String,
)
    if !isdir(output_dir)
        mkpath(output_dir)
    end
  
    instance_name = replace(basename(instance_path), r"\.(mps|MPS|qps|QPS)(\.gz)?$" => "")
  
    function inner_solve()
        lower_file_name = lowercase(basename(instance_path))
        if endswith(lower_file_name, ".mps") ||
            endswith(lower_file_name, ".mps.gz") ||
            endswith(lower_file_name, ".qps") ||
            endswith(lower_file_name, ".qps.gz")
            lp = cuHPDLP.qps_reader_to_standard_form(instance_path)
        else
            error(
                "Instance has unrecognized file extension: ", 
                basename(instance_path),
            )
        end
    
        if parameters.verbosity >= 1
            println("Instance: ", instance_name)
        end

        output::cuHPDLP.SaddlePointOutput = cuHPDLP.optimize(parameters, lp)
    
        log = cuHPDLP.SolveLog()
        log.instance_name = instance_name
        log.command_line_invocation = join([PROGRAM_FILE; ARGS...], " ")
        log.termination_reason = output.termination_reason
        log.termination_string = output.termination_string
        log.iteration_count = output.iteration_count
        log.solve_time_sec = output.iteration_stats[end].cumulative_time_sec
        log.solution_stats = output.iteration_stats[end]
        log.solution_type = cuHPDLP.POINT_TYPE_UNSPECIFIED
    
        summary_output_path = joinpath(output_dir, instance_name * "_summary.json")
        open(summary_output_path, "w") do io
            write(io, JSON3.write(log, allow_inf = true))
        end
    
        log.iteration_stats = output.iteration_stats
        full_log_output_path =
            joinpath(output_dir, instance_name * "_full_log.json.gz")
        GZip.open(full_log_output_path, "w") do io
            write(io, JSON3.write(log, allow_inf = true))
        end
    
        primal_output_path = joinpath(output_dir, instance_name * "_primal.txt")
        write_vector_to_file(primal_output_path, output.primal_solution)
    
        dual_output_path = joinpath(output_dir, instance_name * "_dual.txt")
        write_vector_to_file(dual_output_path, output.dual_solution)
    end     

    inner_solve()
   
    return
end

function warm_up(lp::cuHPDLP.QuadraticProgrammingProblem)
    restart_params = cuHPDLP.construct_restart_parameters(
        cuHPDLP.ADAPTIVE,       # NO_RESTARTS FIXED_FREQUENCY ADAPTIVE
        1000,                   # restart_frequency_if_fixed
        0.36,                   # artificial_restart_threshold
        0.2,                    # sufficient_reduction_for_restart
        0.8,                    # necessary_reduction_for_restart
        0.5,                    # primal_weight_update_smoothing
    )

    termination_params_warmup = cuHPDLP.construct_termination_criteria(
        # optimality_norm = L2,
        eps_optimal_absolute = 1.0e-4,
        eps_optimal_relative = 1.0e-4,
        eps_primal_infeasible = 1.0e-8,
        eps_dual_infeasible = 1.0e-8,
        time_sec_limit = Inf,
        iteration_limit = 100,
        kkt_matrix_pass_limit = Inf,
    )

    params_warmup = cuHPDLP.PdhgParameters(
        10,
        false,
        1.0,
        1.0,
        true,
        0,
        true,
        64,
        termination_params_warmup,
        restart_params,
        cuHPDLP.AdaptiveStepsizeParams(0.3,0.6),
    )

    cuHPDLP.optimize(params_warmup, lp);
end


function parse_command_line()
    arg_parse = ArgParse.ArgParseSettings()

    ArgParse.@add_arg_table! arg_parse begin
        "--instance_path"
        help = "The path to the instance to solve in .mps.gz or .mps format."
        arg_type = String
        required = true

        "--output_directory"
        help = "The directory for output files."
        arg_type = String
        required = true

        "--tolerance"
        help = "KKT tolerance of the solution."
        arg_type = Float64
        default = 1e-4

        "--time_sec_limit"
        help = "Time limit."
        arg_type = Float64
        default = 3600.0
    end

    return ArgParse.parse_args(arg_parse)
end


function main()
    parsed_args = parse_command_line()
    instance_path = parsed_args["instance_path"]
    tolerance = parsed_args["tolerance"]
    time_sec_limit = parsed_args["time_sec_limit"]
    output_directory = parsed_args["output_directory"]

    lp = cuHPDLP.qps_reader_to_standard_form(instance_path)

    oldstd = stdout
    redirect_stdout(devnull)
    warm_up(lp);
    redirect_stdout(oldstd)

    restart_params = cuHPDLP.construct_restart_parameters(
        cuHPDLP.ADAPTIVE,       # NO_RESTARTS FIXED_FREQUENCY ADAPTIVE
        1000,                   # restart_frequency_if_fixed
        0.36,                   # artificial_restart_threshold
        0.2,                    # sufficient_reduction_for_restart
        0.8,                    # necessary_reduction_for_restart
        0.5,                    # primal_weight_update_smoothing
    )

    termination_params = cuHPDLP.construct_termination_criteria(
        # optimality_norm = L2,
        eps_optimal_absolute = tolerance,
        eps_optimal_relative = tolerance,
        eps_primal_infeasible = 1.0e-8,
        eps_dual_infeasible = 1.0e-8,
        time_sec_limit = time_sec_limit,
        iteration_limit = typemax(Int32),
        kkt_matrix_pass_limit = Inf,
    )

    params = cuHPDLP.PdhgParameters(
        10,
        false,
        1.0,
        1.0,
        true,
        2,
        true,
        64,
        termination_params,
        restart_params,
        cuHPDLP.AdaptiveStepsizeParams(0.3,0.6),  
    )

    solve_instance_and_output(
        params,
        output_directory,
        instance_path,
    )

end

main()