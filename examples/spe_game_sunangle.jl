"""
Solve a SPE trajectory game using mixed complementarity programming.

This code is heavily adapted from ASE 389: Game-Theoretic Modeling for Multi-Agent Systems Assignment 5.

Authors: Kyle Shiller & Pete Lealiiee
"""


""" Utilities for constructing trajectory games, in which each player wishes to # 2B[1]
solve a problem of the form:
                min_{τᵢ}   fᵢ(τ, θ)

where all vehicles must jointly satisfy the constraints
                           g̃(τ, θ) = 0
                           h̃(τ, θ) ≥ 0.

Here, τᵢ is the ith vehicle's trajectory, consisting of states and controls. The shared
constraints g̃ and h̃ incorporate dynamic feasibility, fixed initial condition, actuator and
state limits, environment boundaries, and collision-avoidance.
"""

using SPEsim
using LazySets: LazySets
using TrajectoryGamesBase:
    TrajectoryGamesBase,
    PolygonEnvironment,
    ProductDynamics,
    TimeSeparableTrajectoryGameCost,
    TrajectoryGame,
    GeneralSumCostStructure,
    num_players,
    time_invariant_linear_dynamics,
    unstack_trajectory,
    stack_trajectories,
    state_dim,
    control_dim,
    state_bounds,
    control_bounds,
    OpenLoopStrategy,
    JointStrategy,
    RecedingHorizonStrategy,
    rollout
using TrajectoryGamesExamples: planar_double_integrator, animate_sim_steps
using BlockArrays: mortar, blocks, BlockArray
using GLMakie: GLMakie
using Makie: Makie
using PATHSolver: PATHSolver
using LinearAlgebra: norm_sqr, norm, dot, I
using ProgressMeter: ProgressMeter
using CSV: write
using Plots: plot, plot!, savefig

"Utility to set up a (two player) SPE game."
function setup_trajectory_game(; environment = PolygonEnvironment(4, 500))
    # PolygonEnvironment(num_sides, radius)
    #NOTE: The environment here inherently imposes state constraints and is used for visualization.

    #: Define time step for simulation
    dt = 5 # s (time step) ##### INPUT #####

    cost = let
        function stage_cost(x, u, t, θ)
            x1, x2  = blocks(x)
            u1, u2  = blocks(u)

            #: Extract game state vector (x_p - x_e)
            x_diff  = x1 - x2

            #: Define cost structure
            Q           = Matrix(1I, length(x1), length(x1)) ##### INPUT #####
            Q[1:3,1:3]  = Matrix(1I, 3, 3) * 100 ##### INPUT #####
            R           = Matrix(1I, length(u1), length(u1)) * 1e7 ##### INPUT #####
            gam0_sq     = 2
            cost        = (x_diff' * Q * x_diff + u1' * R * u1 - gam0_sq * u2' * R * u2) / 2

            #: Virtual reference spacecraft orbital parameters
            SMA             = 15000 # km (semi-major axis)
            arg_latitude    = pi/2 # rad (90 deg, argument of latitude)
            RAAN            = pi/4 # rad (45 deg, right ascension of ascending node)
            inc             = pi/12 # rad (15 deg, inclination)

            #: Define sun vector
            sun_vec_I       = [1/sqrt(10), 3/sqrt(10), 0] # sun vector in inertial frame
            R3_RAAN         = [cos(RAAN) -sin(RAAN) 0; sin(RAAN) cos(RAAN) 0; 0 0 1]
            R1_inc          = [1 0 0; 0 cos(inc) -sin(inc); 0 sin(inc) cos(inc)]
            R3_arg_lat      = [cos(arg_latitude) -sin(arg_latitude) 0; sin(arg_latitude) cos(arg_latitude) 0; 0 0 1]
            sun_vec_L       = R3_arg_lat * R1_inc * R3_RAAN * sun_vec_I # sun vector in LVLH frame

            #: Define sun angle cost
            gam1            = 10000 ##### INPUT #####
            sun_angle_cost  = gam1 * dot(-x_diff[1:3], sun_vec_L) #/ norm(x_diff[1:3])
            cost            += sun_angle_cost

            #: P1 (pursuer) wants to minimize cost, and P2 (evader) wants to maximize cost.
            [
                cost,   # pursuer
                -cost,  # evader
            ]

        end

        function reducer(stage_costs)
            reduce(.+, stage_costs) ./ length(stage_costs)
        end

        TimeSeparableTrajectoryGameCost(stage_cost, reducer, GeneralSumCostStructure(), 1.0)
    end

    function coupling_constraints(xs, us, θ)
        mapreduce(vcat, xs) do x
            x1, x2 = blocks(x)

            # Players need to stay at least 1 m away from one another.
            norm_sqr(x1[1:2] - x2[1:2]) - 0 #by changing to 0, this constraint is effectively removed (and we allow rendezvous)
        end
    end

    #: Define continuous HCW dynamics
    SMA             = 15000 # km (virtual reference semi-major axis)
    mu              = 3.986e5 # km^3/s^2 (gravitational parameter)
    n               = sqrt(mu / SMA^3) # rad/s (virtual reference mean motion)
    # mass            = 1 # kg (mass of spacecraft, assumed constant/same for both sats)
    A_cts           = zeros(6, 6)
    A_cts[1:3, 4:6] = Matrix(1I, 3, 3)
    A_cts[4,1]      = 3*n^2
    A_cts[4,5]      = 2*n
    A_cts[5,4]      = -2*n
    A_cts[6,3]      = -n^2
    B_cts           = zeros(6, 3)
    B_cts[4:6,1:3]  = Matrix(1I, 3, 3) #* 1/mass

    #: Discretize HCW dynamics
    A_discrete      = Matrix(1I, 6, 6) + A_cts * dt
    B_discrete      = B_cts * dt

    #: Define state/input limits for each player ##### INPUT #####
    p_pos_lim   = Inf                # m
    e_pos_lim   = 1000 #Inf                # m
    # p_ctrl_lim  = 0.01                  # m/s^2
    # e_ctrl_lim  = 0.5*p_ctrl_lim        # m/s^2
    p_ctrl_lim  = 10                    # m/s^2
    e_ctrl_lim  = 0.9*p_ctrl_lim        # m/s^2
    p_vel_bound = 200                   # m/s
    e_vel_bound = 0.9*p_vel_bound       # m/s

    #: Define dynamics for each player
    p_dynamics = time_invariant_linear_dynamics(; A=A_discrete, B=B_discrete,
                            state_bounds = (; lb = [-p_pos_lim, -p_pos_lim, -p_pos_lim, -p_vel_bound, -p_vel_bound, -p_vel_bound],
                                              ub = [p_pos_lim, p_pos_lim, p_pos_lim, p_vel_bound, p_vel_bound, p_vel_bound]
                                              ),
                            control_bounds = (; lb = -[p_ctrl_lim, p_ctrl_lim, p_ctrl_lim], ub = [p_ctrl_lim, p_ctrl_lim, p_ctrl_lim])
                            )

    e_dynamics = time_invariant_linear_dynamics(; A=A_discrete, B=B_discrete,
                            state_bounds = (; lb = [-e_pos_lim, -e_pos_lim, -e_pos_lim, -e_vel_bound, -e_vel_bound, -e_vel_bound],
                                              ub = [e_pos_lim, e_pos_lim, e_pos_lim, e_vel_bound, e_vel_bound, e_vel_bound]),
                            control_bounds = (; lb = -[e_ctrl_lim, e_ctrl_lim, e_ctrl_lim], ub = [e_ctrl_lim, e_ctrl_lim, e_ctrl_lim])
                            )

    dynamics = ProductDynamics([p_dynamics, e_dynamics])

    TrajectoryGame(dynamics, cost, environment, coupling_constraints)
end

"Utility for unpacking trajectory."
function unpack_trajectory(flat_trajectory; dynamics::ProductDynamics)
    trajs = Iterators.map(1:num_players(dynamics), blocks(flat_trajectory)) do ii, τ
        horizon = Int(length(τ) / (state_dim(dynamics, ii) + control_dim(dynamics, ii)))
        num_states = state_dim(dynamics, ii) * horizon
        X = reshape(τ[1:num_states], (state_dim(dynamics, ii), horizon))
        U = reshape(τ[(num_states + 1):end], (control_dim(dynamics, ii), horizon))

        (; xs = eachcol(X) |> collect, us = eachcol(U) |> collect)
    end

    stack_trajectories(trajs)
end

"Utility for packing trajectory."
function pack_trajectory(traj)
    trajs = unstack_trajectory(traj)
    mapreduce(vcat, trajs) do τ
        vcat(reduce(vcat, τ.xs), reduce(vcat, τ.us))
    end
end

"Convert a TrajectoryGame to a ParametricGame."
function build_parametric_game(; game = setup_trajectory_game(), horizon = 10)
    N = 2
    N == num_players(game) || error("Should have only two players.")

    # Construct costs.
    function player_cost(τ, θ, player_index)
        (; xs, us) = unpack_trajectory(τ; game.dynamics)
        ts = Iterators.eachindex(xs)
        Iterators.map(xs, us, ts) do x, u, t
            game.cost.discount_factor^(t - 1) * game.cost.stage_cost(x, u, t, θ)[player_index]
        end |> game.cost.reducer
    end

    fs = [(τ, θ) -> player_cost(τ, θ, ii) for ii in 1:N]

    # Dummy individual constraints.
    gs = [(τ, θ) -> [0] for _ in 1:N]
    hs = [(τ, θ) -> [0] for _ in 1:N]

    # Shared equality constraints.
    g̃ = (τ, θ) -> let
        (; xs, us) = unpack_trajectory(τ; game.dynamics)

        # Force all players to start at the given initial condition.
        g̃1 = xs[1] - θ

        # Dynamics constraints.
        ts = Iterators.eachindex(xs)
        g̃2 = mapreduce(vcat, ts[2:end]) do t
            xs[t] - game.dynamics(xs[t - 1], us[t - 1])
        end

        vcat(g̃1, g̃2)
    end

    # Shared inequality constraints.
    h̃ =
        (τ, θ) -> let
            (; xs, us) = unpack_trajectory(τ; game.dynamics)

            # Collision-avoidance constriant.
            h̃1 = game.coupling_constraints(xs, us, θ)

            # Environment boundaries.
            env_constraints = TrajectoryGamesBase.get_constraints(game.env)
            h̃2 = mapreduce(vcat, xs) do x
                env_constraints(x)
            end

            # Actuator/state limits.
            actuator_constraint = TrajectoryGamesBase.get_constraints_from_box_bounds(
                control_bounds(game.dynamics),
            )
            h̃3 = mapreduce(vcat, us) do u
                actuator_constraint(u)
            end

            state_constraint =
                TrajectoryGamesBase.get_constraints_from_box_bounds(state_bounds(game.dynamics))
            h̃4 = mapreduce(vcat, xs) do x
                state_constraint(x)
            end

            vcat(h̃1, h̃2, h̃3, h̃4)
            #vcat(h̃2, h̃3, h̃4)
        end

    ParametricGame(;
        objectives = fs,
        equality_constraints = gs,
        inequality_constraints = hs,
        shared_equality_constraint = g̃,
        shared_inequality_constraint = h̃,
        parameter_dimension = state_dim(game.dynamics),
        primal_dimensions = [
            horizon * (state_dim(game.dynamics, ii) + control_dim(game.dynamics, ii)) for ii in 1:N
        ],
        equality_dimensions = [1 for _ in 1:N],
        inequality_dimensions = [1 for _ in 1:N],
        shared_equality_dimension = state_dim(game.dynamics) +
                                    (horizon - 1) * state_dim(game.dynamics),
        shared_inequality_dimension = horizon * (
            1 +
            N * length(LazySets.constraints_list(game.env.set)) +
            sum(isfinite.(control_bounds(game.dynamics).lb)) +
            sum(isfinite.(control_bounds(game.dynamics).ub)) +
            sum(isfinite.(state_bounds(game.dynamics).lb)) +
            sum(isfinite.(state_bounds(game.dynamics).ub))
        ),
    )
end

"Generate an initial guess for primal variables following a zero input sequence."
function generate_initial_guess(;
    game::TrajectoryGame{<:ProductDynamics},
    parametric_game::ParametricGame,
    horizon,
    initial_state,
)
    rollout_strategy =
        map(1:num_players(game)) do ii
            (x, t) -> zeros(control_dim(game.dynamics, ii))
        end |> TrajectoryGamesBase.JointStrategy

    zero_input_trajectory =
        TrajectoryGamesBase.rollout(game.dynamics, rollout_strategy, initial_state, horizon)

    vcat(
        pack_trajectory(zero_input_trajectory),
        zeros(total_dim(parametric_game) - sum(parametric_game.primal_dimensions)),
    )
end

"Solve a parametric trajectory game, where the parameter is just the initial state."
function TrajectoryGamesBase.solve_trajectory_game!(
    game::TrajectoryGame{<:ProductDynamics},
    horizon,
    initial_state,
    strategy;
    parametric_game = build_parametric_game(; game, horizon),
    verbose = false,
    solving_info = nothing,
)
    # Solve, maybe with warm starting.
    if !isnothing(strategy.last_solution) && strategy.last_solution.status == PATHSolver.MCP_Solved
        solution = solve(
            parametric_game,
            initial_state;
            initial_guess = strategy.last_solution.variables,
            verbose,
        )
    else
        solution = solve(
            parametric_game,
            initial_state;
            initial_guess = generate_initial_guess(; game, parametric_game, horizon, initial_state),
            verbose,
        )
    end

    if !isnothing(solving_info)
        push!(solving_info, solution.info)
    end

    # Update warm starting info.
    if solution.status == PATHSolver.MCP_Solved
        strategy.last_solution = solution
    end
    strategy.solution_status = solution.status

    # Pack solution into OpenLoopStrategy.
    trajs = unstack_trajectory(unpack_trajectory(mortar(solution.primals); game.dynamics))
    JointStrategy(map(traj -> OpenLoopStrategy(traj.xs, traj.us), trajs))
end

"Receding horizon strategy that supports warm starting."
Base.@kwdef mutable struct WarmStartRecedingHorizonStrategy
    game::TrajectoryGame
    parametric_game::ParametricGame
    receding_horizon_strategy::Any = nothing
    time_last_updated::Int = 0
    turn_length::Int
    horizon::Int
    last_solution::Any = nothing
    context_state::Any = nothing
    solution_status::Any = nothing
end

function (strategy::WarmStartRecedingHorizonStrategy)(state, time)
    plan_exists = !isnothing(strategy.receding_horizon_strategy)
    time_along_plan = time - strategy.time_last_updated + 1
    plan_is_still_valid = 1 <= time_along_plan <= strategy.turn_length

    update_plan = !plan_exists || !plan_is_still_valid
    if update_plan
        strategy.receding_horizon_strategy = TrajectoryGamesBase.solve_trajectory_game!(
            strategy.game,
            strategy.horizon,
            state,
            strategy;
            strategy.parametric_game,
        )
        strategy.time_last_updated = time
        time_along_plan = 1
    end

    strategy.receding_horizon_strategy(state, time_along_plan)
end

"Visualize a strategy `γ` on a makie canvas using the base color `color`."
function TrajectoryGamesBase.visualize!(
    canvas,
    γ::Makie.Observable{<:OpenLoopStrategy};
    color = :black,
    weight_offset = 0.0,
)
    Makie.series!(canvas, γ; color = [(color, min(1.0, 0.9 + weight_offset))])
end

function Makie.convert_arguments(::Type{<:Makie.Series}, γ::OpenLoopStrategy)
    traj_points = map(s -> Makie.Point2f(s[1:2]), γ.xs)
    ([traj_points],)
end

function main(;
    # initial_state = mortar([[50., 50., 50., 0.01, 0.01, 0.01], [0., 0., 0., 0., 0., 0.]]), ##### INPUT #####
    # horizon = 20, ##### INPUT #####
    initial_state = mortar([[6000., -8000., -3000., 0., 0., 0.], [0., 0., 0., 0., 0., 0.]]), ##### INPUT #####
    horizon = 20, ##### INPUT #####
)
    env_size = 10000 ##### INPUT #####
    # env_size = 120000 ##### INPUT #####
    environment = PolygonEnvironment(4, env_size*sqrt(2))
    game = setup_trajectory_game(; environment)
    parametric_game = build_parametric_game(; game, horizon)

    turn_length = 3 ##### INPUT ##### unsure if we should change this
    sim_steps = let
        n_sim_steps = 120 ##### INPUT ##### Note: multiply by dt=5 to get total time
        progress = ProgressMeter.Progress(n_sim_steps)
        receding_horizon_strategy =
            WarmStartRecedingHorizonStrategy(; game, parametric_game, turn_length, horizon)

        rollout(
            game.dynamics,
            receding_horizon_strategy,
            initial_state,
            n_sim_steps;
            get_info = (γ, x, t) ->
                (ProgressMeter.next!(progress); γ.receding_horizon_strategy),
        )
        # add if statement to check if rendezvous is achieved??? If so, break out of loop and return sim_steps
    end

    #: Extract game states and inputs from sim_steps
    states, inputs, strategies = sim_steps
    write("sim_results/states.csv", states)
    write("sim_results/inputs.csv", inputs)

    #: Define constant sun vector (again) for extracting sun angle throughout engagement
    arg_latitude    = pi/2 # rad (90 deg, argument of latitude)
    RAAN            = pi/4 # rad (45 deg, right ascension of ascending node)
    inc             = pi/12 # rad (15 deg, inclination)
    sun_vec_I       = [1/sqrt(10), 3/sqrt(10), 0] # sun vector in inertial frame
    R3_RAAN         = [cos(RAAN) -sin(RAAN) 0; sin(RAAN) cos(RAAN) 0; 0 0 1]
    R1_inc          = [1 0 0; 0 cos(inc) -sin(inc); 0 sin(inc) cos(inc)]
    R3_arg_lat      = [cos(arg_latitude) -sin(arg_latitude) 0; sin(arg_latitude) cos(arg_latitude) 0; 0 0 1]
    sun_vec_L       = R3_arg_lat * R1_inc * R3_RAAN * sun_vec_I # sun vector in LVLH frame
    println("Sun vector (LVLH): ", sun_vec_L)

    #: Parse states/inputs into pursuer and evader states/inputs
    p_states    = zeros(length(states), 6)
    e_states    = zeros(length(states), 6)
    p_inputs    = zeros(length(states), 3)
    e_inputs    = zeros(length(states), 3)
    sun_angles  = zeros(length(states), 1)

    for ii in 1:length(states)
        p_states[ii,:] = states[ii][1:6]
        e_states[ii,:] = states[ii][7:12]
        p_inputs[ii,:] = inputs[ii][1:3]
        e_inputs[ii,:] = inputs[ii][4:6]
        # sun_angles[ii] = dot(p_states[ii,1:3]-e_states[ii,1:3], sun_vec_L) / norm(p_states[ii,1:3]-e_states[ii,1:3])
    end
    error_states    = p_states - e_states
    sun_angles      = [acosd(dot(error_states[ii,1:3], sun_vec_L) / norm(error_states[ii,1:3])) for ii in 1:length(states)]

    println("Final x error: ", error_states[end,1])
    println("Final y error: ", error_states[end,2])
    println("Final z error: ", error_states[end,3])
    println("Final range error: ", norm(error_states[end,1:3]))
    
    #: Plot states/inputs
    time = 0:5:(length(states)-1)*5

    plot(time, p_states[:,1:3]/1000, xlabel = "Time [s]", ylabel = "Position [km]", label = ["x_p" "y_p" "z_p"], title = "Pursuer vs Evader Position")
    plot!(time, e_states[:,1:3]/1000, label = ["x_e" "y_e" "z_e"])
    savefig("sim_results/spe_sun_player_range_states.png")

    plot(time, error_states[:,1:3]/1000, xlabel = "Time [s]", ylabel = "Position Error [km]", label = ["x" "y" "z"], title = "Position Error")
    savefig("sim_results/spe_sun_range_error_range_states.png")

    plot(p_states[:,1]/1000, p_states[:,2]/1000, p_states[:,3]/1000, color = :red, label = "Pursuer", camera = (70, 20), title = "Pursuer vs Evader Trajectories")
    plot!(e_states[:,1]/1000, e_states[:,2]/1000, e_states[:,3]/1000, color = :blue, label = "Evader", xlabel = "x [km]", ylabel = "y [km]", zlabel = "z [km]")
    plot!((p_states[1,1]/1000, p_states[1,2]/1000, p_states[1,3]/1000), marker = :circle, markersize = 3, color = :red, label = "Pursuer Initial State")
    plot!((e_states[1,1]/1000, e_states[1,2]/1000, e_states[1,3]/1000), marker = :circle, markersize = 3, color = :blue, label = "Evader Initial State")
    savefig("sim_results/spe_sun_3d_plot.png")

    plot(time, p_inputs, xlabel = "Time [s]", ylabel = "Inputs [N/kg]", label = ["u_x" "u_y" "u_z"], title = "Pursuer Inputs")
    savefig("sim_results/spe_sun_pursuer_inputs.png")

    plot(time, e_inputs, xlabel = "Time [s]", ylabel = "Inputs [N/kg]", label = ["u_x" "u_y" "u_z"], title = "Evader Inputs")
    savefig("sim_results/spe_sun_evader_inputs.png")

    plot(time, sun_angles, xlabel = "Time [s]", ylabel = "Relative Sun Angle [deg]", title = "Relative Sun Angle")
    savefig("sim_results/spe_sun_sunangles.png")

    anim = @animate for ii in 1:length(states)
        plot(p_states[:,1]/1000, p_states[:,2]/1000, p_states[:,3]/1000, color = :red, camera = (70, 20), title = "Pursuer vs Evader Trajectories")
        plot!(e_states[:,1]/1000, e_states[:,2]/1000, e_states[:,3]/1000, color = :blue, xlabel = "x [km]", ylabel = "y [km]", zlabel = "z [km]")
        plot!((p_states[ii,1]/1000, p_states[ii,2]/1000, p_states[ii,3]/1000), marker = :circle, markersize = 3, color = :red, label = "Pursuer")
        plot!((e_states[ii,1]/1000, e_states[ii,2]/1000, e_states[ii,3]/1000), marker = :circle, markersize = 3, color = :blue, label = "Evader")
    end
    gif(anim, "sim_results/spe_sun_capture.gif", fps = 20)

    anim2 = @animate for ii in 1:2:360
        plot(p_states[:,1]/1000, p_states[:,2]/1000, p_states[:,3]/1000, color = :red, label = "Pursuer", camera = (ii, 10), title = "Pursuer vs Evader Trajectories")
        plot!(e_states[:,1]/1000, e_states[:,2]/1000, e_states[:,3]/1000, color = :blue, label = "Evader", xlabel = "x [km]", ylabel = "y [km]", zlabel = "z [km]")
        plot!((p_states[1,1]/1000, p_states[1,2]/1000, p_states[1,3]/1000), marker = :circle, markersize = 3, color = :red, label = "Evader Initial State")
        plot!((e_states[1,1]/1000, e_states[1,2]/1000, e_states[1,3]/1000), marker = :circle, markersize = 3, color = :blue, label = "Evader Initial State")
    end
    gif(anim2, "sim_results/spe_sun_360.gif", fps = 35)

    animate_sim_steps(game, sim_steps; live = false, framerate = 20, show_turn = true, xlims = (-env_size, env_size), ylims = (-env_size, env_size))
    (; sim_steps, game)
end
