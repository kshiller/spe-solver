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

"Utility to set up a (two player) SPE game."
function setup_trajectory_game(; environment = PolygonEnvironment(4, 300)) #TODO! find suitable default environment
    # PolygonEnvironment(num_sides, radius)
    #NOTE: The environment here inherently imposes state constraints and is used for visualization.
    cost = let
        function stage_cost(x, u, t, θ)
            x1, x2 = blocks(x)
            u1, u2 = blocks(u)

            #: Define cost structure
            Q       = Matrix(1I, length(x1), length(x1))
            R       = Matrix(1I, length(u1), length(u1)) * 1e-2 #TODO! correct to 1e8 when completed testing
            gam_sq  = 2
            x_diff  = x1 - x2
            cost    = dot(x_diff, Q * x_diff) + dot(u1, R * u1) - gam_sq * dot(u2, R * u2)

            #: P1 (pursuer) wants to minimize cost, and P2 (evader) wants to maximize cost.
            [
                cost,
                -cost, #TODO! verify this is correct implementation
            ]

            # # P1 wants to go fast, and P2 wants to be close to P1.
            # [
            #     -sqrt(0.1 + norm(x1[3:4])) + 0.01norm_sqr(u1),
            #     norm_sqr(x2[1:2] - x1[1:2]) + 0.01norm_sqr(u2),
            # ]
        end

        function reducer(stage_costs)
            reduce(.+, stage_costs) ./ length(stage_costs)
        end

        TimeSeparableTrajectoryGameCost(stage_cost, reducer, GeneralSumCostStructure(), 1.0)
    end

    function coupling_constraints(xs, us, θ)
        mapreduce(vcat, xs) do x
            x1, x2 = blocks(x)

            #TODO! update coupled constraints; there may be none! see other constraints below
            # Players need to stay at least 1 m away from one another.
            norm_sqr(x1[1:2] - x2[1:2]) - 0#1
        end
    end

    #: Define HCW dynamics
    SMA     = 15000 # km (virtual reference semi-major axis)
    mu      = 3.986e5 # km^3/s^2 (gravitational parameter)
    n       = sqrt(mu / SMA^3) # rad/s (virtual reference mean motion)
    # mass    = 1 # kg (mass of spacecraft, assumed constant/same for both sats)
    A = zeros(6, 6)
    A[1:3, 4:6] = Matrix(1I, 3, 3)
    A[4,1]      = 3*n^2
    A[4,5]      = 2*n
    A[5,4]      = -2*n
    A[6,3]      = -n^2
    B = zeros(6, 3)
    B[4:6,1:3]  = Matrix(1I, 3, 3) #* 1/mass

    #TODO! verify dynamics implementation
    #TODO! update state/control bounds as needed
    agent_dynamics = time_invariant_linear_dynamics(; A, B,
                            state_bounds = (; lb = [-Inf, -Inf, -Inf, -50, -50, -50], ub = [Inf, Inf, Inf, 50, 50, 50]),
                            control_bounds = (; lb = [-10, -10, -10], ub = [10, 10, 10])
                            )

    # agent_dynamics = planar_double_integrator(;
    #     state_bounds = (; lb = [-Inf, -Inf, -50, -50], ub = [Inf, Inf, 50, 50]),
    #     control_bounds = (; lb = [-100, -100], ub = [100, 100]),
    # )
    
    dynamics = ProductDynamics([agent_dynamics for _ in 1:2])

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
            #TODO! verify once coupling_constraints updated
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
    initial_state = mortar([[50, 50, 50, 0.01, 0.01, 0.01], [0, 0, 0, 0, 0, 0]]),
    horizon = 10, #TODO! update horizon (like sliding window horizon in MPC)
)
    environment = PolygonEnvironment(4, 300) #TODO! create proper environment
    game = setup_trajectory_game(; environment)
    parametric_game = build_parametric_game(; game, horizon)

    turn_length = 3 #TODO! update turn length???
    sim_steps = let
        n_sim_steps = 2000 #TODO! update number of simulation steps
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
    end

    animate_sim_steps(game, sim_steps; live = false, framerate = 20, show_turn = true)
    (; sim_steps, game)
end
