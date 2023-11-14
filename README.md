# Spacecraft Pursuit-Evasion Game with Sun Lighting
This code simulates a spacecraft pursuit-evasion (SPE) game between two players.

This repository is heavily based on Assignment 5 from ASE 389: Game-Theoretic Modeling for Multi-Agent Systems at UT Austin.


## Setup
To activate this simulators's package:
```console
julia> ]
pkg> activate .
  Activating environment at `<path to repo>/Project.toml`
(RPOsim) pkg>
```
Now exit package mode by hitting the `[delete]` key. You should see the regular Julia REPL prompt. Type:
```console
julia> using Revise
julia> using RPOsim
```


## Examples
The `examples/trajectory_game_example.jl` file encodes a trajectory game as a MCP, using the implementation in `src/parametric_game.jl`.:

```console
julia> includet("examples/trajectory_game_example.jl")
julia> (; sim_steps, game) = main();
```

This will create a video of the animated result called `sim_steps.mp4`.


## Test Implementation
This currently contains no additional tests from Assignment 5. May be used in the future for some basic troubleshooting.

To run tests locally in the REPL:
```console
julia> ]
(RPOsim) pkg> test
```

Alternatively:
```console
julia> include("test/runtests.jl")
```


## Parametric MCP Background
Parametric optimization problems are just optimization problems where the objective and/or constraints depend upon a set of parameters (i.e., rather than only upon the decision variables). For example, if this is a trajectory optimization problem, the parameter could represent a reference speed. The [ParametricMCPs](https://github.com/lassepe/ParametricMCPs.jl) package will let us formulate and solve parametric MCPs using the [PATH](https://pages.cs.wisc.edu/~ferris/path.html) solver, and additionally _differentiate the solution map with respect to the parameter_.