# Spacecraft Pursuit-Evasion Game with Sun Angle Considerations
This code simulates a spacecraft pursuit-evasion (SPE) game between two players.

This repository is heavily based on Assignment 5 from ASE 389: Game-Theoretic Modeling for Multi-Agent Systems at UT Austin.


## Setup
To activate this simulators's package:
```console
julia> ]
pkg> activate .
  Activating environment at `<path to repo>/Project.toml`
(SPEsim) pkg>
```
Now exit package mode by hitting the `[delete]` key. You should see the regular Julia REPL prompt. Type:
```console
julia> using Revise
julia> using SPEsim
julia> using Plots
```


## Examples
The `examples/spe_game_base.jl` file encodes a basic trajectory game as a MCP, using the implementation in `src/parametric_game.jl`:

```console
julia> includet("examples/spe_game_base.jl")
julia> (; sim_steps, game) = main();
```

Substitute `spe_game_base.jl` with `spe_game_sunangle.jl` for a SPE game with sun angle considerations

This will create a video of the animated result called `sim_steps.mp4`.


## Test Implementation
This currently contains no additional tests from Assignment 5. May be used in the future for some basic troubleshooting.

To run tests locally in the REPL:
```console
julia> ]
(SPEsim) pkg> test
```

Alternatively:
```console
julia> include("test/runtests.jl")
```


## Parametric MCP Background
Parametric optimization problems are just optimization problems where the objective and/or constraints depend upon a set of parameters (i.e., rather than only upon the decision variables). For example, if this is a trajectory optimization problem, the parameter could represent a reference speed. The [ParametricMCPs](https://github.com/lassepe/ParametricMCPs.jl) package will let us formulate and solve parametric MCPs using the [PATH](https://pages.cs.wisc.edu/~ferris/path.html) solver, and additionally _differentiate the solution map with respect to the parameter_.