"""
A type storing the status at the end of a call to `ssa`:

- **termination_status** : whether the simulation stops at the final time (`finaltime`) or early due to zero propensity function (`zeroprop`)
- **nsteps** : the number of steps taken during the simulation.
"""

struct SSAStats
    termination_status :: String
    nsteps :: Int64
end

"""
A type storing the call to `ssa`:

- **initial_state** : a `Vector` of `Int64`, representing the initial states of the system.
- **F** : a `Function` or a callable type, which itself takes two arguments; x, a `Vector` of `Int64` representing the states, and parameters, a `Vector` of `Float64` representing the parameters of the system.
- **rates** : a `Matrix` of `Int64`, representing the transitions of the system, organised by row.
- **parameters** : a `Vector` of `Float64` representing the parameters of the system.
- **tf** : the final simulation time (`Float64`).
- **alg** : the algorithm used (`Symbol`, either `:gillespie`, `jensen`, or `tjc`).
- **tvc** : whether rates are time varying.
"""
struct SSAArgs{X, Ftype, N, P}
    initial_state :: X
                F :: Ftype
            rates :: N
       parameters :: P
               tf :: Float64
              alg :: Symbol
              tvc :: Bool
end

"
This type stores the output of `ssa`, and comprises of:

- **time** : a `Vector` of `Float64`, containing the times of simulated events.
- **data** : a `Matrix` of `Int64`, containing the simulated states.
- **stats** : an instance of `SSAStats`.
- **args** : arguments passed to `ssa`.

"
struct SSAResult
     time :: Vector{Float64}
     data :: Matrix{Int64}
    stats :: SSAStats
     args :: SSAArgs
end


"
This function is a substitute for `StatsBase.sample(wv::WeightVec)`,
which avoids recomputing the sum and size of the weight vector,
as well as a type conversion of the propensity vector.
It takes the following arguments:

- weights : an `Array{Float64, 1}`, representing propensity function weights.
- total : the sum of `w`.
- n : the length of `w`.

"
function propensity_sample(weights, s, n)
    t = rand() * s
    i = 1
    cw = weights[1]
    while cw < t && i < n
        i += 1
        @inbounds cw += weights[i]
    end
    return i
end

const pfsample = propensity_sample

"""
This function performs Gillespie's stochastic simulation algorithm. It takes the following arguments:

- initial_state: a `Vector` of `Int64`, representing the initial states of the system.

- F: a `Function` or a callable type, which itself takes two arguments; 
     x, a `Vector` of `Int64` representing the states, and parameters, 
     a `Vector` of `Float64` representing the parameters of the system.

- rates: a `Matrix` of `Int64`, representing the transitions of the system, organised by row.

- parameters: a `Vector` of `Float64` representing the parameters of the system.

- stop_time: the final simulation time (`Float64`).
"""
function gillespie(initial_state, propensity_function, rates, parameters, stop_time)

    # Args
    args = SSAArgs(initial_state, propensity_function, rates, parameters, stop_time, :gillespie, false)

    # Set up time array
    times = Float64[]
    time = 0.0
    push!(times, time)

    # Set up initial state
    nstates = length(initial_state)
    state = copy(initial_state')
    data = copy(Array(initial_state))

    # Number of propensity functions
    n_propensities = size(rates, 1)

    # Main loop
    termination_status = "finaltime"
    nsteps = 0

    while time <= stop_time
        propensity = propensity_function(state, parameters)

        # Update time
        total_propensity = sum(propensity)

        if total_propensity == 0.0
            termination_status = "zeroprop"
            break
        end

        # Time step
        time_step = rand(Exponential(1 / total_propensity))
        time += time_step
        push!(times, time)

        # Update event
        event = propensity_sample(propensity, total_propensity, n_propensities)

        if state isa SVector
            @inbounds state[1] += rates[event, :]
        else
            Δx = view(rates, event, :)

            for i in 1:nstates
                @inbounds state[1, i] += Δx[i]
            end
        end

        push!(data, state...)

        # Update nsteps
        nsteps += 1
    end

    statistics = SSAStats(termination_status, nsteps)
    data = transpose(reshape(data, length(state), nsteps+1))

    return SSAResult(times, data, statistics, args)
end

include("truejump.jl")
include("jensen.jl")
