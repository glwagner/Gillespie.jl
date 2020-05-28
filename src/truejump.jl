"
This function performs the true jump method for piecewise deterministic Markov processes. It takes the following arguments:

- **initial_state** : a `Vector` of `Int64`, representing the initial states of the system.
- **F** : a `Function` or a callable type, which itself takes three arguments; x, a `Vector` of `Int64` representing the states, parameters, a `Vector` of `Float64` representing the parameters of the system, and t, a `Float64` representing the time of the system.
- **nu** : a `Matrix` of `Int64`, representing the transitions of the system, organised by row.
- **parameters** : a `Vector` of `Float64` representing the parameters of the system.
- **tf** : the final simulation time (`Float64`).
"
function truejump(initial_state::AbstractVector{Int64},F::Base.Callable,nu::AbstractMatrix{Int64},parameters::AbstractVector{Float64},tf::Float64)
    # Args
    args = SSAArgs(initial_state,F,nu,parameters,tf,:tjm,true)
    # Set up time array
    ta = Vector{Float64}()
    t = 0.0
    push!(ta,t)
    # Set up initial x
    nstates = length(initial_state)
    x = initial_state'
    xa = copy(initial_state)
    # Number of propensity functions
    numpf = size(nu,1)
    # Main loop
    termination_status = "finaltime"
    nsteps = 0
    while t <= tf
        ds = rand(Exponential(1.0))
        f = (u)->(quadgk((u)->sum(F(x,parameters,u)),t,u)[1]-ds)
        newt = fzero(f,t)
        if newt>tf
          break
        end
        t=newt
        pf = F(x,parameters,t)
        # Update time
        sumpf = sum(pf)
        if sumpf == 0.0
            termination_status = "zeroprop"
            break
        end
        push!(ta,t)
        # Update event
        ev = pfsample(pf,sumpf,numpf)
        if x isa SVector
            @inbounds x[1] += nu[ev,:]
        else
            deltax = view(nu,ev,:)
            for i in 1:nstates
                @inbounds x[1,i] += deltax[i]
            end
        end
        for xx in x
            push!(xa,xx)
        end
        # update nsteps
        nsteps += 1
    end
    stats = SSAStats(termination_status,nsteps)
    xar = transpose(reshape(xa,length(x),nsteps+1))
    return SSAResult(ta,xar,stats,args)
end
