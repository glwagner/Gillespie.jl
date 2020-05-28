"
This function performs stochastic simulation using thinning/uniformization/Jensen's method, returning only the thinned jumps. It takes the following arguments:

- **initial_state** : a `Vector` of `Int64`, representing the initial states of the system.
- **F** : a `Function` or a callable type, which itself takes two arguments; x, a `Vector` of `Int64` representing the states, and parameters, a `Vector` of `Float64` representing the parameters of the system. In the case of time-varying systems, a third argument, a `Float64` representing the time of the system should be added
- **nu** : a `Matrix` of `Int64`, representing the transitions of the system, organised by row.
- **parameters** : a `Vector` of `Float64` representing the parameters of the system.
- **tf** : the final simulation time (`Float64`).
- **max_rate**: the maximum rate (`Float64`).
"
function jensen(initial_state::AbstractVector{Int64},F::Base.Callable,nu::AbstractMatrix{Int64},parameters::AbstractVector{Float64},tf::Float64,max_rate::Float64,thin::Bool=true)
    if thin==false
      return jensen_alljumps(initial_state::AbstractVector{Int64},F::Base.Callable,nu::Matrix{Int64},parameters::AbstractVector{Float64},tf::Float64,max_rate::Float64)
    end
    tvc=true
    try
      F(initial_state,parameters,0.0)
    catch
      tvc=false
    end
    # Args
    args = SSAArgs(initial_state,F,nu,parameters,tf,:jensen,tvc)
    # Set up time array
    ta = Vector{Float64}()
    t = 0.0
    push!(ta,t)
    # Set up initial x
    nstates = length(initial_state)
    x = copy(initial_state')
    xa = copy(initial_state)
    # Number of propensity functions; one for no event
    numpf = size(nu,1)+1
    # Main loop
    termination_status = "finaltime"
    nsteps = 0
    while t <= tf
        dt = rand(Exponential(1/max_rate))
        t += dt
        if tvc
          pf = F(x,parameters,t)
        else
          pf = F(x,parameters)
        end
        # Update time
        sumpf = sum(pf)
        if sumpf == 0.0
            termination_status = "zeroprop"
            break
        end
        if sumpf > max_rate
            termination_status = "upper_bound_exceeded"
            break
        end
        # Update event
        ev = pfsample([pf; max_rate-sumpf],max_rate,numpf+1)
        if ev < numpf
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
          push!(ta,t)
          # update nsteps
          nsteps += 1
        end
    end
    stats = SSAStats(termination_status,nsteps)
    xar = transpose(reshape(xa,length(x),nsteps+1))
    return SSAResult(ta,xar,stats,args)
end

"
This function performs stochastic simulation using thinning/uniformization/Jensen's method, returning all the jumps, both real and 'virtual'. It takes the following arguments:

- **initial_state** : a `Vector` of `Int64`, representing the initial states of the system.
- **F** : a `Function` or a callable type, which itself takes two arguments; x, a `Vector` of `Int64` representing the states, and parameters, a `Vector` of `Float64` representing the parameters of the system. In the case of time-varying systems, a third argument, a `Float64` representing the time of the system should be added
- **nu** : a `Matrix` of `Int64`, representing the transitions of the system, organised by row.
- **parameters** : a `Vector` of `Float64` representing the parameters of the system.
- **tf** : the final simulation time (`Float64`).
- **max_rate**: the maximum rate (`Float64`).
"
function jensen_alljumps(initial_state::AbstractVector{Int64},F::Base.Callable,nu::AbstractMatrix{Int64},parameters::AbstractVector{Float64},tf::Float64,max_rate::Float64)
    # Args
    tvc=true
    try
      F(initial_state,parameters,0.0)
    catch
      tvc=false
    end
    # Args
    args = SSAArgs(initial_state,F,nu,parameters,tf,:jensen,tvc)
    # Set up time array
    ta = Vector{Float64}()
    t = 0.0
    push!(ta,t)
    while t < tf
      dt = rand(Exponential(1/max_rate))
      t += dt
      push!(ta,t)
    end
    nsteps=length(ta)-1
    # Set up initial x
    nstates = length(initial_state)
    x = copy(initial_state')
    xa = Array{Int64,1}(undef, (nsteps+1)*nstates)
    xa[1:nstates] = x
    # Number of propensity functions; one for no event
    numpf = size(nu,1)+1
    # Main loop
    termination_status = "finaltime"
    k=1 # step counter
    while k <= nsteps
        if tvc
          t=ta[k]
          pf=F(x,parameters,t)
        else
          pf = F(x,parameters)
        end
        sumpf = sum(pf)
        if sumpf == 0.0
            termination_status = "zeroprop"
            break
        end
        if sumpf > max_rate
            termination_status = "upper_bound_exceeded"
            break
        end
        # Update event
        ev = pfsample([pf; max_rate-sumpf],max_rate,numpf+1)
        if ev < numpf # true jump
          deltax = view(nu,ev,:)
          for i in 1:nstates
              @inbounds xa[k*nstates+i] = xa[(k-1)*nstates+i]+deltax[i]
          end
        else
          for i in 1:nstates
              @inbounds xa[k*nstates+i] = xa[(k-1)*nstates+i]
          end
        end
        k +=1
    end
    stats = SSAStats(termination_status,nsteps)
    xar = transpose(reshape(xa,length(x),nsteps+1))
    return SSAResult(ta,xar,stats,args)
end

"This takes a single argument of type `SSAResult` and returns a `DataFrame`."
function ssa_data(s::SSAResult)
    hcat(DataFrame(time=s.time),convert(DataFrame,s.data))
end
