using LinearAlgebra
using Plots
using DataFrames, CSV, Tables

mutable struct MDP
    γ
    S
    A
    T
    R
    TR
end

struct ValueFunctionPolicy
    P
    U
end

struct ValueIteration
    convergence
end

function lookahead(P::MDP,U,s,a)
    S,T,R,γ = P.S, P.T, P.R, P.γ
    return R(s,a) + γ*sum(T(s,a,s′)*U[i] for (i,s′) in enumerate(S))
end

function backup(P::MDP, U, s)
    return maximum(lookahead(P,U,s,a) for a in P.A)
end

function greedy(P::MDP,U,s)
    u,a = findmax(a->lookahead(P,U,s,a),P.A)
    return (a=a,u=u)
end

(π::ValueFunctionPolicy)(s) = greedy(π.P,π.U,s).a

function solve(M::ValueIteration,P::MDP)
    U = [0.0 for s in P.S]
    U_prev = copy(U)
    converge = false
    i = 0
    while !converge
        i += 1
        print(i,"\n")
        U = [backup(P,U,s) for s in P.S]
        
        if maximum(abs.(U-U_prev)) < M.convergence
            converge = true
        end
        plot1 = heatmap(U,show=true)
        display(plot1)
        print(maximum(abs.(U-U_prev)),"\n")
        U_prev = copy(U)
    end
    return ValueFunctionPolicy(P,U)
end

function state_action_trans(dim,s,a)
    # Given a grid dimension (int), a state (linear index), an action (int)
    # return corresponding s′ (Cartesian index) in the (dim,dim) grid world

    s_y, s_x = CartesianIndices((dim,dim))[s][1],CartesianIndices((dim,dim))[s][2]   # CartesianIndices return (y,x)

    # action 1,2,3,4 corresponds to up (y-1), down (y+1), left (x-1), right (x+1)
    # due to the nature of CartesianIndices
    if a == 1
        s′_x, s′_y = s_x, s_y-1
    elseif a == 2
        s′_x, s′_y = s_x, s_y+1
    elseif a == 3
        s′_x, s′_y = s_x-1, s_y
    elseif a == 4
        s′_x, s′_y = s_x+1, s_y
    end
    return (s′_x, s′_y)
end

###############
# No wind
function trans_prob(dim,fire,s,a,s′)
    # extract cartesian index of fire and agent from linear index
    fire_y, fire_x = CartesianIndices((dim,dim))[fire][1],CartesianIndices((dim,dim))[fire][2]
    s_y, s_x = CartesianIndices((dim,dim))[s][1],CartesianIndices((dim,dim))[s][2]
    s′_y, s′_x = CartesianIndices((dim,dim))[s′][1],CartesianIndices((dim,dim))[s′][2]
    # compute distance between fire and agent
    dist = floor(norm([fire_x, fire_y]-[s_x,s_y])*2)/2
    # different level of turbulence as a piecewise function of distance
    # for a specific range of dist, there is ω probability the agent will move in a random direction,
    # each direction with ω/4 probability. There is an additional 1-ω probability the drone will move
    # in the desired position.
    if dist <= 0.5
        ω = 0.9
    elseif dist <= 1
        ω = 0.8
    elseif dist <= 1.5
        ω = 0.6
    elseif dist <= 2
        ω = 0.4
    elseif dist <= 2.5
        ω = 0.2
    else
        ω = 0
    end

    if (s′_x, s′_y) == state_action_trans(dim,s,a)
        T = 1-ω+ω/4
    elseif norm([s′_x, s′_y]-[s_x, s_y]) == 1
        T = ω/4
    else
        T = 0
    end
    return T
end

dim = 20
γ = 0.95
State = Matrix{Float32}(undef,dim,dim)
S = LinearIndices(State)
A = [1,2,3,4]
fire = S[5,10]
T(s,a,s′) = trans_prob(dim,fire,s,a,s′)
R(s,a) = if s == fire return 1 else 0 end
P = MDP(γ,S,A,T,R,nothing)
M = ValueIteration(10^(-4))
Result = solve(M,P)
CSV.write("ValueIteration.csv", Tables.table(Result.U), writeheader=false)