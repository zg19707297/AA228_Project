using LinearAlgebra
using Plots
using DataFrames, CSV, Tables
using StatsBase

mutable struct MDP
    γ
    S
    A
    T
    R
    TR
end

struct MonteCarloTreeSearch
    P
    N
    Q
    d
    m
    c
    U
end

function(π::MonteCarloTreeSearch)(s)
    for k in 1:π.m
        simulate!(π,s)
    end
    A = possible_action(s,π.P.A)
    return argmax(a->π.Q[(s,a)],A)
end

function simulate!(π::MonteCarloTreeSearch, s, d=π.d)
    if d ≤ 0
        return 0.0
    end
    P, N, Q, c = π.P, π.N, π.Q, π.c
    A, TR, γ = P.A, P.TR, P.γ
    # print(P.A,"\n")
    A = possible_action(s,A)
    if !haskey(N, (s,first(A)))
        for a in A
            N[(s,a)] = 0
            Q[(s,a)] = 0.0
        end
        return π.U(s)
    end
    a = explore(π,s)
    s′, r = TR(s,a)
    q = r + γ*simulate!(π,s′,d-1)
    N[(s,a)] += 1
    Q[(s,a)] += (q-Q[(s,a)])/N[(s,a)]
    return q
end

bonus(Nsa,Ns) = Nsa == 0 ? Inf : sqrt(log(Ns)/Nsa)

function explore(π::MonteCarloTreeSearch,s)
    A, N, Q, c = π.P.A, π.N, π.Q, π.c
    A = possible_action(s,A)
    Ns = sum(N[(s,a)] for a in A)
    return argmax(a->Q[(s,a)] + c*bonus(N[(s,a)], Ns), A)
end

function possible_action(s,A)
    a = copy(A)
    if mod(s,dim) == 1 A = filter!(x->x≠1,a) end
    if mod(s,dim) == 0 A = filter!(x->x≠2,a) end
    if s ≤ dim A = filter!(x->x≠3,a) end
    if s > dim*dim-dim A = filter!(x->x≠4,a) end
    return A
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
    if checkindex(Bool,1:dim,s′_x) && checkindex(Bool,1:dim,s′_y)
        return LinearIndices((dim,dim))[s′_y,s′_x]
    else
        return s
    end
end

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

    if s′ == state_action_trans(dim,s,a)
        T = 1-ω+ω/4
    elseif norm([s′_x, s′_y]-[s_x, s_y]) == 1
        T = ω/4
    else
        T = 0
    end
    return T
end

function trans_reward(A,dim,fire,s,a)
    s′_all = [state_action_trans(dim,s,action) for action in A]
    # print(s′_all)
    ω = [trans_prob(dim,fire,s,a,s′) for s′ in s′_all]
    # print(ω)
    s′ = sample(s′_all,Weights(ω))
    r = s′ == fire ? 1 : 0
    return s′, r
end

global dim = 20
γ = 0.95
State = Matrix{Float32}(undef,dim,dim)
S = LinearIndices(State)
A = [1,2,3,4]
fire = S[5,10]
T(s,a,s′) = trans_prob(dim,fire,s,a,s′)
R(s,a) = if s == fire return 1 else 0 end
TR(s,a) = trans_reward(A,dim,fire,s,a)
P = MDP(γ,S,A,T,R,TR)
N = Dict{Tuple{Int64,Int64},Int64}()
Q = Dict{Tuple{Int64,Int64},Float64}()
d = 10
m = 100
c = 1
function value_fcn(s,A)
    A = possible_action(s,A)
    return argmax(a->π.Q[(s,a)],A)
end
U(s) = value_fcn(s,A)
π = MonteCarloTreeSearch(P,N,Q,d,m,c,U)
function propagate()
    s_curr = 1
    i = 1
    while s_curr != fire
        a_curr = π(s_curr)
        s_curr, _ = trans_reward(A,dim,fire,s_curr,a_curr)
        i += 1
        print(i,",",a_curr,",",s_curr,"\n")
        # if i == 2 break end
    end
end
propagate()
