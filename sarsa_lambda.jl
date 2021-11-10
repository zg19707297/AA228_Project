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

# mutable struct EpsilonGreedyExploration
#     ϵ
#     σ
# end

mutable struct SarsaLambda
    S
    A
    γ
    Q
    N
    α
    λ
    l
end

# function (π::EpsilonGreedyExploration)(model,s)
#     A, ϵ = model.A, π.ϵ
#     A = possible_action(s,A)
#     # print(s,A,"\n")
#     if rand() < ϵ
#         return rand(A)
#     end
#     Q(s,a) = lookahead(model, s, a)
#     return argmax(a->Q(s,a),A)
# end

# function simulate(P::MDP, model, π, h, s)
#     for i in 1:h
#         a = π(model, s)
#         # print(s,",",a,"\n")
#         s′, r = P.TR(s,a)
#         # print(s′,"\n")
#         model = update!(model, s, a, r, s′)
#         s = s′
#     end
#     return model
# end

function sampling(P::MDP,k,s)
    M = Matrix{Int}(undef,1,4)
    for i in 1:50
        print(i)
        for j in 1:k
            A = possible_action(s,P.A)
            a = sample(A,Weights(ones(length(A))))
            s′, r = P.TR(s,a)
            s = s′
            M = cat(M,[s a s′ r],dims=1)
        end
    end
    return M[2:end,:]
end
        

lookahead(model::SarsaLambda,s,a) = model.Q[s,a]

function update!(model::SarsaLambda,s,a,r,s′)
    if model.l ≠ nothing
        γ, λ, Q, α, l = model.γ, model.λ, model.Q, model.α, model.l
        model.N[l.s,l.a] += 1
        δ = l.r + γ*Q[s,a] - Q[l.s,l.a]
        for s in model.S
            for a in model.A
                model.Q[s,a] += α*δ*model.N[s,a]
                model.N[s,a] *= γ*λ
            end
        end
    else
        model.N[:,:] .= 0.0
    end
    model.l = (s=s,a=a,r=r)
    return model
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
    # print(s,a,s′,T,"\n")
    return T
end

function trans_reward(A,dim,fire,s,a)
    A = possible_action(s,A)
    s′_all = [state_action_trans(dim,s,action) for action in A]
    # print(s,s′_all,"\n")
    ω = [trans_prob(dim,fire,s,a,s′) for s′ in s′_all]
    # print(s,a,ω,"\n")
    s′ = sample(s′_all,Weights(ω))
    r = s′ == fire ? 1 : 0
    # print(s,a,s′,"\n")
    return s′, r
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

function possible_action(s,A)
    a = copy(A)
    if mod(s,dim) == 1 A = filter!(x->x≠1,a) end
    if mod(s,dim) == 0 A = filter!(x->x≠2,a) end
    if s ≤ dim A = filter!(x->x≠3,a) end
    if s > dim*dim-dim A = filter!(x->x≠4,a) end
    return A
end

dim = 20
γ = 0.95
State = Matrix{Float32}(undef,dim,dim)
S = LinearIndices(State)
A = [1,2,3,4]
fire = S[5,10]
T(s,a,s′) = trans_prob(dim,fire,s,a,s′)
R(s,a) = if s == fire return 1 else 0 end
TR(s,a) = trans_reward(A,dim,fire,s,a)
P = MDP(γ,S,A,T,R,TR)
Q = zeros(length(P.S),length(P.A))
N = zeros(length(P.S),length(P.A))
α = 0.1 #learning rate
λ = 0.9
model = SarsaLambda(P.S, P.A, P.γ, Q, N, α, λ, nothing)
# global ϵ = 0.7
# global σ = 1
k = 1000
# π = EpsilonGreedyExploration(ϵ,σ)

function compute(model)
    M = sampling(P,k,1)
    ndata = size(M,1)
    converge = false
    i = 0
    s, a, r, s′ = M[:,1], M[:,2], M[:,3], M[:,4]
    while !converge
        i += 1
        print(i,"\n")
        Q_prev = copy(model.Q)
        for row in 1:ndata
            model = update!(model,s[row],a[row],r[row],s′[row])
        end
        print(maximum(abs.(model.Q-Q_prev)),"\n")
        if maximum(abs.(model.Q-Q_prev)) < 10^(-5)
            converge = true
        end
    end
    return model
end
model = compute(model)