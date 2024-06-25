struct RampProtocol{Mi,Ma,T,F}
    delta_mins::Mi
    delta_maxes::Ma
    total_time::T
    smooth_step::F
    function RampProtocol(Δmin::Mi, Δmax::Ma, total_time::T, smooth_step::F) where {Mi,Ma,T,F}
        if length(Δmin) != 3 || length(Δmax) != 3
            throw(ArgumentError("Δmin and Δmax must have length 3"))
        end
        new{Mi,Ma,T,F}(Δmin, Δmax, total_time, smooth_step)
    end
end

RampProtocol(Δmin, Δmax, T, k::Number) = RampProtocol(Δmin, Δmax, T, smooth_step(k))

smooth_step(k, x) = 1 / 2 + tanh(k * x) / 2
smooth_step(k) = Base.Fix1(smooth_step, k)

@inbounds function get_deltas(p::RampProtocol, t)
    T = p.total_time
    f = p.smooth_step
    Δmin = p.delta_mins
    Δmax = p.delta_maxes
    shifts = @SVector [0.0, T / 3, 2T / 3]
    fi(i) = Δmin[i] + (Δmax[i] - Δmin[i]) * f(cos(2pi * (t - shifts[i]) / T))
    ntuple(fi, Val(3))
end

(ramp::RampProtocol)(t) = get_deltas(ramp, t)

## Rubens ramp
sigmoid(x, x0, k) = 1 / (1 + exp(-k * (x - x0)))
struct RubensRampProtocol{R,EM,HM,T}
    rate::R
    eps_max::EM
    hop_max::HM
    total_time::T
end

function (rr::RubensRampProtocol)(t)
    t_final = 2 * rr.total_time
    rate = rr.rate
    hop_max = rr.hop_max
    eps_max = rr.eps_max
    t_current = t
    Δ1 = hop_max * (sigmoid(t_current, t_final / 7, rate) - sigmoid(t_current, 2 * t_final / 7, rate) + sigmoid(t_current, 4 * t_final / 7, rate) - sigmoid(t_current, 5 * t_final / 7, rate))
    Δ3 = hop_max * (sigmoid(t_current, 2 * t_final / 7, rate) - sigmoid(t_current, 3 * t_final / 7, rate) + sigmoid(t_current, 5 * t_final / 7, rate) - sigmoid(t_current, 6 * t_final / 7, rate))
    Δ2 = eps_max * (sigmoid(-t_current, -1 * t_final / 7, rate) + sigmoid(t_current, 3 * t_final / 7, rate) - sigmoid(t_current, 4 * t_final / 7, rate) + sigmoid(t_current, 6 * t_final / 7, rate))
    return (Δ1, Δ2, Δ3)
end

# function rubens_hamiltonian((eps, hop, delta, P))
#     dotPs = (P[0, 1], P[2, 4], P[3, 5])
#     sum(eps .* dotPs)
#     hopPs = (P[1, 2], P[1, 3], P[4, 5])

# end