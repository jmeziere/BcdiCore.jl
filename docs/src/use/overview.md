In general, BcdiCore.jl will be called by developers of phase retrieval codes, not end users. BcdiCore.jl implements loss functions and derivatives of loss functions for atomic models, mesoscale models, multiscale models, and traditional projection-based methods.

## Available loss functions

Currently, BcdiCore.jl implements two types of losses, the average ``L_2`` norm and the average log-likelihood. 

Explicitly, the average ``L_2`` loss is defined as

```math
L_2 = \frac{1}{N} \sum_u \left( \lvert G(u) \rvert - \lvert F(u) \rvert \right)^2
```
where ``G(u)`` is the simulated electric field, ``\lvert F(u) \rvert^2`` is the measured intensity at a point ``u`` in reciprocal space, and ``N`` is the total number of meaurement points.

The average log-likelihood (for the Poisson distribution) is defined as

```math
\ell = \frac{1}{N} \sum_u \lvert G(u) \rvert^2 - \lvert F(u) \rvert^2 \ln{\left(\lvert G(u) \rvert^2 \right)}
```
