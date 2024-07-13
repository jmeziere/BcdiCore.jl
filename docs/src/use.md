# Overview

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

# Atomic Models

## Mathematical Definitions

For the atomic model, ``G(u)`` is defined as

```math
G(h,k,l) = \sum_j e^{-i (x_j (h+G_h) + y_j (k+G_k) + z_j (l+G_l))} \\
```

where ``x_j, y_j, z_j`` are atom positions and ``h,k,l`` represent a distance away from some scattering vector ``G_h, G_k, G_l`` in reciprocal space. It is important that the ``h,k,l`` value are integers and that they range from ``-\frac{n}{2} \to \frac{n}{2}-1``, so both real space and reciprocal space positions must be scaled. The ``x_j,y_j,z_j`` positions should be shifted to lie between ``0 \to 1`` and should be multiplied by ``2\pi`` to capture the missing ``2 \pi`` scaling in the Fourier transform exponent.

## Usage

Calculating the loss function and its derivative for the atomic model is done in three steps. First, the ```BcdiCore.AtomicState``` struct is created. Then, the atom positions are set by calling ```BcdiCore.setpts!```. Finally, the loss function is calculated with ```BcdiCore.loss```.

```
state = AtomicState(lossType, scale, intens, G, h, k, l)
setpts!(state, x, y, z, getDeriv)
lossVal = loss(state, getDeriv, getLoss)
```

If the derivative is requested with the ```getDeriv``` variable, the results are stored in ```state.xDeriv```,  ```state.yDeriv```, and ```state.zDeriv```.


# Mesoscale Models

## Mathematical Definitions

Similar to the atomic model, ``G(u)`` is initially defined as

```math
G(h,k,l) = \sum_j e^{-i (x'_j (h+G_h) + y'_j (k+G_k) + z'_j (l+G_l))} \\
```

where ``x'_j, y'_j, z'_j`` are atom positions and ``h,k,l`` represent a distance away from some scattering vector ``G_h, G_k, G_l`` in reciprocal space. However, ``x'_j, y'_j, z'_j`` can be thought of as an addition of lattice spacings and displacement vectors, i.e.  ``x_j+ux_j, y_j+uy_j, z_j+uz_j``. Then, if ``G_h,G_k,G_l`` are reciprocal lattice vectors, we find that ``x \cdot G`` is an integer multiple of ``2\pi``, so it does not affect the simulated electric field. We are then left with

```math
G(h,k,l) = \sum_j e^{-i (x_j G_h + y_j G_k + uz_j G_l)} e^{-i (ux_j (h+G_h) + uy_j (k+G_k) + uz_j (l+G_l))} \\
```

Coarse graining to get a mesoscale model, we get

```math
G(h,k,l) = \sum_j \rho_j e^{-i (x_j h + y_j k + uz_j l)} e^{-i (ux_j (h+G_h) + uy_j (k+G_k) + uz_j (l+G_l))} \\
```

Again, it is important that the ``h,k,l`` value are integers and that they range from ``-\frac{n}{2} \to \frac{n}{2}-1``, so both real space and reciprocal space positions must be scaled. The ``x'_j,y'_j,z'_j`` positions should be shifted to lie between ``0 \to 1`` and should be multiplied by ``2\pi`` to capture the missing ``2 \pi`` scaling in the Fourier transform exponent.

## Usage

Calculating the loss function and its derivative for the mesoscale model is done in three steps. First, the ```BcdiCore.MesoState``` struct is created. Then, the atom positions are set by calling ```BcdiCore.setpts!```. Finally, the loss function is calculated with ```BcdiCore.loss```.

```
state = MesoState(lossType, scale, intens, G, h, k, l)
setpts!(state, x, y, z, rho, ux, uy, uz, getDeriv)
lossVal = loss(state, getDeriv, getLoss)
```

If the derivative is requested with the ```getDeriv``` variable, the results are stored in ```state.rhoDeriv```, ```state.uxDeriv```,  ```state.uyDeriv```, and ```state.uzDeriv```.

# Traditional Models

## Mathematical Definitions

Similar to the mesoscale model, ``G(u)`` is initially defined as

```math
G(h,k,l) = \sum_j \rho_j e^{-i (x_j h + y_j k + uz_j l)} e^{-i (ux_j (h+G_h) + uy_j (k+G_k) + uz_j (l+G_l))} \\
```

where ``x_j, y_j, z_j`` are real space positions, ``ux_j, uy_j, uz_j`` are diplacement vectors, and ``h,k,l`` represent a distance away from some scattering vector ``G_h, G_k, G_l`` in reciprocal space. However, we assume that, because the distance from the scattering vector and the displacement vectors are small, ``u \cdot h`` is negligible. So we are left with

```math
G(h,k,l) = \sum_j \rho_j e^{-i (x_j h + y_j k + uz_j l)} e^{-i (ux_j G_h + uy_j G_k + uz_j G_l)} \\
```

Then, we combine the entire ``\rho_j e^{-i (ux_j G_h + uy_j G_k + uz_j G_l)}`` quantity as one variable and get

```math
G(h,k,l) = \sum_j \psi_j e^{-i (x_j h + y_j k + uz_j l)} \\
```

In this case, this is an ordinary Fourier transform, so we put the factor of ``2\pi`` back into ``G(h,k,l)`` to get

```math
G(h,k,l) = \sum_j \psi_j e^{-2 \pi i (x_j h + y_j k + uz_j l)} \\
```

## Usage

Calculating the loss function and its derivative for the traditional model is done in two steps. First, the ```BcdiCore.TradState``` struct is created. Then, the loss function is calculated with ```BcdiCore.loss```.

```
state = TradState(losstype, scale, intens, realSpace)
lossVal = loss(state, getDeriv, getLoss)
```

If the derivative is requested with the ```getDeriv``` variable, the result us stored in ```state.deriv```.

# Multiscale Models

## Mathematical Definitions

The multiscale model is a combination of an atomic scale and a mesoscale model. In this case,  ``G(h,k,l)`` is defined as

```math
G(h,k,l) = G_a(h,k,l) + G_m(h,k,l)
```

where ``a`` signifies the atomic model and ``m`` signifies the mesoscale model.

## Usage

Calculating the loss function and its derivative for the mesoscale model is done in three steps. First, the ```BcdiCore.MultiState``` struct is created. Then, the atom positions are set by calling ```BcdiCore.setpts!```. Finally, the loss function is calculated with ```BcdiCore.loss```.

```
state = MultiState(lossType, scale, intens, G, h, k, l)
setpts!(state, x, y, z, mx, my, mz, rho, ux, uy, uz, getDeriv)
lossVal = loss(state, getDeriv, getLoss)
```

Here ```x, y, z``` are atomic positions and ```mx, my, mz``` are the real space locations of the mesoscale model.

If the derivative is requested with the ```getDeriv``` variable, the results are stored in ```state.xDeriv```,  ```state.yDeriv```, and ```state.zDeriv```, ```state.rhoDeriv```, ```state.uxDeriv```,  ```state.uyDeriv```, and ```state.uzDeriv```.
