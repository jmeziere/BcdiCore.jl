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
