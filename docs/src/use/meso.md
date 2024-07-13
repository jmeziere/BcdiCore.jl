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
