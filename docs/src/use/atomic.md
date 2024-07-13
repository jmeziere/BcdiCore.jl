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
