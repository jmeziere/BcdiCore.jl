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
