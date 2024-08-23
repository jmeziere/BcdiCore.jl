# BcdiCore Documentation

## About

Bragg Coherent Diffraction Imaging (BCDI) Core implements some of the core functionality used for the Julia BCDI packages BcdiCore.jl implements the loss functions and derivatives of loss functions used in these packages.

While this package is marked as BCDI specific, the methods are more general and can be used in many phase retrieval problems. In the future, this package may be incorporated into a more general phase retrieval core package.

Currently, this entire package must be run with access to GPUs. This may change in the future (especially if Issues requesting it are opened), but for our research group, using GPUs is a necessity.

## Installation

BcdiCore.jl is registered in the Julia general registry and can be installed by running in the REPL package manager (```]```):

```add BcdiCore```
