# BcdiCore

<!-- [![Build Status](https://github.com/jmeziere/BcdiCore.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/jmeziere/BcdiCore.jl/actions/workflows/CI.yml?query=branch%3Amain) -->
[![](https://img.shields.io/badge/Docs-Full-blue.svg)](https://byu-cxi.github.io/BcdiDocs/dev)
[![](https://img.shields.io/badge/Docs-Part-blue.svg)](https://byu-cxi.github.io/BcdiCore.jl/dev)

## About

Bragg Coherent Diffraction Imaging (BCDI) Core implements some of the core functionality used in a suite of julia packages solving the BCDI problem. BcdiCore.jl implements the loss functions and derivatives of loss functions used in these packages.

While this package is marked as BCDI specific, the methods are more general and can be used in many phase retrieval problems. In the future, this package may be incorporated into a more general phase retrieval core package.

Currently, this entire package must be run with access to GPUs. This may change in the future (especially if Issues requesting it are opened), but for our research group, using GPUs is a necessity.

## Installation

Currently, BcdiCore.jl is not registered in the Julia general registry and can be installed by running in the REPL package manager (```]```):

```add BcdiCore```
