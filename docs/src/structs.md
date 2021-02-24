# GibbsSampler.jl

```@meta
CurrentModule = GibbsSampler
DocTestSetup = quote
    using GibbsSampler
end
```
There are different MCMC sampling algorithm is available with GibbsSampler.jl package.

## MH Sampler
The MH sampler can be configured as struct like below:

```@docs
MH
```

## adHMC Sampler
The HMC sampler can be configured as struct like below:

```@docs
adHMC
```
The `backend` variable is used to select the automatic differentiation backend. `ForwardDiff` and `ReversedDiff` are the values to select forward and reverse differentiation, respectively.

## adNUTS Sampler
The NUTS sampler can be configured as struct like below:

```@docs
adNUTS
```
The `backend` variable is used to select the automatic differentiation backend. `ForwardDiff` and `ReversedDiff` are the values to select forward and reverse differentiation, respectively.