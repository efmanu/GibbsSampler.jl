# GibbsSampler.jl

```@meta
CurrentModule = GibbsSampler
DocTestSetup = quote
    using GibbsSampler
end
```
There are different MCMC sampling algorithm is availbale with GibbsSampler.jl package

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

## adNUTS Sampler
The NUTS sampler can be configured as struct like below:

```@docs
adNUTS
```