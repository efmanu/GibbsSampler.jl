# GibbsSampler.jl

```@meta
CurrentModule = GibbsSampler
DocTestSetup = quote
    using GibbsSampler
end
```

This package helps to generate posterior samples using [Gibbs sampling algorithm](https://en.wikipedia.org/wiki/Gibbs_sampling) from a specified multivariate probability distribution when direct sampling is difficult.
This Julia package supports MH and HMC based algorithms with different automatic differentiation backends.

## Gibbs Sampling
```@docs
gibbs(alg, sample_alg, logJoint::Function;  
		revt = [reverse_transform for _ in 1:find_var_count(sample_alg)],
		itr = 100, burn_in = Int(round(itr*sample_alg[:n_grp]*0.2)),
		chain_type=:default, progress = true
	) where {T <: Distribution}

```

### `sample_alg` parameter configuration

`sample_alg` contains the information regarding parameter groups, mapping MCMC sampling algorithms defined in `alg` parameter for each group, independent and dependent sampling. The `sample_alg` should be defined as a dictionary with certain keys and values. One example is shown below:

```julia
alg = [MH(), adHMC()]

sample_alg = Dict(
	:n_grp => 2,
	1 => Dict(
		:type => :ind,
		:n_vars => 2,
		1 => Dict(
			:proposal => MvNormal(zeros(2),1.0),
			:n_eles => 2,
			:alg => 1
		),
		2 => Dict(
			:proposal => Normal(0.0,1.0),
			:n_eles => 1,
			:alg => 1
		)
	),
	2 => Dict(
		:type => :dep,
		:n_vars => 2,
		:alg => 2,
		1 => Dict(
			:proposal => MvNormal(zeros(3),1.0),
			:n_eles => 3
		),
		2 => Dict(
			:proposal => Normal(0.0,1.0),
			:n_eles => 1
		)
	)
)
```
- `:n_grp` key defines the number of parameter groups, and in this example, two groups exist. Then each group is defined keys with continuous numbers, here 1 and 2.

Each group information is again stored in a dictionary, and it also contains many keys. Then each parameter is defined keys with continuous numbers, here 1 and 2.

- The key `:type` is to select parallel sampling of parameters or for group sampling, and it has two options `:ind` and `:dep`. 
	- The value `:ind` chooses parallel sampling of group parameters. Each parameter in this group is sampled independently by selecting previous values. Then all group parameters will be joined together to form the next sample. 
	- If `:type` is `:dep`, those group parameter sampled together to generate the next sample. 
- `n_vars` defines the number of parameters in each group. 
- `:alg` contains the index of the MCMC sampling algorithm defined in the `alg` parameter. If `:type` value is `:ind` then, it is possible to sample each parameter using different samplers. So, it is not required to define `:alg` key in `:ind` groups.

Each parameter information is again stored in a dictionary, and it also contains many keys. 
- `:proposal` : Proposal distribution of that parameter. It is mandatory for `MH` based sampling; however, it is not required for `adHMC` or `adNUTS` based sampling.

- `:n_eles`: Number of elements in each parameter.

- `:alg`: If group `:type` is `:ind`, then we have to define MCMC sampling algorithm with each parameter with `:alg` key.

