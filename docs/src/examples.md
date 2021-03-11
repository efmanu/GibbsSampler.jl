# Examples
This section contains different examples that describes the usage of [GibbsSampler.jl](https://github.com/efmanu/GibbsSampler.jl) package.

## Different MCMC Samplers for parameter sampling

The [GibbsSampler.jl](https://github.com/efmanu/GibbsSampler.jl) allows the use of **MH**, **HMC**, **NUTS** MCMC samplers with the help of [AdvancedMH.jl](https://github.com/TuringLang/AdvancedMH.jl) and [AdvancedHMC.jl](https://github.com/TuringLang/AdvancedHMC.jl) package. The `GibbsSampler.gibbs(...)` function has an input corresponds to `alg`, which decides the MCMC sampler availble based on structs defined by the package (Eg: `MH()`, `adHMC()`, `adNUTS()` etc.). This algorithm need to to be mapped to parameter groups using another dictionary input `sample_alg`. The key of `sample_alg` represents the parameter group and the value vector contains the index of the sampling algorithm defined in `alg` with proposal distribution.

### Use of AdvancedMH as MCMC sampler
The `MH()` struct defined with [GibbsSampler.jl](https://github.com/efmanu/GibbsSampler.jl) package is used to select MCMC sampler for each parameter in Gibbs sampling. 


#### Example 1
In this example, we defined variable `sample_alg` with a group with two sub groups. One of the subgroup sampled together with HMC sampler and other subgroup variables sampled in parallel using MH sampler.
```julia
#use packages
using GibbsSampler
using Distributions

#define MCMC samplers
alg = [MH(), adHMC()]

#define sample_alg parameter
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
#define prior distribution
prior = [MvNormal([1.0,2.0],1.0),Normal(2.0,1.0), MvNormal([2.0,4.0,3.0],1.0),Normal(-1.0,1.0)]

#define logjoint function
logJoint(params) = sum(logpdf.(prior, params))

#sample
chn = gibbs(alg, sample_alg, logJoint, itr = 10000, chain_type = :mcmcchain)
```

#### Example 2

In this example, we defined variable `sample_alg` with a group with two sub groups. One of the subgroup sampled together and other subgroup variables sampled in parallel.

```julia
alg = [MH(), adHMC()]
sample_alg = Dict(
	:n_grp => 1,
	#grp
	1 => Dict(		
		:type => :ind,
		:n_sub_grp => 2,		
		#sub grp
		1 => Dict(			
			:type => :ind,
			:n_vars => 2,
			#params
			1 => Dict(
				:proposal => MvNormal(zeros(2),1.0),
				:n_eles => 2,
				:alg => 1
			),
			#params
			2 => Dict(
				:proposal => Normal(0.0,1.0),
				:n_eles => 1,
				:alg => 1
			)
		),
		#sub grp
		2 => Dict(			
			:type => :dep,
			:n_vars => 2,
			:alg => 1,
			#params
			1 => Dict(
				:proposal => MvNormal(zeros(2),1.0),
				:n_eles => 2,			
			),
			#params
			2 => Dict(
				:proposal => Normal(0.0,1.0),
				:n_eles => 1
			)
		)
	)
)

prior = [MvNormal([2.0, 3.0],1.0), Normal(2.0,1.0), MvNormal([4.0,5.0],1.0), Normal(1.0,1.0)]
logJoint(params) = sum(logpdf.(prior, params))
#sample using gibbs sampler
chn = gibbs(alg, sample_alg, logJoint, itr = 10000, chain_type = :mcmcchain)
```
#### Example 3

In this example, we defined variable `sample_alg` with a group with two sub groups. Variables in both sub groups sampled together.

```julia
sample_alg = Dict(
	:n_grp => 1,
	1 => Dict(		
		:type => :dep,
		:n_sub_grp => 2,
		:alg => 1,		
		#sub grp
		1 => Dict(			
			:n_vars => 2,
			#params
			1 => Dict(
				:proposal => Normal(0.0,1.0),
				:n_eles => 1			
			),
			#params
			2 => Dict(
				:proposal => MvNormal(zeros(3),1.0),
				:n_eles => 3,
			)
		),
		#sub grp
		2 => Dict(			
			:n_vars => 2,
			#params
			1 => Dict(
				:proposal => Normal(0.0,1.0),
				:n_eles => 1,				
			),
			#params
			2 => Dict(
				:proposal => Normal(0.0,1.0),
				:n_eles => 1,
			)
		)
	)
)

prior = [Normal(3.0,1.0), MvNormal([1.0,2.0,3.0],1.0),Normal(5.0,1.0), Normal(2.0,1.0)]
logJoint(params) = sum(logpdf.(prior, params))
#sample using gibbs sampler
chn = gibbs(alg, sample_alg, logJoint, itr = 10000, chain_type = :mcmcchain)

```

#### Example 4

In this example, we defined variable `sample_alg` with 3 groups with details as follwos:
- Group 1: 2 Subgroups, sampled in parallel
	- Subgroup 1: Two variables, sampled in parallel
	- Subgroup 2: Two variables, sampled together
- Group 2: 2 Subgroups, sampled together
- Group 3; No subgroups, only two variables, sampled together

```julia
sample_alg = Dict(
	:n_grp => 3,
	#grp
	1 => Dict(		
		:type => :ind,
		:n_sub_grp => 2,		
		#sub grp
		1 => Dict(			
			:type => :ind,
			:n_vars => 2,
			#params
			1 => Dict(
				:proposal => MvNormal(zeros(2),1.0),
				:n_eles => 2,
				:alg => 1
			),
			#params
			2 => Dict(
				:proposal => Normal(0.0,1.0),
				:n_eles => 1,
				:alg => 1
			)
		),
		#sub grp
		2 => Dict(			
			:type => :dep,
			:n_vars => 2,
			:alg => 1,
			#params
			1 => Dict(
				:proposal => MvNormal(zeros(2),1.0),
				:n_eles => 2,			
			),
			#params
			2 => Dict(
				:proposal => Normal(0.0,1.0),
				:n_eles => 1
			)
		)
	),
	2 => Dict(		
		:type => :dep,
		:n_sub_grp => 2,
		:alg => 1,		
		#sub grp
		1 => Dict(			
			:n_vars => 2,
			#params
			1 => Dict(
				:proposal => Normal(0.0,1.0),
				:n_eles => 1			
			),
			#params
			2 => Dict(
				:proposal => MvNormal(zeros(3),1.0),
				:n_eles => 3,
			)
		),
		#sub grp
		2 => Dict(			
			:n_vars => 2,
			#params
			1 => Dict(
				:proposal => Normal(0.0,1.0),
				:n_eles => 1,				
			),
			#params
			2 => Dict(
				:proposal => Normal(0.0,1.0),
				:n_eles => 1,
			)
		)
	),
	3 => Dict(
		:type => :dep,
		:n_vars => 2,
		:alg => 1,
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
prior = [
	MvNormal([1.0,2.0],1.0), Normal(3.0,1.0), MvNormal([4.0,5.0],1.0), Normal(4.0,1.0), 
	Normal(3.0,1.0), MvNormal([2.0,1.0,5.0],1.0), Normal(4.0,1.0), Normal(3.0,1.0), 
	MvNormal([4.0,2.0,1.0],1.0), Normal(3.0,1.0)
]

logJoint(params) = sum(logpdf.(prior, params))
#sample using gibbs sampler
chn = gibbs(alg, sample_alg, logJoint, itr = 10000, chain_type = :mcmcchain)
```

#### Example 5
In this example, we defined variable `sample_alg` with a group with two sub groups. One of the subgroup sampled together with NUTS sampler and other subgroup variables sampled in parallel using MH sampler.
```julia
#use packages
using GibbsSampler
using Distributions

#define MCMC samplers
alg = [MH(), adHMC(), adNUTS()]

#define sample_alg parameter
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
		:alg => 3,
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
#define prior distribution
prior = [MvNormal([1.0,2.0],1.0),Normal(2.0,1.0), MvNormal([2.0,4.0,3.0],1.0),Normal(-1.0,1.0)]

#define logjoint function
logJoint(params) = sum(logpdf.(prior, params))

#sample
chn = gibbs(alg, sample_alg, logJoint, itr = 10000, chain_type = :mcmcchain)
```


#### Example 6
In this example, we defined variable `sample_alg` with a group with two sub groups. One of the subgroup sampled together with NUTS sampler with reverse differentiation backend.
```julia
#use packages
using GibbsSampler
using Distributions
using ReverseDiff

#define MCMC samplers
alg = [MH(), adHMC(), adNUTS(backend = ReverseDiff)]

#define sample_alg parameter
sample_alg = Dict(
	:n_grp => 1,
	1 => Dict(
		:type => :dep,
		:n_vars => 2,
		:alg => 3,
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
#define prior distribution
prior = [MvNormal([2.0,4.0,3.0],1.0),Normal(-1.0,1.0)]

#define logjoint function
function  logJoint(params)
	@show typeof(params[2]) size(params[2])
	lpdf = identity.(logpdf.(prior, params))
	return sum(lpdf)
end

#sample
chn = gibbs(alg, sample_alg, logJoint, itr = 10000, chain_type = :mcmcchain)
```

#### Example 7
In this example, we defined variable `sample_alg` ....
```julia
#use packages
using GibbsSampler
using Distributions
using ReverseDiff

#define MCMC samplers
alg = [MH(), adHMC(), adNUTS(backend = ReverseDiff)]
sample_alg = Dict(
	:n_grp => 1,
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
	)
)

#define prior distribution
prior = [MvNormal([2.0,3.0],1.0),Normal(-1.0,1.0)]

#define logjoint function
function logJoint(params)
	sumval = 0.0
	for jk in 1:length(params)
		sumval +=logpdf(prior[jk], params[jk])
	end
	return sumval
end

#sample
chn = gibbs(alg, sample_alg, logJoint, itr = 200, chain_type = :mcmcchain)
```