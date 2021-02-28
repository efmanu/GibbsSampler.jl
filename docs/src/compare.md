# Comparison with [AdvancedHMC.jl](https://github.com/TuringLang/AdvancedHMC.jl)

### Comparison 1

This example compares the mean of samples generated using **Gibbs** and **AdvancedMH** sampling methods. Only one group of parameter is considered with one parameter in this group. The proposal distribution of parameter is chosen as `MvNormal(zeros(2),1.0)` and sampled uisng `MH` sampler to sample independently.

```julia
#use packages
using GibbsSampler, Distributions
using AdvancedMH, MCMCChains

#define MCMC samplers

alg = [MH(), adHMC()]

#define variable groups and sampling methods
sample_alg = Dict(
	:n_grp => 1,
	1 => Dict(
		:type => :ind,
		:n_vars => 1,
		1 => Dict(
			:proposal => MvNormal(zeros(2),1.0),
			:n_eles => 2,
			:alg => 1
		)
	)
)
prior = [MvNormal([2.0,3.0],1.0)]
logJoint(params) = sum(logpdf.(prior, params))
#sample using gibbs sampler
chn = gibbs(alg, sample_alg, logJoint, itr = 10000, chain_type = :mcmcchain)


proposalmh = [MvNormal(zeros(2),1.0)]
model = DensityModel(logJoint)
spl = RWMH(proposalmh)
chain = sample(model, spl, 10000; chain_type=Vector{NamedTuple})

#show nmean of each elements
p1 = [chain[i][:param_1][1] for i in 1:length(chain)]
@show mean(p1) mean(chn["param[1][1]"])
```
```
mean(p1) = 2.0386819762387196
mean(chn["param[1][1]"]) = 1.9957525645926026
1.9957525645926026

```
```julia
p2 = [chain[i][:param_1][2] for i in 1:length(chain)]
@show mean(p2) mean(chn["param[1][2]"])
```
```
mean(p2) = 3.003012417905884
mean(chn["param[1][2]"]) = 3.0387702707202373
3.0387702707202373
```

### Comparison 2

This example compares the mean of samples generated using **Gibbs** and **AdvancedMH** sampling methods. Only one group of parameter is considered with two parameters. The proposal distribution of parameters are chosen as `MvNormal(zeros(2),1.0)`, `Normal(0.0,1.0)` respectively and sampled uisng `MH` sampler and sampled independently.

```julia
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
prior = [MvNormal([2.0,3.0],1.0), Normal(4.0, 1.0)]
logJoint(params) = sum(logpdf.(prior, params))
#sample using gibbs sampler
chn = gibbs(alg, sample_alg, logJoint, itr = 10000, chain_type = :mcmcchain)

proposalmh = [MvNormal(zeros(2),1.0), Normal(0.0,1.0)]
model = DensityModel(logJoint)
spl = RWMH(proposalmh)
chain = sample(model, spl, 10000; chain_type=Vector{NamedTuple})
p1 = [chain[i][:param_1][1] for i in 1:length(chain)]
@show mean(p1) mean(chn["param[1][1]"])

p2 = [chain[i][:param_1][2] for i in 1:length(chain)]
@show mean(p2) mean(chn["param[1][2]"])

p3 = [chain[i][:param_2] for i in 1:length(chain)]
@show mean(p3) mean(chn["param[2][1]"])
```

### Comparison 3

This example compares the mean of samples generated using **Gibbs** and **AdvancedMH** sampling methods. Only one group of parameter is considered with two parameters. The proposal distribution of parameters are chosen as `MvNormal(zeros(2),1.0)`, `Normal(0.0,1.0)` respectively and sampled uisng `MH` sampler and sampled together.

```julia

sample_alg = Dict(
	:n_grp => 1,
	1 => Dict(
		:type => :dep,
		:n_vars => 2,
		:alg => 2,
		1 => Dict(
			:proposal => MvNormal(zeros(2),1.0),
			:n_eles => 2
		),
		2 => Dict(
			:proposal => Normal(0.0,1.0),
			:n_eles => 1
		)
	)
)
prior = [MvNormal([2.0,3.0],1.0), Normal(4.0, 1.0)]
logJoint(params) = sum(logpdf.(prior, params))
#sample using gibbs sampler
chn = gibbs(alg, sample_alg, logJoint, itr = 10000, chain_type = :mcmcchain)

proposalmh = [MvNormal(zeros(2),1.0), Normal(0.0,1.0)]
model = DensityModel(logJoint)
spl = RWMH(proposalmh)
chain = sample(model, spl, 10000; chain_type=Vector{NamedTuple})

p1 = [chain[i][:param_1][1] for i in 1:length(chain)]
@show mean(p1) mean(chn["param[1][1]"])

p2 = [chain[i][:param_1][2] for i in 1:length(chain)]
@show mean(p2) mean(chn["param[1][2]"])

p3 = [chain[i][:param_2] for i in 1:length(chain)]
@show mean(p3) mean(chn["param[2][1]"])
```

### Comparison 4

This example compares the mean of samples generated using **Gibbs** and **AdvancedMH** sampling methods. Only two groups is considered with one parameter in each group. The proposal distribution of parameters are chosen as `MvNormal(zeros(2),1.0)`, `MvNormal(zeros(3),1.0)` respectively and sampled uisng `MH` sampler and sampled together.

```julia
sample_alg = Dict(
	:n_grp => 2,
	1 => Dict(
		:type => :ind,
		:n_vars => 1,
		1 => Dict(
			:proposal => MvNormal(zeros(2),1.0),
			:n_eles => 2,
			:alg => 1
		)
	),
	2 => Dict(
		:type => :ind,
		:n_vars => 1,
		1 => Dict(
			:proposal => MvNormal(zeros(3),1.0),
			:n_eles => 3,
			:alg => 1
		)
	)
)
prior = [MvNormal([2.0,3.0],1.0), MvNormal([4.0,3.0, 5.0],1.0)]
logJoint(params) = sum(logpdf.(prior, params))
#sample using gibbs sampler
chn = gibbs(alg, sample_alg, logJoint, itr = 10000, chain_type = :mcmcchain)

proposalmh = [MvNormal(zeros(2),1.0), MvNormal(zeros(3),1.0)]
model = DensityModel(logJoint)
spl = RWMH(proposalmh)
chain = sample(model, spl, 10000; chain_type=Vector{NamedTuple})

p1 = [chain[i][:param_1][1] for i in 1:length(chain)]
@show mean(p1) mean(chn["param[1][1]"])

p2 = [chain[i][:param_1][2] for i in 1:length(chain)]
@show mean(p2) mean(chn["param[1][2]"])

p3 = [chain[i][:param_2][1] for i in 1:length(chain)]
@show mean(p3) mean(chn["param[2][1]"])

p4 = [chain[i][:param_2][2] for i in 1:length(chain)]
@show mean(p4) mean(chn["param[2][2]"])

p5 = [chain[i][:param_2][3] for i in 1:length(chain)]
@show mean(p5) mean(chn["param[2][3]"])
```
