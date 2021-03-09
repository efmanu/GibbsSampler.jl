using Test
using AdvancedMH
using MCMCChains
using GibbsSampler
using Distributions


@test "gibbs" begin
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
test isapprox(mean(p3),mean(chn["param[2][1]"]), atol=0.5)
end