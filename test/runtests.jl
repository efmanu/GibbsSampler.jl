using Test
using AdvancedMH
using MCMCChains
using GibbsSampler
using Distributions
@testset "gibbs_sampling" begin
	
	
	proposal = [Normal(2.0,3.0), Normal(3.0,3.0)]
	function proposalf() 
		return rand.(proposal)
	end
	priors = proposal

	function logJoint(params)	
		logPrior= sum(logpdf.(priors, params))
		return logPrior
	end

	# Construct a DensityModel.
	mdl = DensityModel(logJoint)

	# Set up our sampler with a joint multivariate Normal proposal.
	spl = RWMH(MvNormal([2.0,3.0],3.0))

	# Sample from the posterior.
	chm = sample(mdl, spl, 100000; param_names=["μ", "σ"], chain_type=Chains)
	chn = GibbsSampler.gibbs(proposal, logJoint;itr = 100000)
	@test isapprox(mean(Array(chn[1,2:end])),mean(chm[:μ]), atol=0.1)
end
@testset "gibbs_likelihood" begin

	proposal = [Normal(1.0,5.0), Normal(0.0,5.0)]
	function proposalf() 
		return rand.(proposal)
	end
	priors = proposal

	model(z) = z[1] + z[2]
	output = 5.0
	function logJoint1(params)	
		logPrior= sum(logpdf.(priors, params))
		logLikelihood = logpdf(Normal(model(params)), output)
		return logPrior + logLikelihood
	end

	mdl1 = DensityModel(logJoint1)

	# Set up our sampler with a joint multivariate Normal proposal.
	spl1 = RWMH(MvNormal([1.0,0.0],5.0))

	# Sample from the posterior.
	chm = sample(mdl1, spl1, 100000; param_names=["μ", "σ"], chain_type=Chains)
	chn = GibbsSampler.gibbs(proposal, logJoint1;itr = 100000)

	@show mean(Array(chn[1,2:end])) mean(chm[:μ])
	@test isapprox(mean(Array(chn[1,2:end])),mean(chm[:μ]), atol=0.5)

end