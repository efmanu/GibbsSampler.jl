using Test
using GibbsSampler
using Distributions
using MHSampler
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
	chn = GibbsSampler.gibbs(proposal, logJoint;itr = 10000)

	chm = MHSampler.mhsample(proposalf, logJoint, itr = 10000)	
	@test isapprox(mean(Array(chn[1,2:end])),mean(Array(chm[1,2:end])), atol=0.2)
end