function gibbs(prior1::Distribution, proposal1::Distribution; 
	prior2 = nothing, proposal2 = nothing, model=nothing, output = nothing,
	itr = 100, burn_in = Int(round(itr*0.2))
)	

	if prior2 isa Nothing
		function logJoint(params)	
			logPrior= logpdf(prior1, params)
			logLikelihood = 0
			if !(model isa Nothing)
				logLikelihood = sum(logpdf.(Normal(model(params), 1.0), output))
			end
			return logPrior + logLikelihood
		end
		return MHSampler.mhsample(proposal1, logJoint, itr = itr, burn_in=burn_in)
	else
		states = Dict()
		prev_proposal_1 = rand(proposal1)
		prev_proposal_2 = rand(proposal2)
		if model isa Nothing
			function logJointm1(params)	
				logPrior= logpdf(prior1, params)
				return logPrior
			end
			param1 = MHSampler.mhsample(proposal1, logJointm1, itr = itr, burn_in=0)
			function logJointm2(params)	
				logPrior= logpdf(prior2, params)
				return logPrior
			end
			param2 = MHSampler.mhsample(proposal2, logJointm2, itr = itr, burn_in=0)
			all_params = hcat(Array(param1[1,2:end]),Array(param2[1,2:end]))
			[states["itr_$i"]= all_params[i,:] for i =1:length(all_params[:,1])]
			return MHSampler.data_formatting(states, burn_in, itr)
		else
			
			for i=1:itr
				function logJoint1(params)	
					logPrior= logpdf(prior1, params)
					logLikelihood = 0
					if !(model isa Nothing)
						logLikelihood = sum(logpdf.(Normal(model(params, prev_proposal_2), 1.0), output))
					end
					return logPrior + logLikelihood
				end

				current_proposal_1 = MHSampler.mhsample(proposal1, logJoint1, itr = 2, burn_in =1)[1,2]
				function logJoint2(params)	
					logPrior= logpdf(prior2, params)
					logLikelihood = 0
					if !(model isa Nothing)
						logLikelihood = sum(logpdf.(Normal(model(current_proposal_1, params), 1.0), output))
					end
					return logPrior + logLikelihood
				end
				
				current_proposal_2 = MHSampler.mhsample(proposal2, logJoint2, itr = 2, burn_in =1)[1,2]
				states["itr_$i"] = vcat(current_proposal_1, current_proposal_2)
				prev_proposal_1 = current_proposal_1
				prev_proposal_2 = current_proposal_2
			end
			return MHSampler.data_formatting(states, burn_in, itr)
		end
	end
end