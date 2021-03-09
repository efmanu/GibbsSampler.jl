"""
	gibbs(alg, sample_alg, logJoint::Function;  
		revt = [reverse_transform for _ in 1:find_var_count(sample_alg)],
		itr = 100, burn_in = Int(round(itr*sample_alg[:n_grp]*0.2)),
		chain_type=:default, progress = true
	) where {T <: Distribution}

To generate posterior samples using Gibbs sampling algorithm

# Inputs
- `alg` 			:MCMC algorithms based on structs defined in this package as vector eg: `alg = [MH()]`
- `sample_alg` 		:A dictionary maps `alg` to parameter groups index and it contains proposal distribution
if required by the sampling algorithm. 

Eg: `sample_alg = Dict(
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
)`

Here key is the paramter group and first index in the value (Vector) maps to `alg` index. Second index in the `sample_alg`
is the proposal distribution. This not mandatory.

- `logJoint` 		:Log PDF as a function
# Keyword Arguments
- `itr` 			:Number of iterations
- `burn_in` 		:Burn in from samples
- `chain_type`	:Sample chain type. default value is `:default`. Samples chains formated using `MCMCChain.jl`
by choosing `chain_type` as `:mcmcchain`
- `progress` 		:To show the sampling progress. Default value is `true`.
# Output
- `chn `			:Generated samples
"""
function gibbs(alg, sample_alg, logJoint::Function;  
	revt = [reverse_transform for _ in 1:find_var_count(sample_alg)],
	itr = 100, burn_in = Int(round(itr*sample_alg[:n_grp]*0.2)),
	chain_type=:default, progress = true
) where {T <: Distribution}
	if progress
		prog = Progress(itr, dt=0.5,
	             barglyphs=BarGlyphs('|','█', ['▁' ,'▂' ,'▃' ,'▄' ,'▅' ,'▆', '▇'],' ','|',),
	             barlen=50)
	end
	param_val, param_proposal = generate_ini_paramval(sample_alg)
	states = Dict(
		"itr_$(0)" => deepcopy(param_val)
	)
	for i in 1:itr
		if progress
			ProgressMeter.next!(prog, showvalues = [(:iter,i), (:samples, param_val)])
		end	
		p_loc = 1 #varibale count per iteration
		for gx = 1:sample_alg[:n_grp]
			if  (haskey(sample_alg[gx], :n_sub_grp))
				if sample_alg[gx][:type] == :ind
					bk_param_val = deepcopy(param_val)
					for sgx = 1:sample_alg[gx][:n_sub_grp]						
						if sample_alg[gx][sgx][:type] == :ind
							for px in 1:sample_alg[gx][sgx][:n_vars]
								function step_wrapper_sgindvar(new_param)
									nw_param_val =[]
									if alg[sample_alg[gx][sgx][px][:alg]] isa MH
										nw_param_val =deepcopy([bk_param_val[1:p_loc-1]...,new_param, bk_param_val[p_loc+1:end]...])
									else							
										nw_param_val = deepcopy([bk_param_val[1:p_loc-1]...,revt[p_loc](new_param), bk_param_val[p_loc+1:end]...])
									end
									return logJoint(nw_param_val)
								end	
								proposal = param_proposal[p_loc]
								initial_θ = states["itr_$(gx+(i-1)*sample_alg[:n_grp]-1)"][p_loc]
								out_sample = proposal_sampling(step_wrapper_sgindvar, initial_θ, proposal, alg[sample_alg[gx][sgx][px][:alg]])
								param_val[p_loc] = deepcopy(out_sample)
								p_loc +=1
							end
						elseif sample_alg[gx][sgx][:type] == :dep
							proposal = deepcopy(param_proposal[p_loc:p_loc+sample_alg[gx][sgx][:n_vars]-1])
							function step_wrapper_sgdepvar(new_param)
								nw_param_val =deepcopy(bk_param_val)
								if alg[sample_alg[gx][sgx][:alg]] isa MH						
									nw_param_val =[bk_param_val[1:p_loc-1]...,new_param..., bk_param_val[p_loc+length(new_param):end]...]
								else						
									ri_loc = 1
									for ri in 1:sample_alg[gx][sgx][:n_vars]
										nw_param_val[p_loc+ri-1] = deepcopy(revt[p_loc+ri-1](new_param[ri_loc:ri_loc+sample_alg[gx][sgx][ri][:n_eles]-1]))
										ri_loc += sample_alg[gx][sgx][ri][:n_eles]
									end
								end
								return logJoint(nw_param_val)
							end
							#sample
							initial_θ = deepcopy(states["itr_$(gx+(i-1)*sample_alg[:n_grp]-1)"][p_loc:p_loc+sample_alg[gx][sgx][:n_vars]-1])
							out_sample = proposal_sampling(step_wrapper_sgdepvar, initial_θ, proposal, alg[sample_alg[gx][sgx][:alg]])
							param_val[p_loc:p_loc+length(out_sample)-1] = deepcopy(out_sample)
							p_loc += sample_alg[gx][sgx][:n_vars]
						else
							throw("Error: No type is found")
						end			
					end
					states["itr_$(gx+(i-1)*sample_alg[:n_grp])"] = deepcopy(param_val)
				elseif sample_alg[gx][:type] == :dep
					gp_var_count = find_group_var_count(sample_alg, gx)					
					proposal = deepcopy(param_proposal[p_loc:p_loc+gp_var_count-1])
					function step_wrapper_gpdep(new_param)
						nw_param_val =deepcopy(param_val)
						if alg[sample_alg[gx][:alg]] isa MH						
							nw_param_val =[param_val[1:p_loc-1]...,new_param..., param_val[p_loc+length(new_param):end]...]
						else				
							ri_loc = 1
							for rsi in 1:sample_alg[gx][:n_sub_grp]
								for rvi in 1:sample_alg[gx][rsi][:n_vars]
									ri = rvi + (rsi - 1)*sample_alg[gx][:n_sub_grp]									
									nw_param_val[p_loc+ri-1] = deepcopy(revt[p_loc+ri-1](new_param[ri_loc:ri_loc+sample_alg[gx][rsi][rvi][:n_eles]-1]))
									# @show nw_param_val[ri] param_val[ri] new_param[ri_loc:ri_loc+sample_alg[gx][rsi][rvi][:n_eles]-1]
									ri_loc += sample_alg[gx][rsi][rvi][:n_eles]
								end
							end
						end
						return logJoint(nw_param_val)
					end
					initial_θ = deepcopy(states["itr_$(gx+(i-1)*sample_alg[:n_grp]-1)"][p_loc:p_loc+gp_var_count-1])
					out_sample = proposal_sampling(step_wrapper_gpdep, initial_θ, proposal, alg[sample_alg[gx][:alg]])
					param_val[p_loc:p_loc+length(out_sample)-1] = deepcopy(out_sample)
					states["itr_$(gx+(i-1)*sample_alg[:n_grp])"] = deepcopy(param_val)
					p_loc += gp_var_count
				else
					throw("Error: No type found")
				end	
			elseif (haskey(sample_alg[gx], :n_vars))
				if sample_alg[gx][:type] == :ind			

					bk_param_val = deepcopy(param_val)
					Threads.@threads for px in 1:sample_alg[gx][:n_vars]
						try								
							local n_p_loc	= p_loc + px - 1
							function step_wrapper(new_param)
								nw_param_val =[]
								if alg[sample_alg[gx][px][:alg]] isa MH
									nw_param_val =deepcopy([bk_param_val[1:n_p_loc-1]...,new_param, bk_param_val[n_p_loc+1:end]...])
								else							
									nw_param_val = deepcopy([bk_param_val[1:n_p_loc-1]...,revt[n_p_loc](new_param), bk_param_val[n_p_loc+1:end]...])
								end
								return logJoint(nw_param_val)
							end
							initial_θ = states["itr_$(gx+(i-1)*sample_alg[:n_grp]-1)"][n_p_loc]
							proposal = param_proposal[n_p_loc]
							out_sample = proposal_sampling(step_wrapper, initial_θ, proposal, alg[sample_alg[gx][px][:alg]])
							param_val[n_p_loc] = deepcopy(out_sample)
						catch e
				           @show e 
				       end			
						# p_loc +=1
					end	
					p_loc += sample_alg[gx][:n_vars]

					states["itr_$(gx+(i-1)*sample_alg[:n_grp])"] = deepcopy(param_val)							
				elseif sample_alg[gx][:type] == :dep
					# @show gx param_val
					proposal = deepcopy(param_proposal[p_loc:p_loc+sample_alg[gx][:n_vars]-1])
					function step_wrapperdep(new_param)
						nw_param_val =deepcopy(param_val)
						if alg[sample_alg[gx][:alg]] isa MH						
							nw_param_val =[param_val[1:p_loc-1]...,new_param..., param_val[p_loc+length(new_param):end]...]
						else						
							# nw_param_val = copy(param_val)
							ri_loc = 1
							for ri in 1:sample_alg[gx][:n_vars]
								nw_param_val[p_loc+ri-1] = deepcopy(revt[p_loc+ri-1](new_param[ri_loc:ri_loc+sample_alg[gx][ri][:n_eles]-1]))
								ri_loc += sample_alg[gx][ri][:n_eles]
							end
						end
						return logJoint(nw_param_val)
					end
					#sample
					initial_θ = deepcopy(states["itr_$(gx+(i-1)*sample_alg[:n_grp]-1)"][p_loc:p_loc+sample_alg[gx][:n_vars]-1])
					# @show initial_θ
					out_sample = proposal_sampling(step_wrapperdep, initial_θ, proposal, alg[sample_alg[gx][:alg]])
					param_val[p_loc:p_loc+length(out_sample)-1] = deepcopy(out_sample)
					states["itr_$(gx+(i-1)*sample_alg[:n_grp])"] = deepcopy(param_val)
					p_loc += sample_alg[gx][:n_vars]
				else
					throw("Error: No type found")
				end	
			else
				throw("Error: No subgroups or no varibles found")
			end

		end
	end
	if progress
		ProgressMeter.finish!(prog)
	end
	delete!(states, "itr_0")
	return format_chain(states, burn_in, itr*sample_alg[:n_grp], chain_type=chain_type)
end

function proposal_sampling(step_wrapper::Function, initial_θ,
 proposalDist, sample_alg::MH)
	model = AdvancedMH.DensityModel(step_wrapper)
	spl = AdvancedMH.RWMH(proposalDist)
	chain = AdvancedMH.sample(model, spl, sample_alg.n_samples; init_params = initial_θ)
	return chain[end].params
end
function proposal_sampling(step_wrapper::Function, initial_θ,
 proposalDist, sample_alg::adHMC)
	initial_nθ = copy(Array{Float64}(form_single_vec(initial_θ)))
	D = length(initial_nθ)
	ℓπ_grad(θ) = return (step_wrapper(θ), sample_alg.backend.gradient(step_wrapper, θ))	
	metric = DiagEuclideanMetric(D)
	hamiltonian = Hamiltonian(metric, step_wrapper, ℓπ_grad)
	initial_ϵ = find_good_stepsize(hamiltonian, initial_nθ)
	integrator = Leapfrog(initial_ϵ)
	proposal = AdvancedHMC.StaticTrajectory(integrator, 1)
	adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))
	samples, stats = AdvancedHMC.sample(hamiltonian, proposal, initial_nθ, sample_alg.n_samples, adaptor, sample_alg.n_adapts, verbose =false)
	return reform_data(samples[end], initial_θ)
 end
function proposal_sampling(step_wrapper::Function, initial_θ,
	proposalDist, sample_alg::adNUTS)
	initial_nθ = copy(Array{Float64}(form_single_vec(initial_θ)))
	D = length(initial_nθ)
	ℓπ_grad(θ) = return (step_wrapper(θ),sample_alg.backend.gradient(step_wrapper, θ))	
	metric = DiagEuclideanMetric(D)
	hamiltonian = Hamiltonian(metric, step_wrapper, ℓπ_grad)
	initial_ϵ = find_good_stepsize(hamiltonian, initial_nθ)
	integrator = Leapfrog(initial_ϵ)
	proposal =  AdvancedHMC.NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator)
	adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))
	samples, stats = AdvancedHMC.sample(hamiltonian, proposal, initial_nθ, sample_alg.n_samples, adaptor, sample_alg.n_adapts, verbose =false)
	return reform_data(samples[end], initial_θ)
end





