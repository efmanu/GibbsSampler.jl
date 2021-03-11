function grp_wrapper_indvar(states, param_val, param_proposal, sample_alg, gx, 
	revt, p_loc, alg, i, logJoint)	
	bk_param_val = deepcopy(param_val)
	Threads.@threads for px in 1:sample_alg[gx][:n_vars]	
		n_p_loc	= p_loc + px - 1		
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
	end
	p_loc += sample_alg[gx][:n_vars]
	return p_loc, param_val
end

function grp_wrapper_depvar(states, param_val, param_proposal, sample_alg, gx, 
	revt, p_loc, alg, i, logJoint)
	proposal = deepcopy(param_proposal[p_loc:p_loc+sample_alg[gx][:n_vars]-1])
	function step_wrapper(new_param)
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
	out_sample = proposal_sampling(step_wrapper, initial_θ, proposal, alg[sample_alg[gx][:alg]])
	param_val[p_loc:p_loc+length(out_sample)-1] = deepcopy(out_sample)	
	p_loc += sample_alg[gx][:n_vars]
	return p_loc, param_val
end
function grp_wrapper_depsubgrp(states, param_val, param_proposal, sample_alg, gx, 
	revt, p_loc, alg, i, logJoint)
	gp_var_count = find_group_var_count(sample_alg, gx)					
	proposal = deepcopy(param_proposal[p_loc:p_loc+gp_var_count-1])
	function step_wrapper(new_param)
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
	out_sample = proposal_sampling(step_wrapper, initial_θ, proposal, alg[sample_alg[gx][:alg]])
	param_val[p_loc:p_loc+length(out_sample)-1] = deepcopy(out_sample)					
	p_loc += gp_var_count
	return p_loc, param_val
end
function grp_wrapper_indsubgrp(states, param_val, param_proposal, sample_alg, gx, 
	revt, p_loc, alg, i, logJoint)
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
	return p_loc, param_val
end