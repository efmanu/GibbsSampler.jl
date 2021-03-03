Base.vec(x::Float64) = [x]
function forward_transform(x)
	return vec(x)
end

function reverse_transform(x)
	if (length(x) == 1) && (x isa Vector)
		return x[1]
	else
		return x
	end
end

function form_vector(param_val, idx, new_param)
	nw_param_val = [param_val[1:idx-1]..., new_param, param_val[idx+1:end]...]
	return nw_param_val
end

function extract_sample_alg(alg, sample_alg)
	len_s = length(sample_alg)
	val = Array{Any}(undef,len_s)
	for loc in 1:len_s
		if (!isassigned(sample_alg[loc],3)) && (alg[sample_alg[loc][1]] isa MH)
			throw("Error: MH sampler requires a proposal distribution")
		elseif ((alg[sample_alg[loc][1]] isa adHMC) || (alg[sample_alg[loc][1]] isa adNUTS)) && ((!isassigned(sample_alg[loc],3)))
			if sample_alg[loc][2] == 1
				val[loc] = Normal(0.0,1.0)
			else
				val[loc] = MvNormal(zeros(sample_alg[loc][2]),1.0)
			end			
		else
			val[loc] = sample_alg[loc][3]
		end
	end
	return val
end
function format_chain(states, burn_in, itr; chain_type=:default)
	chain =DataFrame();
	if(!isempty(states))
		lps = length(states["itr_1"])
		param_names =[]
		for ln in 1:lps
			param_st = forward_transform(states["itr_1"][ln])
			for ps in 1:length(param_st)
				push!(param_names,"param[$(ln)][$(ps)]")
			end			
		end
		chain.var = param_names
		for i in (burn_in+1):itr
			all_val =[]
			for ln in 1:lps	
				param_st = forward_transform(states["itr_$(i)"][ln])
				push!(all_val, param_st...)
			end
			chain[!,Symbol((i-burn_in))] = all_val
		end
		if(chain_type == :default)
			return chain
		elseif(chain_type == :mcmcchain)
			return Chains(Matrix{Float64}(Array(chain[:,2:end])'), param_names)
		else
			throw("Error: Chain type not found")
		end
	end
	return chain
end
function reform_data(samples, initial_θ)
	if length(samples) == 1
		return samples[1]
	else
		single_vec = []
		start = 1
		for (idx,val) in enumerate(initial_θ)
			len = length(val)
			if len == 1
				new_val = samples[start]
			else
				new_val = reshape(samples[start:start+len-1], size(initial_θ[idx]))
			end		
			push!(single_vec, new_val)
			start += len
		end
		return single_vec
	end
end
function find_group_var_count(sample_alg, gx)
	g = 0
	if  (haskey(sample_alg[gx], :n_sub_grp))
		for sgx in 1:sample_alg[gx][:n_sub_grp]
			g += sample_alg[gx][sgx][:n_vars]
		end
	elseif (haskey(sample_alg[gx], :n_vars))
		g += sample_alg[gx][:n_vars]
	else
		throw("Error: No subgroups or no varibles found")
	end	
	return g
end
function find_var_count(sample_alg)
	g = 0
	for gx in 1:sample_alg[:n_grp]
		if  (haskey(sample_alg[gx], :n_sub_grp))
			for sgx in 1:sample_alg[gx][:n_sub_grp]
				g += sample_alg[gx][sgx][:n_vars]
			end
		elseif (haskey(sample_alg[gx], :n_vars))
			g += sample_alg[gx][:n_vars]
		else
			throw("Error: No subgroups or no varibles found")
		end	
	end
	return g
end
function generate_ini_paramval(sample_alg)
	param_val = []
	param_proposal =[]
	#foreach groups
	for gx in 1:sample_alg[:n_grp]
		if  (haskey(sample_alg[gx], :n_sub_grp))
			for sgx in 1:sample_alg[gx][:n_sub_grp]
				for px in 1:sample_alg[gx][sgx][:n_vars]
					if haskey(sample_alg[gx][sgx][px], :proposal)
						push!(param_val, rand(sample_alg[gx][sgx][px][:proposal]))
						push!(param_proposal,sample_alg[gx][sgx][px][:proposal])
					else
						if sample_alg[gx][sgx][px][:n_eles] == 1
							push!(param_val, rand(Normal(0.0,1.0)))
							push!(param_proposal,Normal(0.0,1.0))
						else
							push!(param_val, rand(MvNormal(zeros(sample_alg[gx][sgx][px][:n_eles]),1.0)))
							push!(param_proposal,MvNormal(zeros(sample_alg[gx][sgx][px][:n_eles]),1.0))
						end
					end			
				end
			end
		elseif (haskey(sample_alg[gx], :n_vars))
			for px in 1:sample_alg[gx][:n_vars]
				if haskey(sample_alg[gx][px], :proposal)
					push!(param_val, rand(sample_alg[gx][px][:proposal]))
					push!(param_proposal,sample_alg[gx][px][:proposal])
				else
					if sample_alg[gx][px][:n_eles] == 1
						push!(param_val, rand(Normal(0.0,1.0)))
						push!(param_proposal,Normal(0.0,1.0))
					else
						push!(param_val, rand(MvNormal(zeros(sample_alg[gx][px][:n_eles]),1.0)))
						push!(param_proposal,MvNormal(zeros(sample_alg[gx][px][:n_eles]),1.0))
					end
				end			
			end
		else
			throw("Error: No subgroups or no varibles found")
		end
		
	end
	return param_val, param_proposal
end
function form_single_vec(param_val)
	single_vec = []
	for (idx,val) in enumerate(param_val)
		append!(single_vec, val...)
	end
	return single_vec
end
