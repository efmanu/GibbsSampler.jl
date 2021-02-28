
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
			:proposal => Normal(1.0,1.0),
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
			:proposal => Normal(-1.0,1.0),
			:n_eles => 1
		)
	)
)
params = [rand(2), 2.8,rand(3),0.3]
prior = [MvNormal(zeros(2),1.0),Normal(2.0,3.0), MvNormal(zeros(3),1.0),Normal(-1.0,3.0)]
logJoint(params) = sum(logpdf.(prior, params))
param_val, param_proposal = generate_ini_paramval(sample_alg)
states = Dict(
	"itr_$(0)" => copy(param_val)
)

itr = 2
for i in 1:itr
	p_loc = 1 #varibale count per iteration
	for gx = 1:sample_alg[:n_grp]
		if sample_alg[gx][:type] == :ind
			bk_param_val = copy(param_val)
			for px in 1:sample_alg[gx][:n_vars]
				function step_wrapper(new_param)
					nw_param_val =[]
					if alg[sample_alg[gx][px][:alg]] isa MH
						nw_param_val =copy([bk_param_val[1:p_loc-1]...,new_param, bk_param_val[p_loc+1:end]...])
					else
						nw_param_val = copy([bk_param_val[1:p_loc-1]...,revt[p_loc](new_param), bk_param_val[p_loc+1:end]...])
					end
					return logJoint(nw_param_val)
				end
				initial_θ = states["itr_$(i-1)"][p_loc]
				proposal = param_proposal[p_loc]
				out_sample = proposal_sampling(step_wrapper, initial_θ, proposal, alg[sample_alg[gx][px][:alg]])
				param_val[p_loc] = copy(out_sample)
				p_loc +=1
			end	
			states["itr_$(gx+(i-1)*sample_alg[:n_grp])"] = copy(param_val)				
		elseif sample_alg[gx][:type] == :dep
			proposal = param_proposal[p_loc:p_loc+sample_alg[gx][:n_vars]-1]
			function step_wrapper(new_param)
				nw_param_val =[]
				if alg[sample_alg[gx][:alg]] isa MH
					nw_param_val =copy([param_val[1:p_loc-1]...,new_param..., param_val[p_loc+length(new_param):end]...])
				else
					nw_param_val = copy(param_val)
					ri_loc = 1
					for ri in 1:sample_alg[gx][:n_vars]
						nw_param_val[p_loc+ri-1] = copy(revt[p_loc+ri-1](new_param[ri_loc:ri_loc+sample_alg[gx][ri][:n_eles]-1]))
						ri_loc += sample_alg[gx][1][:n_eles]
					end
				end
				return logJoint(nw_param_val)
			end
			#sample
			initial_θ = copy(states["itr_$(i-1)"][p_loc:p_loc+sample_alg[gx][:n_vars]-1])
			out_sample = proposal_sampling(step_wrapper, initial_θ, proposal, alg[sample_alg[gx][:alg]])
			param_val[p_loc:p_loc+length(out_sample)-1] = copy(out_sample)
			states["itr_$(gx+(i-1)*sample_alg[:n_grp])"] = copy(param_val)
			p_loc += sample_alg[gx][:n_vars]
		else
			throw("Error: No type is found")
		end
	end
end

function form_single_vec(param_val)
	single_vec = []
	for (idx,val) in enumerate(param_val)
		append!(single_vec, val...)
	end
	return single_vec
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
	ℓπ_grad(θ) = return (step_wrapper(θ), al.backend.gradient(step_wrapper, θ))	
	metric = DiagEuclideanMetric(D)
	hamiltonian = Hamiltonian(metric, step_wrapper, ℓπ_grad)
	initial_ϵ = find_good_stepsize(hamiltonian, initial_nθ)
	integrator = Leapfrog(initial_ϵ)
	proposal = AdvancedHMC.StaticTrajectory(integrator, 1)
	adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))
	samples, stats = AdvancedHMC.sample(hamiltonian, proposal, initial_nθ, sample_alg.n_samples, adaptor, sample_alg.n_adapts, verbose =false)
	return reform_data(samples[end], initial_θ)
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
				new_val = final_sample[start]
			else
				new_val = reshape(final_sample[start:start+len-1], (3,))
			end		
			push!(single_vec, new_val)
			start += len
		end
		return single_vec
	end
end
function generate_ini_paramval(sample_alg)
	param_val = []
	param_proposal =[]
	for gx in 1:sample_alg[:n_grp]
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
	end
	return param_val, param_proposal
end
