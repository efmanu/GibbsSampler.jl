##***One independent group with dependent and independent variables
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

##*** one group with dependent sub groups (2)
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

##*** 3 grpups...2 of them having sub groups one having no sub groups. One group sampled dependently, 
### other one of the subgroup samples dependently and the other one samples independetly

alg = [MH(), adHMC()]
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

####above with HMC******************************

##***One independent group with dependent and independent variables
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
				:alg => 2
			),
			#params
			2 => Dict(
				:proposal => Normal(0.0,1.0),
				:n_eles => 1,
				:alg => 2
			)
		),
		#sub grp
		2 => Dict(			
			:type => :dep,
			:n_vars => 2,
			:alg => 2,
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

##*** one group with dependent sub groups (2)
sample_alg = Dict(
	:n_grp => 1,
	1 => Dict(		
		:type => :dep,
		:n_sub_grp => 2,
		:alg => 2,		
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

##**************one group with no subgroups variables sampled independently

sample_alg = Dict(
	:n_grp => 2,
	1 => Dict(
		:type => :ind,
		:n_vars => 1,
		1 => Dict(
			:proposal => MvNormal(zeros(2),1.0),
			:n_eles => 2,
			:alg => 2
		)
	),
	2 => Dict(
		:type => :ind,
		:n_vars => 1,
		1 => Dict(
			:proposal => MvNormal(zeros(3),1.0),
			:n_eles => 3,
			:alg => 2
		)
	)
)
prior = [MvNormal([2.0,3.0],1.0), MvNormal([4.0,3.0, 5.0],1.0)]
logJoint(params) = sum(logpdf.(prior, params))
#sample using gibbs sampler
chn = gibbs(alg, sample_alg, logJoint, itr = 10000, chain_type = :mcmcchain)

##**************one group with no subgroups variables sampled together
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

##*** 3 grpups...2 of them having sub groups one having no sub groups. One group sampled dependently, 
### other one of the subgroup samples dependently and the other one samples independetly

alg = [MH(), adHMC()]
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
				:alg => 2
			),
			#params
			2 => Dict(
				:proposal => Normal(0.0,1.0),
				:n_eles => 1,
				:alg => 2
			)
		),
		#sub grp
		2 => Dict(			
			:type => :dep,
			:n_vars => 2,
			:alg => 2,
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
		:alg => 2,		
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
prior = [
	MvNormal([1.0,2.0],1.0), Normal(3.0,1.0), MvNormal([4.0,5.0],1.0), Normal(4.0,1.0), 
	Normal(3.0,1.0), MvNormal([2.0,1.0,5.0],1.0), Normal(4.0,1.0), Normal(3.0,1.0), 
	MvNormal([4.0,2.0,1.0],1.0), Normal(3.0,1.0)
]

logJoint(params) = sum(logpdf.(prior, params))
#sample using gibbs sampler
chn = gibbs(alg, sample_alg, logJoint, itr = 10000, chain_type = :mcmcchain)
