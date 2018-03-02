function parameters()
#Initializing the tuneable parameters
lambda = 0.065
N_f = 3
noIters=30
return (lambda,N_f,noIters)
end


function RecommendSys()
#(training_sparse,test_sparse)=readData()
(lambda,N_f,noIters)=parameters()

R=[5 5 2 0;2 0 3 5;0 5 0 3;3 0 0 5]
R_t=R'

(n_u,n_m)=size(R)
MM = rand(n_m,N_f-1)
FirstRow=zeros(Float64,n_m)
for i=1:n_m
    FirstRow[i]=mean(nonzeros(R[:,1]))
end
M = [FirstRow';MM']
LamI=lambda*eye(N_f)
(r,c,v)=findnz(R)
II=sparse(r,c,1)
locWtU=sum(II,1)
locWtM=sum(II,1)
U=zeros(n_u,N_f)

for i=1:noIters
    for u=1:n_u   	
        movies=find(R_t[:,u])  
        M_u=M[:,movies] 
        vector=M_u*full(R_t[movies,u])
        matrix=(M_u*M_u')+locWtU[u]*LamI
        x=matrix\vector
        U[u,:]=x
    end
    
    for m=1:n_m      
  	users=find(R[:,m])      
        U_m=U[users,:]
        vector=U_m'*full(R[users,m])
        matrix=(U_m'*U_m)+locWtM[m]*LamI
        x=matrix\vector
        M[:,m]=x
     end

end

println("U")
println(round(U,2))
println("M")
println(round(M,2))
predicted= U*M
println("Predicted Matrix")
println(round(predicted,0))
println("Original Matrix")
println(R)

end

