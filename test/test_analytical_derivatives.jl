module TestDerivatives
using Dynare #Use analytical_derivatives branch
using LinearAlgebra
using Dynare.FastLapackInterface
using Test
using SparseArrays
using SuiteSparse
using GeneralizedSylvesterSolver

#Model
context = @dynare "models/analytical_derivatives/fs2000_sa.mod" "params_derivs_order=1" "notmpterms";
model = context.models[1]
results = context.results.model_results[1]
wss = Dynare.StaticWs(context)
params = context.work.params
steadystate = results.trends.endogenous_steady_state
endogenous = repeat(steadystate, 3)
exogenous = results.trends.exogenous_steady_state 

#Get StaticJacobian
df_dx = Dynare.get_static_jacobian!(wss, params, steadystate, exogenous, model) 

#Get StaticJacobianParams
(df_dp, gp) = Dynare.DFunctions.SparseStaticParametersDerivatives!(steadystate, exogenous, params)
 

#1. Get Jacobian matrix using the inverse of df_dx
struct DerivativesWorkspace
    dense_df_dx::Matrix{Float64}
    df_dx_inv::Matrix{Float64}
    result::Matrix{Float64}
end

function init_derivatives_workspace(n::Int, m::Int)
    dense_df_dx = Matrix{Float64}(undef, n, n)
    df_dx_inv = Matrix{Float64}(undef, n, n)
    result = Matrix{Float64}(undef, n, m)
    return DerivativesWorkspace(dense_df_dx, df_dx_inv, result)
end

function Derivatives(workspace::DerivativesWorkspace, df_dx, df_dp)
    dense_df_dx = workspace.dense_df_dx
    copyto!(dense_df_dx, df_dx)

    if det(dense_df_dx) == 0
        error("The matrix is not invertible")
    end

    df_dx_inv = workspace.df_dx_inv
    df_dx_inv .= dense_df_dx \ I

    result = workspace.result
    return mul!(result, -df_dx_inv, df_dp)
end


#2. Get Jacobian matrix using FastLapackInterface (dense matrix)
struct Derivatives2Workspace
    dense_df_dx::Matrix{Float64}
    LU_df_dx::LU{Float64, Matrix{Float64}} #not used yet
    dx_dp::Matrix{Float64}
end


function init_derivatives2_workspace(n::Int, m::Int)
    dense_df_dx = Matrix{Float64}(undef, n, n)
    LU_df_dx = lu(Matrix{Float64}(I, n, n)) #not used yet
    dx_dp = Matrix{Float64}(undef, n, m)
    return Derivatives2Workspace(dense_df_dx, LU_df_dx, dx_dp)
end


function Derivatives2(workspace::Derivatives2Workspace, df_dx, df_dp)
    dense_df_dx = workspace.dense_df_dx
    copyto!(dense_df_dx, df_dx)

    LU_df_dx = LU(LAPACK.getrf!(LUWs(dense_df_dx), dense_df_dx)...)

    if any(diag(LU_df_dx.U) .== 0)
        error("The matrix is not invertible")
    end

    dx_dp = workspace.dx_dp
    copyto!(dx_dp, -(LU_df_dx \ df_dp))
    # ldiv!(-1.0, LU_df_dx, df_dp, dx_dp)

    return dx_dp
end

#3. Get Jacobian matrix using lu(df_dx) (sparse matrix)
struct Derivatives3Workspace
    # df_dx::SparseMatrixCSC{Float64, Int64}
    LU_df_dx::SuiteSparse.UMFPACK.UmfpackLU{Float64, Int64} #not used yet
    dx_dp::Matrix{Float64}
end

function init_derivatives3_workspace(n::Int, m::Int)
    LU_df_dx = SuiteSparse.UMFPACK.lu(sparse(Matrix{Float64}(I, n, n)))  #not used yet
    dx_dp = Matrix{Float64}(undef, n, m)
    return Derivatives3Workspace(LU_df_dx, dx_dp)
end

function Derivatives3(workspace::Derivatives3Workspace, df_dx, df_dp)
    # Perform Sparse LU decomposition
    # LU_df_dx = workspace.LU_df_dx
    LU_df_dx = lu(df_dx)
    # copyto!(LU_df_dx.L, lu_df_dx.L)
    # copyto!(LU_df_dx.U, lu_df_dx.U)

    if any(diag(LU_df_dx.U) .== 0)
        error("The matrix is not invertible")
    end

    # Get dx_dp
    dx_dp = workspace.dx_dp
    copyto!(dx_dp, -(LU_df_dx \ df_dp))
    # ldiv!(-1.0, LU_df_dx, df_dp, dx_dp)

    return dx_dp
end


#4. Get Jacobian matrix using ldiv! (dense matrix)
struct Derivatives4Workspace
    dense_df_dx::Matrix{Float64}
    LU_df_dx::LU{Float64, Matrix{Float64}}
    dx_dp::Matrix{Float64}
end

function init_derivatives4_workspace(n::Int, m::Int)
    dense_df_dx = Matrix{Float64}(undef, n, n)
    LU_df_dx = lu(Matrix{Float64}(I, n, n))
    dx_dp = Matrix{Float64}(undef, n, m)
    return Derivatives4Workspace(dense_df_dx, LU_df_dx, dx_dp)
end

function Derivatives4(workspace::Derivatives4Workspace, df_dx, df_dp)
    dense_df_dx = workspace.dense_df_dx
    copyto!(dense_df_dx, Matrix(df_dx))
    
    # LU_df_dx = workspace.LU_df_dx
    LU_df_dx = lu(dense_df_dx)
    # copyto!(LU_df_dx.L, lu_df_dx.L)
    # copyto!(LU_df_dx.U, lu_df_dx.U)

    if any(diag(LU_df_dx.U) .== 0)
        error("The matrix is not invertible")
    end

    dx_dp = workspace.dx_dp
    n, m = size(df_dp)

    for i in 1:m
        dp_col = view(df_dp, :, i)
        sol_col = view(dx_dp, :, i)
        ldiv!(sol_col, LU_df_dx, dp_col)
        sol_col .*= -1
    end

    return dx_dp
end


# Initialize
n = model.endogenous_nbr
m = model.parameter_nbr
workspace1 = init_derivatives_workspace(n, m)
workspace2 = init_derivatives2_workspace(n, m)
workspace3 = init_derivatives3_workspace(n, m)
workspace4 = init_derivatives4_workspace(n, m)

@time sol1 = Derivatives(workspace1, df_dx, df_dp);
@time sol2 = Derivatives2(workspace2, df_dx, df_dp);
@time sol3 = Derivatives3(workspace3, df_dx, df_dp);
@time sol4 = Derivatives4(workspace4, df_dx, df_dp);

# @show Derivatives(df_dx, df_dp);
@test sol1 ≈ sol2
@test sol1 ≈ sol3
@test sol1 ≈ sol4 

###
## Derivatives of first order solution
### 

wsd = Dynare.DynamicWs(context, order=2)
jacobian = Dynare.get_dynamic_jacobian!(
    wsd,
    params,
    endogenous,
    exogenous,
    steadystate,
    model,
    2,
)

n = model.endogenous_nbr
A = jacobian[:, 2*n .+ (1:n)]
B = jacobian[:, n .+ (1:n)]
C = jacobian[: , 1:n]

#Get DynamicJacobianParams
lli = model.lead_lag_incidence
endogenous3 = repeat(steadystate, 3)
endogenous = endogenous3[findall(!iszero, vec(lli'))]
(dr_dp, gp) = Dynare.DFunctions.SparseDynamicParametersDerivatives!(endogenous, exogenous, params, steadystate, 2, [], [])

dA_dp = zeros(n, n, m)
dB_dp = zeros(n, n, m)
dC_dp = zeros(n, n, m)

# derivatives of A, B, C with respect to parameter
k1 = findall(!iszero, lli[3, :])
k2 = lli[3, k1]
dA_dp[:, k1, :] .= gp[:, k2, :]
k1 = findall(!iszero, lli[2, :])
k2 = lli[2, k1]
dB_dp[:, k1, :] .= gp[:, k2, :]
k1 = findall(!iszero, lli[1, :])
k2 = lli[1, k1]
dC_dp[:, k1, :] .= gp[:, k2, :]

localapproximation!(irf=0, display=false);

# solution to UQME
X = zeros(n, n)
# set nonzero columns
X[:, k1] .= context.results.model_results[1].linearrationalexpectations.g1_1

#Generalized Sylvester: ax + bxc = d
a = A*X + B
b = Matrix(A)
c = X
X2 = X*X
d = zeros(n, n, m)

a_orig = copy(a)
b_orig = copy(b)
c_orig = copy(c)
d_orig = copy(d)

order=1
GSws = GeneralizedSylvesterWs(n,n,n,order)
#Solve UQME using generalized_sylvester_solver!
for i in 1:m
    @views begin
        mul!(d[:,:,i], dA_dp[:,:,i], X2)
        mul!(d[:,:,i], dB_dp[:,:,i], X, true, true) 
        d[:,:,i] .+= dC_dp[:,:,i]
        d_orig[:, :, i] .= d[:, :, i]
    end
    #    generalized_sylvester_solver!(a, b, c, d[:,:,i], order, GSws)
    @views d[:, :, i] .= reshape((kron(I(n), a) + kron(c', b))\vec(d[:, :, i]), n, n) 
end
#using Serialization
#Serialization.serialize("test.jls", (a_orig, b_orig, c_orig, d_orig))

#Test 1
for i in 1:m
    @test a_orig*d[:,:,i] + b_orig*d[:,:,i]*c_orig ≈ d_orig[:,:,i] #Fail
end

#Test 2
for i in 1:m
    @test d[:,:,i] ≈ reshape((kron(I(n^order),a_orig) + kron(c_orig',b_orig))\vec(d_orig[:,:,i]),n,n^order) #Fail
end

#Test 3
using FiniteDifferences

params0 = copy(context.work.params)

params1 = copy(params0)
h = 0.00001
params1[5] += h

function funX(param)
    context.work.params[5] = param
    localapproximation!(irf=0, display=false);
    return copy(context.results.model_results[1].linearrationalexpectations.g1_1)
end

fd = central_fdm(5, 1)
dX_dz_tuple = FiniteDifferences.jacobian(fd, funX, params[5])
dX_dz_matrix = reshape(dX_dz_tuple[1], n, 4)
#dX_dz = permutedims(reshape(dX_dz_matrix, m, n, n), (2, 3, 1))


@show "analytical"
display(d[:, :, 5])
@show "finite diff"
display(dX_dz_matrix)

copy!(context.work.params, params0)
@show "manual"
display((funX(params1[5]) - funX(params0[5])) ./ h)

#@test d ≈ dX_dz_matrix #Fail

#second order derivatives w.r. dynamic endogenous 
values = zeros(size(model.dynamic_g2_sparse_indices, 1))
Dynare.get_dynamic_derivatives2!(
    wsd,
    params,
    endogenous3,
    exogenous,
    steadystate,
    values,
    model,
)
k = vcat(
    model.i_bkwrd_b,
    model.endogenous_nbr .+ model.i_current,
    2*model.endogenous_nbr .+ model.i_fwrd_b,
    3*model.endogenous_nbr .+ collect(1:model.exogenous_nbr)
)
n = 3*model.endogenous_nbr + model.exogenous_nbr
nc = model.n_bkwrd + 2*model.n_both + model.n_current + model.n_fwrd + model.exogenous_nbr
F1 = zeros(model.endogenous_nbr, nc)
@views F1 .= wsd.derivatives[1][:, k]
kk = vec(reshape(1:n*n, n, n)[k, k])

F2 = Matrix(wsd.derivatives[2])[:, kk]
k = collect(5:19)
dB = zeros(15, 15)
for i=1:15
    dB .+= sol1[i, 1] .* F2[:, k]
    k .+= 24
end
@show "dB"
display(dB)
display(dB_dp[:, :, 1])

function funX1(wsd, params, endogenous, exogenous, steadystate, model)
    jacobian = Dynare.get_dynamic_jacobian!(
    wsd,
    params,
    endogenous,
    exogenous,
    steadystate,
    model,
    2,
    )
    return copy(jacobian)
end

function D1(wsd, params, endogenous, exogenous, steadystate, model)
    _params = copy(params)
    h = 0.0001
    _params[1] += h 
    J2 = funX1(wsd, _params, endogenous, exogenous, steadystate, model)
    J1 = funX1(wsd, params, endogenous, exogenous, steadystate, model)
    display(Matrix((J2[:,16:30] - J1[:,16:30])/h))
end

function D2(wsd, params, endogenous, exogenous, steadystate, model)
    _params = copy(params)
    h = 0.0001
    _params[1] += h
    
    J2 = funX1(wsd, _params, endogenous, exogenous, steadystate, model)
    J1 = funX1(wsd, params, endogenous, exogenous, steadystate, model)
    display(Matrix((J2[:,16:30] - J1[:,16:30])/h))
end

function D3(context)
    model = context.models[1]
    context1 = deepcopy(context)
    Dynare.compute_steady_state!(context1)
    ss0 = copy(context1.results.model_results[1].trends.endogenous_steady_state)
    h = 0.0001
    context1.work.params[1] += h
    Dynare.compute_steady_state!(context1)
    ss1 = copy(context1.results.model_results[1].trends.endogenous_steady_state)
    params = copy(context.work.params)
    exogenous = context.results.model_results[1].trends.exogenous_steady_state
    endogenous3 = repeat(ss1, 3)
    J2 = funX1(wsd, params, endogenous3, exogenous, ss1, model)
    endogenous3 = repeat(ss0, 3)
    J1 = funX1(wsd, params, endogenous3, exogenous, ss0, model)
    display(Matrix((J2[:,16:30] - J1[:,16:30])/h))

end
    
@show "D1"
D1(wsd, params, endogenous3, exogenous, steadystate, model)

@show "D3"
dss = zeros(15)
D3(context)

nothing




end # end module   

