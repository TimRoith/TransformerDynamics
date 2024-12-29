import numpy as np
# from tfdy.utils import integral_scalar_prod, sph2cart, pol2cart

# #%%

# class energy:
#     def __init__(self, D, num_int_eval=500):
#         self.D  = D
#         self.Xp = np.linspace(-np.pi, np.pi, num_int_eval)
#         self.X  = pol2cart(self.Xp)
#         self.num_int_eval = num_int_eval
     
#     def __call__(self, prob):
#         p = prob(self.Xp)
#         p = p[:,None] * p[None,:]
#         y = (self.D@self.X.T).T
#         sp = np.exp(np.inner(self.X[:,None,:], y[None,...])).squeeze()
#         return (sp * p).sum()
    
    
    
#%%
from pysr import PySRRegressor

num_eps = 5
num_int_eval = 100
X = np.zeros((num_int_eval * num_eps, 2))

for i in range(num_eps):
    X[(i*num_int_eval):((i+1)*num_int_eval), 0] = (
        np.linspace(-np.pi, np.pi, num_int_eval)
    )
    X[(i*num_int_eval):((i+1)*num_int_eval), -1] = i/num_eps


y = 0 * X[:, 0]

objective = """
function my_custom_objective(tree, dataset::Dataset{T,L}, options) where {T,L}
    (prediction, completion) = eval_tree_array(tree, dataset.X, options)
    
    s = size(dataset.X)
    nie = 100
    ne  = ceil(Int, s[2]/nie)
    out = 0
    
    
    for k in 1:ne
        e = dataset.X[2, k * nie]
        D = [(1 - e) 0; 0 1]
        
        pr = prediction[((k-1) * nie + 1):((k) * nie)]
        aa = sum(pr)
        pr = pr/aa
        out += -10 * sum(min.(0, pr)) # + (1-aa)^2
        pr = max.(0, pr)
        pr = pr/sum(pr)
        
        XX = [0.,0.]
        YY = [0.,0.]
        
        @fastmath @inbounds for i in 1:nie
            @fastmath @inbounds for j in 1:nie
                # XX = [cos(dataset.X[1, i]), sin(dataset.X[1, i])]
                # YY = [cos(dataset.X[1, j]), sin(dataset.X[1, j])]
                # out += exp(XX' * (D * YY)) * pr[i] * pr[j]
                           
                out += exp(
                    (cos(dataset.X[1, i]) * cos(dataset.X[1, j])) * (1 - e) + 
                    (sin(dataset.X[1, i]) * sin(dataset.X[1, j]))
                    ) * pr[i] * pr[j]
            end
        end
    end
    
    
    if !completion
        return L(Inf)
    end

    return out / length(prediction)
end
"""


model = PySRRegressor(
    niterations=100,  # < Increase me for better results
    binary_operators=["+", "*", '-', '/'],
    unary_operators=[
        "exp",
        "sin",
        "cos",
        "tan",
        "inv(x) = 1/x",
        "abs",
        "cosh",
        "sinh",
        "tanh",
        "atan",
        "asinh",
        "sqrt",
        "log",
        "acosh",
        "atanh_clip",
        "erf"
        # ^ Custom operator (julia syntax)
    ],
    turbo=False,
    extra_sympy_mappings={"inv": lambda x: 1 / x},
    # ^ Define operator for SymPy as well
    loss_function=objective,
)

#%%
model.fit(X,y)

#%%
def f(x):
    #return np.cos(X[:,0])**2
    return np.ones((X.shape[0]))

prediction = f(X)
prediction = prediction/np.sum(prediction)
npred = -10 * np.sum(np.minimum(0, prediction))
prediction = np.maximum(0, prediction)
prediction = prediction/np.sum(prediction)

e = X[0, 1]
D = np.array([[(1 - e), 0], [ 0, 1]])

s = X.shape

out = npred

for i in range(s[0]):
    for j in range(s[0]):
        XX = np.array([np.cos(X[i, 0]), np.sin(X[i, 0])])
        YY = np.array([np.cos(X[j, 0]), np.sin(X[j, 0])])
        g = np.exp(XX.T@(D@YY))
        out += g * prediction[i] * prediction[j]

out / len(prediction)
