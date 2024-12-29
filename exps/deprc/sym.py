import numpy as np

    
    
#%%
from pysr import PySRRegressor

num_int_eval = 400
X = np.zeros((num_int_eval, 1))
X[:,0] = np.linspace(-np.pi, np.pi, num_int_eval)


y = 0 * X[:, 0]

objective = """
function my_custom_objective(tree, dataset::Dataset{T,L}, options) where {T,L}

    out = 0
    for phi in range(0,stop=2*pi,length=20)
        phi = Float32(phi)
        (prediction, completion) = eval_tree_array(tree, dataset.X .+ phi, options)
        pr = prediction
        aa = sum(pr)
        pr = pr/aa
        out += -10 * sum(min.(0, pr)) + 10* (1-aa)^2
        pr = max.(0, pr)
        pr = pr/sum(pr)
        
        d = 0.5
        theta0 = asin(1/sqrt(d^2 *cos(phi)^2 + sin(phi)^2) * sin(phi))
        
        if !completion
            return L(Inf)
        end
        
        
        s = size(dataset.X)
        integ = 0
            
        @fastmath @inbounds for j in 1:s[2]
            integ += ( 
                exp(cos(dataset.X[1, j])) * 
                sin(dataset.X[1, j] - theta0) * 
                pr[j]
            )
        end
            
        out += integ^2
    end
    return out
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
        "erf",
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
