def pdf_guess(theta, f):
    evalp = torch.exp(-f * torch.cos(2 * theta))
    return evalp/torch.sum(evalp, axis=-1, keepdims=True)
theta = torch.linspace(-np.pi, np.pi, 1000)[None, :]
f = torch.nn.Parameter(torch.ones(10,1))
Ws = torch.tensor(Ws)

opt = torch.optim.Adam([f], lr=0.1)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=100, factor=0.5, verbose=True)

for i in range(10000):
    opt.zero_grad()
    loss = torch.linalg.norm(pdf_guess(theta, f) - Ws, ord='fro')
    loss.backward()
    opt.step()
    sched.step(loss)
    if i % 100 == 0:
        print(loss.item())
#%%
plt.figure()
plt.plot(ds, f.detach().numpy())
plt.plot(ds, ff)
# %%
p = np.polyfit(ds[1:]-1, f[1:].detach().numpy().ravel(), 2)
p = [1/5, np.e/2, 0]
ff = np.polyval(p, ds - 1)
# %%
