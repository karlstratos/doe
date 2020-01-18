# Difference-of-Entropies (DoE) Estimator

This is a software accompaniment to [1]. It generates samples from a distribution with known mutual information (correlated Gaussians) and optimizes various estimators of mutual information. In particular, we compare our proposed estimator (DoE) with variational lower bounds discussed in [2]. If you find this code useful, please cite [1].

## 1. Training 

To optimize estimators with N=128 samples in each batch for 3000 steps, type

```bash
python main.py --dim 128 --rho 0.5 --N 128 --steps 3000 --nruns 100 --pickle dim128_rho0.5_N128.p --cuda 
```

This performs random grid search over 100 training sessions and stores the best hyperparameter configuration for each model in the pickle file. After optimization, it draws N fresh samples and computes final estimates using best configurations, for instance: 

```text
------------------------------------------------------------------------------------------------------------------------------------------------------
dv        :  10.27               Namespace(N=128, a='e', alpha=0.01, c=1, carry=0.5, clip=5, cuda=True, dim=128, hidden=256, init=0.1, layers=1, lr=0.001, nruns=100, pickle='/dresden/users/jl2529/mmi-limit-experiments/dim128_rho0.5_N128.p', rho=0.5, seed=22033, steps=3000)
mine      :   9.38               Namespace(N=128, a='e', alpha=0.5, c=1, carry=0.99, clip=5, cuda=True, dim=128, hidden=256, init=0.0, layers=1, lr=0.001, nruns=100, pickle='/dresden/users/jl2529/mmi-limit-experiments/dim128_rho0.5_N128.p', rho=0.5, seed=84396, steps=3000)
nwj       :   9.25               Namespace(N=128, a='e', alpha=0.5, c=1, carry=0.99, clip=5, cuda=True, dim=128, hidden=256, init=0.0, layers=1, lr=0.001, nruns=100, pickle='/dresden/users/jl2529/mmi-limit-experiments/dim128_rho0.5_N128.p', rho=0.5, seed=84396, steps=3000)
nwjjs     :   5.55               Namespace(N=128, a='e', alpha=0.99, c=1, carry=0.5, clip=1, cuda=True, dim=128, hidden=256, init=0.05, layers=1, lr=0.001, nruns=100, pickle='/dresden/users/jl2529/mmi-limit-experiments/dim128_rho0.5_N128.p', rho=0.5, seed=89750, steps=3000)
cpc       :   4.82               Namespace(N=128, a='e', alpha=0.01, c=1, carry=0.5, clip=5, cuda=True, dim=128, hidden=256, init=0.1, layers=1, lr=0.001, nruns=100, pickle='/dresden/users/jl2529/mmi-limit-experiments/dim128_rho0.5_N128.p', rho=0.5, seed=22033, steps=3000)
interpol  :   8.18               Namespace(N=128, a='e', alpha=0.01, c=1, carry=0.5, clip=5, cuda=True, dim=128, hidden=256, init=0.1, layers=1, lr=0.001, nruns=100, pickle='/dresden/users/jl2529/mmi-limit-experiments/dim128_rho0.5_N128.p', rho=0.5, seed=22033, steps=3000)
doe       :  18.38               Namespace(N=128, a='e', alpha=0.5, c=1, carry=0.99, clip=10, cuda=True, dim=128, hidden=128, init=0.05, layers=1, lr=0.0003, nruns=100, pickle='/dresden/users/jl2529/mmi-limit-experiments/dim128_rho0.5_N128.p', rho=0.5, seed=29162, steps=3000)
doe_l     :  18.42               Namespace(N=128, a='e', alpha=0.5, c=1, carry=0.99, clip=10, cuda=True, dim=128, hidden=128, init=0.05, layers=1, lr=0.0003, nruns=100, pickle='/dresden/users/jl2529/mmi-limit-experiments/dim128_rho0.5_N128.p', rho=0.5, seed=29162, steps=3000)
------------------------------------------------------------------------------------------------------------------------------------------------------
ln(128): 4.85
I(X,Y): 18.41
```

## 2. Plotting

You can visualize the best training session for each estimator by using the provided script like this: 

```bash
python plot.py dim128_rho0.5_N128.p --figure plot_dim128_rho0.5_N128.pdf 
```

The stored PDF file looks like [this](plot_dim128_rho0.5_N128.pdf).

### References

[1] [Formal Limitations on the Measurement of Mutual Information (McAllester and Stratos, AISTATS 2020)](http://karlstratos.com/publications/aistats20limit.pdf)

[2] [On Variational Bounds of Mutual Information (Poole et al., ICML 2019)](https://arxiv.org/abs/1905.06922)

