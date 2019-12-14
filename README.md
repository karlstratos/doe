# mmi-limit

## I(X,Y) = 106.29 vs ln(N) < 6.24

N=512; CUDA_VISIBLE_DEVICES=0 python main.py --dim 128 --rho 0.9 --N ${N} --steps 3000 --nruns 100 --pickle ~/mmi-limit-experiments/dim128_rho0.9_N${N}.p --cuda > ~/mmi-limit-experiments/log_dim128_rho0.9_N${N}.txt;

N=256; CUDA_VISIBLE_DEVICES=1 python main.py --dim 128 --rho 0.9 --N ${N} --steps 3000 --nruns 100 --pickle ~/mmi-limit-experiments/dim128_rho0.9_N${N}.p --cuda > ~/mmi-limit-experiments/log_dim128_rho0.9_N${N}.txt;

N=128; CUDA_VISIBLE_DEVICES=2 python main.py --dim 128 --rho 0.9 --N ${N} --steps 3000 --nruns 100 --pickle ~/mmi-limit-experiments/dim128_rho0.9_N${N}.p --cuda > ~/mmi-limit-experiments/log_dim128_rho0.9_N${N}.txt;


## I(X,Y) = 18.41 vs ln(N) < 6.24
N=512; CUDA_VISIBLE_DEVICES=3 python main.py --dim 128 --rho 0.5 --N ${N} --steps 3000 --nruns 100 --pickle ~/mmi-limit-experiments/dim128_rho0.5_N${N}.p --cuda > ~/mmi-limit-experiments/log_dim128_rho0.5_N${N}.txt;

N=256; CUDA_VISIBLE_DEVICES=4 python main.py --dim 128 --rho 0.5 --N ${N} --steps 3000 --nruns 100 --pickle ~/mmi-limit-experiments/dim128_rho0.5_N${N}.p --cuda > ~/mmi-limit-experiments/log_dim128_rho0.5_N${N}.txt;

N=128; CUDA_VISIBLE_DEVICES=5 python main.py --dim 128 --rho 0.5 --N ${N} --steps 3000 --nruns 100 --pickle ~/mmi-limit-experiments/dim128_rho0.5_N${N}.p --cuda > ~/mmi-limit-experiments/log_dim128_rho0.5_N${N}.txt;


## I(X,Y) = 4.14 vs ln(N) > 4.85
N=512; CUDA_VISIBLE_DEVICES=6 python main.py --dim 128 --rho 0.25 --N ${N} --steps 3000 --nruns 100 --pickle ~/mmi-limit-experiments/dim128_rho0.25_N${N}.p --cuda > ~/mmi-limit-experiments/log_dim128_rho0.25_N${N}.txt;

N=256; CUDA_VISIBLE_DEVICES=7 python main.py --dim 128 --rho 0.25 --N ${N} --steps 3000 --nruns 100 --pickle ~/mmi-limit-experiments/dim128_rho0.25_N${N}.p --cuda > ~/mmi-limit-experiments/log_dim128_rho0.25_N${N}.txt;

N=128; CUDA_VISIBLE_DEVICES=0 python main.py --dim 128 --rho 0.25 --N ${N} --steps 3000 --nruns 100 --pickle ~/mmi-limit-experiments/dim128_rho0.25_N${N}.p --cuda > ~/mmi-limit-experiments/log_dim128_rho0.25_N${N}.txt;


## Plotting
for R in 0.25 0.5 0.9; do for N in 64 128 256 512; do python plot.py ~/Desktop/mmi-limit-experiments/dim128_rho${R}_N${N}.p --figure ~/Desktop/mmi-limit-experiments/plot_dim128_rho${R}_N${N}.pdf; done; done