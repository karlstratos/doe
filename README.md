# mmi-limit


## Large MI=106.29, small N=64, eval on 10N=640 (previous setting)

R=3000; do CUDA_VISIBLE_DEVICES=0 python main.py --dim 128 --rho 0.9 --N 64 --c 10 --nruns 100 --steps ${R} --figname ~/mmi-limit-experiments/dim128_rho0.9_N64_steps${R}_nruns100_new --cuda > ~/mmi-limit-experiments/log_dim128_rho0.9_N64_steps${R}_nruns100_new.txt;
