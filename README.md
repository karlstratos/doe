# mmi-limit

## Large MI=106.29

N=128; CUDA_VISIBLE_DEVICES=5 python main.py --N ${N} --steps 12000 --nruns 100 --figname ~/mmi-limit-experiments/final_N${N} --cuda > ~/mmi-limit-experiments/log_final_N${N}.txt;
N=256; CUDA_VISIBLE_DEVICES=6 python main.py --N ${N} --steps 6000 --nruns 100 --figname ~/mmi-limit-experiments/final_N${N} --cuda > ~/mmi-limit-experiments/log_final_N${N}.txt;
N=512; CUDA_VISIBLE_DEVICES=7 python main.py --N ${N} --steps 3000 --nruns 100 --figname ~/mmi-limit-experiments/final_N${N} --cuda > ~/mmi-limit-experiments/log_final_N${N}.txt;

## Small MI=6.04

N=128; CUDA_VISIBLE_DEVICES=2 python main.py --N ${N} --dim 64 --rho 0.3 --steps 12000 --nruns 100 --figname ~/mmi-limit-experiments/final_smallMI_N${N} --cuda > ~/mmi-limit-experiments/log_final_smallMI_N${N}.txt;
N=256; CUDA_VISIBLE_DEVICES=3 python main.py --N ${N} --dim 64 --rho 0.3 --steps 6000 --nruns 100 --figname ~/mmi-limit-experiments/final_smallMI_N${N} --cuda > ~/mmi-limit-experiments/log_final_smallMI_N${N}.txt;
N=512; CUDA_VISIBLE_DEVICES=4 python main.py --N ${N} --dim 64 --rho 0.3 --steps 3000 --nruns 100 --figname ~/mmi-limit-experiments/final_smallMI_N${N} --cuda > ~/mmi-limit-experiments/log_final_smallMI_N${N}.txt;
