# mmi-limit


## Large MI=106.29

N=64; R=3000; CUDA_VISIBLE_DEVICES=5 python main.py --N ${N} --steps ${R} --nruns 100 --figname ~/mmi-limit-experiments/N${N}_steps${R}_new --cuda > ~/mmi-limit-experiments/log_N${N}_steps${R}_new.txt;
N=128; R=3000; CUDA_VISIBLE_DEVICES=7 python main.py --N ${N} --steps ${R} --nruns 100 --figname ~/mmi-limit-experiments/N${N}_steps${R}_new --cuda > ~/mmi-limit-experiments/log_N${N}_steps${R}_new.txt;
N=256; R=3000; CUDA_VISIBLE_DEVICES=0 python main.py --N ${N} --steps ${R} --nruns 100 --figname ~/mmi-limit-experiments/N${N}_steps${R}_new --cuda > ~/mmi-limit-experiments/log_N${N}_steps${R}_new.txt;
N=512; R=3000; CUDA_VISIBLE_DEVICES=0 python main.py --N ${N} --steps ${R} --nruns 100 --figname ~/mmi-limit-experiments/N${N}_steps${R}_new --cuda > ~/mmi-limit-experiments/log_N${N}_steps${R}_new.txt;

## Large MI=106.29, large N=2048 (ln(N)=7.62), eval on N=2048

N=2048; R=3000; CUDA_VISIBLE_DEVICES=0 python main.py --N ${N} --steps ${R} --nruns 100 --figname ~/mmi-limit-experiments/N${N}_steps${R}_new --cuda > ~/mmi-limit-experiments/log_N${N}_steps${R}_new.txt;


## Small MI=6.04, large N=1024 (ln(N)=6.93), eval on N=1024

N=1024; R=3000; CUDA_VISIBLE_DEVICES=3 python main.py --rho 0.3 --N ${N} --steps ${R} --nruns 100 --figname ~/mmi-limit-experiments/rho0.3_N${N}_steps${R}_new --cuda > ~/mmi-limit-experiments/log_rho0.3_N${N}_steps${R}_new.txt;



## Large MI=106.29, small N=64 (ln(N)=4.16), eval on 10N=640 (earlier setting)

N=64; R=3000; CUDA_VISIBLE_DEVICES=0 python main.py --N ${N} --c 10 --steps ${R} --nruns 100 --figname ~/mmi-limit-experiments/N${N}_c10_steps${R}_new --cuda > ~/mmi-limit-experiments/log_N${N}_c10_steps${R}_new.txt;
