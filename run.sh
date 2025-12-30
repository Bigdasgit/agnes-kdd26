#FR buffer 
echo 3 > "/proc/sys/vm/drop_caches" 
python3 run_agnes.py --dataset Ogbn --dataset_path /home/nvme0/AGNES/ogbn1M_64bit/ --feature-dim 128 --num-workers 16


#python3 run_agnes.py --dataset even23 --dataset_path /home/nvme1/Sdata/EVEN23/ --feature-dim 128 --num-workers 16

#python3 run_ginex.py --dataset power23 --dataset_path /home/nvme1/Sdata/POWER23/ --feature-dim 128 --num-workers 16

: << "END"

END