gpu=$1
x=$2
y=$3
ro=${4:-mean}
obj=${5-latency}

read -r lr dim1 dim2 dp lgtn lmlp nhead bs mn onorm eps minlr wd <<< $x
read -r cbo enc ch2 ch3 ch4 <<< $y
echo $lr, $dim1, $dim2, $dp, $lgtn, $lmlp, $nhead, $bs, $mn, $onorm, $eps, $minlr, $wd
echo  $cbo, $enc, $ch2, $ch3, $ch4
export PYTHONPATH="$PWD"
export TMPDIR=/tmp
loss=wmape

python examples/model/spark/1.train_variable_stages.py --ch1-type on --ch1-cbo $cbo --ch1-enc $enc --ch2 $ch2 --ch3 $ch3 --ch4 $ch4 \
--obj $obj --model-name $mn --nworkers 6 --bs $bs --epochs $eps --init-lr $lr --min-lr $minlr --weight-decay $wd \
--loss-type $loss --L-gtn $lgtn --L-mlp $lmlp --hidden-dim $dim1 --out-dim $dim1 --mlp-dim $dim2 --ch1-type-dim 8 \
--ch1-cbo-dim 4 --ch1-enc-dim 32 --n-heads $nhead --dropout2 $dp --ped 8 --gpu $gpu --out-norm $onorm --readout $ro


for dim1 in 16 32; do
  for dim2 in 64 128 256; do
    # L_mlp = 3
    bash x_run_glb.sh 0 "0.001 $dim1 $dim2 0.1 4 3 1 512 AVGMLP_GLB None 100 1e-5 1e-2" "off off on on on"
  done
done

for dim1 in 16 32; do
  for dim2 in 64 128 256; do
    # L_mlp = 3
    bash x_run_glb.sh 1 "0.003 $dim1 $dim2 0.1 4 3 1 512 AVGMLP_GLB None 100 1e-5 1e-2" "off off on on on"
  done
done

for dim1 in 16 32; do
  for dim2 in 64 128 256; do
    # L_mlp = 3
    bash x_run_glb.sh 0 "0.001 $dim1 $dim2 0.1 4 2 1 512 AVGMLP_GLB None 100 1e-5 1e-2" "off off on on on"
  done
done

for dim1 in 16 32; do
  for dim2 in 64 128 256; do
    # L_mlp = 3
    bash x_run_glb.sh 1 "0.003 $dim1 $dim2 0.1 4 2 1 512 AVGMLP_GLB None 100 1e-5 1e-2" "off off on on on"
  done
done