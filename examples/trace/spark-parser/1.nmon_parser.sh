input=$1
header=$2
outpath=${3-examples/trace/spark-parser/outs/tpch_100_lhs/1.nmon}


if [[ $input == "hex1" ]]
then
  workers=("node2" "node3" "node4" "node5" "node6")
elif [[ $input == "hex2" ]]
then
  workers=("node8" "node9" "node10" "node11" "node12")
else
  echo "unsupported input $input"
fi

for worker in ${workers[@]}
do
  echo "start working on ${worker}.nmon"
  pyNmonAnalyzer -i $header/${worker}.nmon -o $outpath/${worker}
  echo "finish working on ${worker}.nmon"
done