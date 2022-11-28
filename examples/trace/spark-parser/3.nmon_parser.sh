$input=$1
$header=$2

if [[ $input == "hex1" ]]
then
  workers = ("node2", "node3", "node4", "node5", "node6")
elif [[ $input == "hex2" ]]
then
  workers = ("node7", "node8", "node9", "node10", "node11")
else
  echo "unsupported input $input"
fi

for worker in workers
do
  pyNmonAnalyzer -i $header/${worker}.nmon -o nmon_extracted/${worker}
done