## DAG sql2stages

This part of the code is written to split a physical query plan into DAG of query stages via 'Exchange' operators. Alongside there are also additional functionalities which would be explained later below.

The nodes and edges of the query plan is taken from the spark as input. This information is used to construct a DAG of the entire plan. It is also split into query stages via 'Exchange' operators. A unique id is assigned to every stage and their dependencies are provided in the output. The graphs are constructed using the DGL library.

There are two python scripts as detailed below:
1. ```examples/trace/spark-parser/dag_sql2stages.py```
<!--ts-->
* **Input** - It is fetched directly from the Spark API. The query plan for [application_1666973014171_0015](http://node13-opa:18088/api/v1/applications/application_1666973014171_0015/sql) is taken as example in this script (look into variable `urls`).
* **Output** - The DAGs are generated for the query plan, stages and their dependencies; and visualizations are saved in the directory `examples/trace/spark-parser/application_graphs`. The assignment of nodes to query stages are also shown in the output.
<!--te-->
2. ```examples/trace/spark-parser/dag_sql2stages_verify.py``` - This script varies slightly in few aspects in comparison with the previous in terms of input and output.
<!--ts-->
* **Input** - The application id, nodes and edges information are provided in a csv file. Also the list of completed stages for every application is provided in a csv file. These two files are provided as an example for TPCH in the directory `examples/trace/spark-parser/dag_sql2stages_verify_input`.
* **Processing** - It imports the above python script and reuses its functions as required.
* **Output** - Apart from the same output generated as in the previous script, it provides spark stage allocation of nodes (incomplete info) from spark vizualisation. The script also provides a mapping between the query stage ids computed by the script and that of the spark visualization. Verification is also done, if the known query stage ids from the spark api is a subset of the completed query stages from the input csv file.
<!--te-->