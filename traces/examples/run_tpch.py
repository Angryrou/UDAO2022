from pathlib import Path
import numpy as np

import udao_trace.utils as utils
from udao_trace import SparkProxy

base_dir = Path(__file__).parent
knob_meta_file = str(base_dir / "assets/spark_configuration_aqe_on.json")
spark_proxy = SparkProxy(knob_meta_file)
# for k, v in spark_proxy.get_default_conf().items():
#     print(k, v)

conf_norm = [0.] * len(spark_proxy.knob_list)
conf_denorm = spark_proxy.denormalize(conf_norm)
conf = spark_proxy.construct_configuration(conf_denorm)
print(conf_denorm)
print(conf)
print(spark_proxy.get_default_conf(to_dict=False))

conf_denorm = [2, 1, 20, 2, 2, 0, 1, 60] + [2, 2, 0, 1, 25, 2, 50, 2, 2, 20, 2]
print(spark_proxy.construct_configuration(conf_denorm))
