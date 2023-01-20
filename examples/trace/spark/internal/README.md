## AQE analyses

Compare the performance among
1. run with the default configuration with AQE disabled
   ```bash
   python examples/trace/spark/internal/1.run_default.py --debug 0 --if-aqe 0 --num-trials 3
   ```
2. run with the default configuration with AQE enabled 
   ```bash
   python examples/trace/spark/internal/1.run_default.py --debug 0 --if-aqe 1 --num-trials 3
   ```    
3. run with AQE enabled over different configurations
   ```bash
   python examples/trace/spark/internal/2.knob_hp_tuning.py --target-query 1 --knob-type res --num-conf-lhs 40
   python examples/trace/spark/internal/2.knob_hp_tuning.py --target-query 1 --knob-type sql --num-conf-lhs 40
   python examples/trace/spark/internal/2.knob_hp_tuning.py --target-query 18 --knob-type res --num-conf-lhs 40
   python examples/trace/spark/internal/2.knob_hp_tuning.py --target-query 18 --knob-type sql --num-conf-lhs 40
   
for q in 1 18 {2..17} 19 20 21 22; do
   python examples/trace/spark/internal/2.knob_hp_tuning.py --target-query $q --knob-type res --num-conf-lhs 40 --if-aqe 0 --worker tpch
   python examples/trace/spark/internal/2.knob_hp_tuning.py --target-query $q --knob-type sql --num-conf-lhs 40 --if-aqe 0 --worker tpch
done

   ```
