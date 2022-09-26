# Copyright (c) 2021 Ecole Polytechnique
#
# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description:
#

import sys
import getopt
from model.architecture.batch_models_eliminate import Batch_models

HELP = """
Format: python batch_predict_test.py
"""


def main():

    conf_constraints = {
        # "conf_max": [216, 36, 4, 8, 480, 217, 1, 75, 100000, 512, 500, 2001],
        "conf_max": [144, 24, 4, 8, 192, 145, 1, 0.75, 20000, 256, 100, 144],
        # "conf_min": [8, 2, 2, 4, 12, 7, 0, 50, 1000, 32, 10, 8]
        "conf_min": [64, 8, 2, 6, 24, 35, 0, 0.50, 5000, 64, 10, 36]
    }

    bm = Batch_models()
    bm_inacc = Batch_models(conf_constraints=conf_constraints, accurate=False)
    print('Built model. Ready to make batch predictions for various scenarios...')

    X_predict_cpu = "JobID:1-2;Objective:cpu;k1:48;k2:4;k3:4;k4:8;k5:48;k6:200;k7:1;k8:60;s1:10000;s2:128;s3:10;s4:200"

    X_predict_network = "JobID:13-4;Objective:network;k1:8;k2:2;k3:2;k4:4;k5:12;k6:7;k7:0;k8:50;s1:1000;s2:32;s3:10;s4:8"

    X_predict_latency = "JobID:13-4;Objective:latency;k1:8;k2:2;k3:2;k4:8;k5:12;k6:7;k7:0;k8:60;s1:1000;s2:64;s3:10;s4:8"

    choice = 1
    while True:
        choice = input("Prediction objectives\n 1.latency\n 2.cpu\n 3.network\nEnter your choice number: ")
        if (int(choice) < 1 or int(choice) > 3):
            print(f'\nInvalid choice {choice}! Enter between 1 and 3.')
        else:
            choice = int(choice)
            break

    print()

    if choice==1:
        print(f'Example input for prediction: {X_predict_latency}\n=======')
        Y_predicted_batch = bm.predict(X_predict_latency)
        print(f'Batch predicted value {Y_predicted_batch}\n=======')
        ##
        Y_predicted_global_min = bm.opt_scenario1(zmesg=X_predict_latency, max_iter=100, lr=0.1)
        print(f'Predicted "global" minimum for selected objective: {Y_predicted_global_min}\n=======')
        ##
        Y_predicted_global_min_inacc = bm_inacc.opt_scenario1(zmesg=X_predict_latency, max_iter=100, lr=0.1)
        print(f'Predicted inaccurate "global" minimum for selected objective: {Y_predicted_global_min_inacc}\n=======')
        ##

    elif choice==2:
        print(f'Example input for prediction: {X_predict_cpu}\n=======')
        Y_predicted_batch = bm.predict(X_predict_cpu)
        print(f'Batch predicted value {Y_predicted_batch}\n=======')
        ##
        Y_predicted_global_min = bm.opt_scenario1(zmesg=X_predict_cpu, max_iter=100, lr=0.1)
        print(f'Predicted "global" minimum for selected objective: {Y_predicted_global_min}\n=======')
        ##
        Y_predicted_global_min_inacc = bm_inacc.opt_scenario1(zmesg=X_predict_cpu, max_iter=100, lr=0.1)
        print(f'Predicted inaccurate "global" minimum for selected objective: {Y_predicted_global_min_inacc}\n=======')
        ##

    elif choice==3:
        print(f'Example input for prediction: {X_predict_network}\n=======')
        Y_predicted_batch = bm.predict(X_predict_network)
        print(f'Batch predicted value {Y_predicted_batch}\n=======')
        ##
        Y_predicted_global_min = bm.opt_scenario1(zmesg=X_predict_network, max_iter=100, lr=0.1)
        print(f'Predicted "global" minimum for selected objective: {Y_predicted_global_min}\n=======')
        ##
        Y_predicted_global_min_inacc = bm_inacc.opt_scenario1(zmesg=X_predict_network, max_iter=100, lr=0.1)
        print(f'Predicted inaccurate "global" minimum for selected objective: {Y_predicted_global_min_inacc}\n=======')
        ##


if __name__ == '__main__':
    main()