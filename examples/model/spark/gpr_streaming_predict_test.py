#
# Author(s): Chenghao Lyu <chenghao at cs dot umass dot edu>
#
# Description:
#

import sys
import getopt
from model.architecture.gpr_streaming_models_eliminate import GPR_Streaming_models


HELP = """
Format: python gpr_streaming_predict_test.py
"""


def main():

    gpr_sm = GPR_Streaming_models()
    print('Built GPR model. Ready to make streaming predictions for various scenarios...')

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
        Y_predicted_streaming = gpr_sm.predict(X_predict_latency)
        print(f'Streaming predicted value {Y_predicted_streaming}\n=======')
        ##
        Y_predicted_global_min = gpr_sm.opt_scenario1(zmesg=X_predict_latency, max_iter=100, lr=0.1)
        print(f'Predicted "global" minimum for selected objective: {Y_predicted_global_min}\n=======')
        ##

    elif choice==2:
        print(f'Example input for prediction: {X_predict_cpu}\n=======')
        Y_predicted_streaming = gpr_sm.predict(X_predict_cpu)
        print(f'Streaming predicted value {Y_predicted_streaming}\n=======')
        ##
        Y_predicted_global_min = gpr_sm.opt_scenario1(zmesg=X_predict_cpu, max_iter=100, lr=0.1)
        print(f'Predicted "global" minimum for selected objective: {Y_predicted_global_min}\n=======')
        ##

    elif choice==3:
        print(f'Example input for prediction: {X_predict_network}\n=======')
        Y_predicted_streaming = gpr_sm.predict(X_predict_network)
        print(f'Streaming predicted value {Y_predicted_streaming}\n=======')
        ##
        Y_predicted_global_min = gpr_sm.opt_scenario1(zmesg=X_predict_network, max_iter=100, lr=0.1)
        print(f'Predicted "global" minimum for selected objective: {Y_predicted_global_min}\n=======')
        ##


if __name__ == '__main__':
    main()