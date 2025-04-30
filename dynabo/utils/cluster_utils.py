import os
import time

import numpy as np


def initialise_experiments():
    if "Desktop" not in os.getcwd():
        for i in range(20):
            try:
                from yahpo_gym import local_config

                local_config.init_config()
                local_config.set_data_path("benchmark_data/yahpo_data")
                break

            except Exception as e:
                print(e)
                # sleep for i seconds
                time.sleep(i * np.random.uniform(0.5, 1.5))
