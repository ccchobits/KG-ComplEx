import os
import time
import pandas as pd


class Logger:
    def __init__(self, configs):
        self.configs = configs
        self.path = "../scripts/asset/log/"

    # log: .type: python.dict()
    def write(self, log):
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        with open(os.path.join(self.path, current_time + ".log"), "w") as f:
            f.write(self.get_printable_log(log))

    def get_printable_log(self, log):
        log = pd.DataFrame(pd.DataFrame(log, columns=["tail:raw", "tail:filtered", "head:raw", "head:filtered"]))
        return log.to_string()