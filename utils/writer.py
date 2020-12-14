import time
import pandas as pd


class Writer:
    def __init__(self, configs):
        self.configs = configs
        self.filtered_arguments = ["save_path", "seed", "dataset_path", "mode", "log", "gpu", "n_filter", "kernel"]
        self.default_span = 5
        self.head = ["model"]
        self.order = self.get_order()
        self.path = "../scripts/asset/performance.result"

    # perf.type: python.dict()
    def write(self, perf):
        with open(self.path, "a") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + "|")
            all_arguments = sorted(
                filter(lambda x: x[0] not in self.filtered_arguments, list(vars(self.configs).items())),
                key=lambda x: self.order[x[0]])
            for key, value in all_arguments:
                f.write(key + ":%s" % value + "|")
            f.write("\n")
            f.write(self.get_printable_perf(perf) + "\n")

    def get_order(self):
        all_keys = list(vars(self.configs))
        ordered_keys = self.head + sorted(
            set(all_keys).difference(set(self.head + self.filtered_arguments)))
        order = dict()
        for i, key in enumerate(ordered_keys):
            order[key] = i
        return order

    def get_printable_perf(self, perf):
        perf = pd.DataFrame(perf, index=["tail: raw ranking", "tail: filtered ranking", "head: raw ranking", "head: filtered ranking"])
        for hit in ["hit1", "hit3", "hit10"]:
            if hit in list(perf.columns):
                perf[hit].apply(lambda x: "%.2f%%" % (x * 100))
        return perf.to_string()