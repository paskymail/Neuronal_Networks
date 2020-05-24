from ast import literal_eval
import pandas as pd

df = pd.read_csv("./data/heuristic_samples_metric.csv")

# df = pd.read_csv("./heuristic_samples_metric.csv", converters={'initial_state': literal_eval, 'final_state': literal_eval})

df
