import pandas as pd
import numpy as np
rng = np.random.RandomState(0)
X = rng.random_sample((10, 3))
print(X)

from ipynb.fs.full.Process_Training_Data import class_to_action

print(class_to_action(1))