import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

pd.set_option('display.max_columns',100)
pd.set_option('display.width',1000)
iris = load_iris()
print(iris)