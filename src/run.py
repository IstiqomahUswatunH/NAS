import pandas as pd

from utils import *
from mlpnas import MLPNAS
from CONSTANTS import TOP_N


data = pd.read_csv('DATASETS/merge_gazecom_data.csv')
x = data.drop(columns=['time', 'confidence', 'handlabeller1', 'handlabeller2', 'handlabeller_final', 'speed_2', 'direction_2', 'acceleration_2', 'speed_4', 'direction_4', 'acceleration_4', 'speed_8', 'direction_8', 'acceleration_8', 'speed_16', 'direction_16', 'acceleration_16'], axis=1, inplace=False).values
y = pd.get_dummies(data['handlabeller_final']).values

nas_object = MLPNAS(x, y)
data = nas_object.search()

get_top_n_architectures(TOP_N)
