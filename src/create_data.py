"""A script to generate offline data for training."""
from simple_exp_smooth import ARIMA
import numpy as np


datamodel = ARIMA(alpha=1)
datamodel.eval()
sampleSize = 20
sampleData = datamodel.generateSample(sampleSize)

np.save('data.npy', sampleData.detach().numpy())
