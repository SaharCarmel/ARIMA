from arima_model import ARIMA
import numpy as np


datamodel = ARIMA(p=0, d=1, q=1)
sampleSize = 200
sampleData = datamodel.generateSample(sampleSize)
np.save('data.npy', sampleData.detach().numpy())

print(datamodel.dWeights)
print(datamodel.qWeights)
