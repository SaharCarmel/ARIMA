from arima_model import ARIMA
import torch
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# datamodel = ARIMA(p=0, d=1, q=1)
# sampleSize = 20
trainSize = 14
# sampleData = datamodel.generateSample(sampleSize)


sampleData = torch.tensor(np.load('data.npy'))
sampleSize = len(sampleData)
trainData = sampleData[:trainSize]

predictionModel = ARIMA(p=0, d=1, q=1)
predictionModel.fit(trainData, epochs=10, learningRate=0.1)
print(predictionModel.dWeights)
print(predictionModel.qWeights)

testData = sampleData[trainSize:]
inference = torch.zeros(sampleSize)
errors = torch.tensor(np.random.normal(
    loc=0, scale=1, size=sampleSize), dtype=torch.float32)
with torch.no_grad():
    for i in range(sampleSize - 2):
        inference[i+2] = predictionModel.forward(
            testData[0:i+2], errors[0:i+2])

fig = go.Figure()
fig.add_trace(go.Scatter(x=torch.arange(sampleSize), y=sampleData,
                         mode='lines',
                         name='sampleData'))
fig.add_trace(go.Scatter(x=torch.arange(trainSize), y=inference[0:trainSize].detach().numpy(),
                         mode='lines+markers',
                         name='overfit'))
fig.add_trace(go.Scatter(x=torch.arange(len(inference[trainSize:]))+trainSize, y=inference[trainSize:].detach().numpy(),
                         mode='lines+markers',
                         name='predicted'))
fig.show()
