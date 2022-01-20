"""The main script for training the model."""

from simple_exp_smooth import ARIMA
import torch
import numpy as np
import plotly.graph_objects as go


trainSize = 14
sampleData = torch.tensor(np.load('data.npy'))
sampleSize = len(sampleData)
trainData = sampleData[:trainSize]

predictionModel = ARIMA(alpha=1)
predictionModel.fit(trainData, epochs=300, learningRate=0.01)

testData = sampleData[trainSize:]
inference = torch.zeros(sampleSize)
inference[0] = trainData[-2]
inference[1] = trainData[-1]
with torch.no_grad():
    for i in range(len(testData) - 2):
        inference[i+2] = predictionModel.forward(
            inference[0:i+2])

fig = go.Figure()
fig.add_trace(go.Scatter(x=torch.arange(sampleSize), y=sampleData,
                         mode='lines',
                         name='sampleData'))
fig.add_trace(go.Scatter(x=torch.arange(len(testData))+trainSize,
                         y=inference.detach().numpy(),
                         mode='lines+markers',
                         name='predicted'))
fig.show()
