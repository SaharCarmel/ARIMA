from importlib.metadata import requires
import torch
import plotly.express as px
import plotly.graph_objects as go
import numpy as np


class ARIMA(torch.nn.Module):
    """ARIMA [summary]
    """

    def __init__(self,
                 p: int = 0,
                 d: int = 0,
                 q: int = 0) -> None:
        """__init__ [summary]

        Args:
            p (int): The number of lag observations included in the model,
                    also called the lag order.
            d (int): The number of times that the raw observations are
                    differenced, also called the degree of differencing.
            q (int): The size of the moving average window,
                    also called the order of moving average.
        """
        super(ARIMA, self).__init__()
        self.p = p
        self.pWeights = torch.rand(p)
        self.pWeights.requires_grad = True
        self.q = q
        self.qWeights = torch.rand(q)
        self.qWeights.requires_grad = True
        pass

    def forward(self, x: torch.Tensor, err: torch.Tensor) -> torch.Tensor:
        sample = self.pWeights*x[-1] + self.qWeights*err[-2] + err[-1]
        return sample

    def generateSample(self, length: int) -> torch.Tensor:
        sample = torch.zeros(length)
        noise = torch.tensor(np.random.normal(
            loc=0, scale=1, size=sampleSize), dtype=torch.float32)
        sample[0] = noise[0]
        with torch.no_grad():
            for i in range(length-1):
                sample[i+1] = self.forward(sample[:i+1], noise[:i+2])
                pass
        return sample

    def fit(self):
        pass


if __name__ == '__main__':
    datamodel = ARIMA(p=1, d=0, q=1)
    data = torch.rand(10)
    sampleSize = 200
    trainSize = 140
    sampleData = datamodel.generateSample(sampleSize)
    predictionModel = ARIMA(p=1, d=0, q=1)
    epochs = 100
    learningRate = 0.001
    errors = torch.tensor(np.random.normal(
        loc=0, scale=1, size=sampleSize), dtype=torch.float32)
    for epoch in range(epochs):
        prediction = torch.zeros(sampleSize)
        errors = torch.zeros(sampleSize)
        for i in range(trainSize):
            prediction[i +
                       1] = predictionModel.forward(sampleData[0:i+1], errors[0:i+2])
            pass
        loss = torch.mean(torch.pow(sampleData - prediction, 2))
        print(f'Epoch {epoch} Loss {loss}')
        loss.backward()

        predictionModel.pWeights.data = predictionModel.pWeights.data - \
            learningRate * predictionModel.pWeights.grad.data
        predictionModel.pWeights.grad.data.zero_()

        predictionModel.qWeights.data = predictionModel.qWeights.data - \
            learningRate * predictionModel.qWeights.grad.data
        predictionModel.qWeights.grad.data.zero_()
        pass
    print(predictionModel.pWeights)
    print(datamodel.pWeights)

    print(predictionModel.qWeights)
    print(datamodel.qWeights)
    inference = torch.zeros(sampleSize-trainSize)
    with torch.no_grad():
        for i in range(sampleSize - trainSize):
            inference[i] = predictionModel.forward(
                sampleData[0:i+1], errors[0:i+2])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=torch.arange(sampleSize), y=sampleData,
                             mode='lines',
                             name='sampleData'))
    fig.add_trace(go.Scatter(x=torch.arange(trainSize-1), y=prediction[1:].detach().numpy(),
                             mode='lines+markers',
                             name='overfit'))
    fig.add_trace(go.Scatter(x=(torch.arange(sampleSize-trainSize)+trainSize-2), y=inference.detach().numpy(),
                             mode='lines+markers',
                             name='predicted'))
    fig.show()
    pass
