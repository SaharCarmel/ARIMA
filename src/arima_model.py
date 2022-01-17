from importlib.metadata import requires
from random import seed
import torch

import numpy as np


class ARIMA(torch.nn.Module):
    """ARIMA [summary]
    """

    def __init__(self,
                 p: int = 0,
                 d: int = 0,
                 q: int = 0) -> None:
        """__init__ General ARIMA model constructor.

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
        self.d = d
        self.dWeights = torch.rand(d)
        self.dWeights.requires_grad = True
        self.drift = torch.rand(1)
        pass

    def forward(self, x: torch.Tensor, err: torch.Tensor) -> torch.Tensor:
        """forward the function that defines the ARIMA(0,1,1) model.
        It was written specifically for the case of ARIMA(0,1,1).

        Args:
            x (torch.Tensor): The input data. All the past observations
            err (torch.Tensor): The error term. A normal distribution vector.

        Returns:
            torch.Tensor: The output of the model. The current prediction.
        """
        zData = torch.diff(x)
        zPred = self.dWeights*zData[-1] + \
            self.qWeights*err[-2] + err[-1] + self.drift
        aPred = zPred + x[-1]
        return aPred

    def generateSample(self, length: int) -> torch.Tensor:
        sample = torch.zeros(length)
        noise = torch.tensor(np.random.normal(
            loc=0, scale=1, size=length), dtype=torch.float32)
        sample[0] = noise[0]
        with torch.no_grad():
            for i in range(length-2):
                sample[i+2] = self.forward(sample[:i+2], noise[:i+2])
                pass
        return sample

    def fit(self, trainData: torch.Tensor, epochs: int, learningRate: float) -> None:
        dataLength = len(trainData)
        errors = torch.tensor(np.random.normal(
            loc=0, scale=1, size=dataLength), dtype=torch.float32)
        for epoch in range(epochs):
            prediction = torch.zeros(dataLength)
            for i in range(dataLength-2):
                prediction[i +
                           2] = self.forward(trainData[0:i+2], errors[0:i+2])
                pass
            loss = torch.mean(torch.pow(trainData - prediction, 2))
            print(f'Epoch {epoch} Loss {loss}')
            loss.backward()

            self.dWeights.data = self.dWeights.data - \
                learningRate * self.dWeights.grad.data
            self.dWeights.grad.data.zero_()

            self.qWeights.data = self.qWeights.data - \
                learningRate * self.qWeights.grad.data
            self.qWeights.grad.data.zero_()
    pass
