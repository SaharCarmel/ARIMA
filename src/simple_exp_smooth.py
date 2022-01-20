""" The ARIMA model. """

import torch
import numpy as np


class ARIMA(torch.nn.Module):
    """ARIMA [summary]
    """

    def __init__(self,
                 alpha: int = 0,) -> None:
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
        self.alpha = alpha
        self.alpha = torch.rand(self.alpha)
        self.alpha.requires_grad = True
        self.drift = torch.rand(1)
        self.drift.requires_grad = True
        self.weights = {"alpha": self.alpha,
                        "drift": self.drift}

        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward the function that defines the ARIMA(0,1,1) model.
        It was written based on this https://otexts.com/fpp2/ses.html

        Args:
            x (torch.Tensor): The input data. All the past observations

        Returns:
            torch.Tensor: The output of the model. The current prediction.
        """
        prediction = self.alpha*x[-1] + \
            self.alpha*(1-self.alpha)*x[-2] + self.drift
        return prediction

    def generateSample(self, length: int) -> torch.Tensor:
        """generateSample An helper function to generate a sample of data.

        Args:
            length (int): The length of the sample.

        Returns:
            torch.Tensor: The generated sample.
        """
        sample = torch.rand(length)
        with torch.no_grad():
            for i in range(length-2):
                sample[i+2] = self.forward(sample[:i+2])
                pass
        return sample

    def fit(self,
            trainData: torch.Tensor,
            epochs: int,
            learningRate: float,
            wandb) -> None:
        """fit A function to fit the model. It is a wrapper of the

        Args:
            trainData (torch.Tensor): The training data.
            epochs (int): The number of epochs.
            learningRate (float): The learning rate.
        """
        self.optimizer = torch.optim.SGD(
            self.weights.values(), lr=learningRate)
        self.optimizer.zero_grad()
        dataLength = len(trainData)
        for epoch in range(epochs):
            prediction = torch.rand(dataLength)
            for i in range(dataLength-2):
                prediction[i +
                           2] = self.forward(trainData[0:i+2])
                pass
            loss = torch.mean(torch.pow(trainData - prediction, 2))
            print(f'Epoch {epoch} Loss {loss}')
            loss.backward()
            wandb.log({"loss": loss})
            self.optimizer.step()
            # self.alpha.data = self.alpha.data - \
            #     learningRate * self.alpha.grad.data
            # self.alpha.grad.data.zero_()

            # self.drift.data = self.drift.data - \
            #     learningRate * self.drift.grad.data
            # self.drift.grad.data.zero_()

    pass
