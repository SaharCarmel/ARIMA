import torch
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
        self.pWeights = torch.rand(p+1)
        self.pWeights.requires_grad = True
        self.q = q
        self.qWeights = torch.rand(q+1)
        self.qWeights.requires_grad = True
        pass

    def forward(self, x: torch.Tensor, err: torch.Tensor) -> torch.Tensor:
        relevantPastData = x[-self.p:]
        relevantPastData = torch.cat((relevantPastData, torch.tensor([1])))
        relevantErrors = x[-self.q:]
        relevantErrors = torch.cat((relevantErrors, torch.tensor([1])))
        sample = torch.matmul(relevantPastData, self.pWeights) + \
            torch.matmul(relevantErrors, self.qWeights)
        return sample

    def generateSample(self, length: int) -> torch.Tensor:
        sample = torch.zeros(length)
        noise = torch.FloatTensor(length).uniform_(-1, 1)
        sample[0] = noise[0]
        with torch.no_grad():
            for i in range(length-1):
                sample[i+1] = self.forward(sample[:i+1]) + noise[i+1]
                pass
        return sample

if __name__ == '__main__':
    datamodel = ARIMA(p=1, d=0, q=0)
    data = torch.rand(10)
    sampleSize = 1000
    sampleData = datamodel.generateSample(sampleSize)
    predictionModel = ARIMA(p=1, d=0, q=0)
    epochs = 100
    learningRate = 0.02
    for epoch in range(epochs):
        prediction = torch.zeros(sampleSize)
        for i in range(sampleSize-1):
            prediction[i+1] = predictionModel.forward(sampleData[0:i+1])
            pass
        loss = torch.mean(torch.pow(sampleData - prediction, 2))
        print(f'Epoch {epoch} Loss {loss}')
        loss.backward()

        predictionModel.pWeights.data = predictionModel.pWeights.data - \
            learningRate * predictionModel.pWeights.grad.data
        predictionModel.pWeights.grad.data.zero_()
        pass
    print(predictionModel.pWeights)
    print(datamodel.pWeights)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=torch.arange(sampleSize), y=sampleData,
                             mode='lines',
                             name='sampleData'))
    fig.add_trace(go.Scatter(x=torch.arange(sampleSize), y=prediction.detach().numpy(),
                             mode='lines+markers',
                             name='predicted'))
    fig.show()
    pass
