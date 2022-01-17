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
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        relevantPastData = x[-self.p:]
        relevantPastData = torch.cat((relevantPastData, torch.tensor([1])))
        sample = torch.matmul(relevantPastData, self.pWeights)
        return sample
