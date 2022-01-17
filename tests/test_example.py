import unittest
from src.arima_model import ARIMA


class TestArima(unittest.TestCase):
    def test_init(self):
        model = ARIMA(p=1, d=0, q=0)
        self.assertEqual(len(model.pWeights), 1)

    def test_fit(self):
        testModel = ARIMA(p=0, d=1, q=1)
        sampleData = testModel.generateSample(200)
        fitModel = testModel.fit(sampleData, epochs=100, learningRate=0.01)


if __name__ == '__main__':
    unittest.main()
