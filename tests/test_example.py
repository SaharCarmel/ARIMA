import unittest
from src.arima_model import ARIMA


class TestArima(unittest.TestCase):
    def test_init(self):
        model = ARIMA(p=1, d=0, q=0)
        self.assertEqual(len(model.pWeights), 2)


if __name__ == '__main__':
    unittest.main()
