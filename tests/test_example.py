import unittest
from src.test import capitalize_name


class TestSum(unittest.TestCase):
    def test_captilize(self):
        string = "hello"
        self.assertEqual(capitalize_name(string), string.capitalize())


if __name__ == '__main__':
    unittest.main()
