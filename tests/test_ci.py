import unittest

class TestCI(unittest.TestCase):
    def test_ci(self):
        # Test if CI works
        self.assertEqual(0, 0)
        self.assertEqual(1, 1)
        self.assertEqual(2, 2)

if __name__ == '__main__':
    unittest.main()
