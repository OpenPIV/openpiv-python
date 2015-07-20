import unittest
from hello import say_hello_to

class HelloTest(unittest.TestCase):

    def test_hello(self):
        self.assertEqual(say_hello_to('Marc'), 'Marc')


if __name__ == '__main__':
    unittest.main()
