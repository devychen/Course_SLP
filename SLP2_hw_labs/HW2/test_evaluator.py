import unittest
from evaluator import Evaluator


class TestEvaluator(unittest.TestCase):

    def test_evaluator(self):
        expected = 0.585522
        evaluator = Evaluator()
        evaluator.load_model("UnitTestData/unittest-model.pt")
        evaluator.load_data("UnitTestData/dev-tensors-50.pt")
        report_dict, _ = evaluator.evaluate_model()
        actual = report_dict['macro avg']['f1-score']

        self.assertAlmostEqual(expected, actual, places=4)


if __name__ == '__main__':
    unittest.main()
