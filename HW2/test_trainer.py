import json
import unittest

import torch
import os
from trainer import Trainer
from constants import *


class TestFNNTrainer(unittest.TestCase):

    @classmethod
    def setUp(self) -> None:
        self.trainer = Trainer()
        train_dict = torch.load("UnitTestData/train-tensors-100.pt")
        dev_dict = torch.load("UnitTestData/dev-tensors-50.pt")

        self.X_train = train_dict[X_KEY]
        self.y_train = train_dict[Y_KEY]
        self.l_map = train_dict[MAP_KEY]
        self.X_dev = dev_dict[X_KEY]
        self.y_dev = dev_dict[Y_KEY]

        self.n_dims = self.X_train.shape[1]
        self.n_classes = len(self.l_map)

        with open("UnitTestData/unittest-model-info.json", 'r') as f:
            meta = json.load(f)

        self.hidden_size = meta[HIDDEN_SIZE_KEY]
        self.n_epochs = meta[N_EPOCHS_KEY]
        self.lr = meta[LEARNING_RATE_KEY]

    @classmethod
    def tearDown(self) -> None:
        self.trainer = None
        self.X_train = None
        self.y_train = None
        self.l_map = None
        self.X_dev = None
        self.y_dev = None

    def test_load_train_tensors1(self):
        expected = self.X_train
        self.trainer._load_train_tensors("UnitTestData/train-tensors-100.pt")
        actual = self.trainer.X_train
        self.assertTrue(torch.equal(expected, actual))

    def test_load_train_tensors2(self):
        expected = self.y_train
        self.trainer._load_train_tensors("UnitTestData/train-tensors-100.pt")
        actual = self.trainer.y_train
        self.assertTrue(torch.equal(expected, actual))

    def test_load_train_tensors3(self):
        expected = self.l_map
        self.trainer._load_train_tensors("UnitTestData/train-tensors-100.pt")
        actual = self.trainer.label_map
        self.assertDictEqual(expected, actual)

    def test_load_dev_tensors1(self):
        expected = self.X_dev
        self.trainer._load_dev_tensors("UnitTestData/dev-tensors-50.pt")
        actual = self.trainer.X_dev
        self.assertTrue(torch.equal(expected, actual))

    def test_load_dev_tensors2(self):
        expected = self.y_dev
        self.trainer._load_dev_tensors("UnitTestData/dev-tensors-50.pt")
        actual = self.trainer.y_dev
        self.assertTrue(torch.equal(expected, actual))

    def test_load_dev_tensors3(self):
        expected = self.l_map
        self.trainer._load_train_tensors("UnitTestData/dev-tensors-50.pt")
        actual = self.trainer.label_map
        self.assertDictEqual(expected, actual)

    def test_load_data1(self):
        expected_X_train = self.X_train
        expected_y_train = self.y_train
        expected_X_dev = self.X_dev
        expected_y_dev = self.y_dev
        expected_lmap = self.l_map
        self.trainer.load_data("UnitTestData/train-tensors-100.pt", "UnitTestData/dev-tensors-50.pt")
        self.assertTrue(torch.equal(expected_X_train, self.trainer.X_train))
        self.assertTrue(torch.equal(expected_y_train, self.trainer.y_train))
        self.assertTrue(torch.equal(expected_X_dev, self.trainer.X_dev))
        self.assertTrue(torch.equal(expected_y_dev, self.trainer.y_dev))
        self.assertDictEqual(expected_lmap, self.trainer.label_map)

    def test_load_data2(self):
        expected = self.X_train.shape[1]
        self.trainer.load_data("UnitTestData/train-tensors-100.pt", "UnitTestData/dev-tensors-50.pt")
        actual = self.trainer.n_dims
        self.assertEqual(expected, actual)

    def test_load_data3(self):
        expected = len(self.l_map)
        self.trainer.load_data("UnitTestData/train-tensors-100.pt", "UnitTestData/dev-tensors-50.pt")
        actual = self.trainer.n_classes
        self.assertEqual(expected, actual)

    def test_train1(self):

        self.trainer.X_train = self.X_train
        self.trainer.y_train = self.y_train
        self.trainer.X_dev = self.X_dev
        self.trainer.y_dev = self.y_dev
        self.trainer.label_map = self.l_map

        # Define parameters
        n_epochs = 150
        hidden_size = 128
        learning_rate = 0.005

        self.trainer.n_samples = self.X_train.shape[0]
        self.trainer.n_dims = self.X_train.shape[1]
        self.trainer.n_classes = len(self.l_map)
        self.trainer.train(hidden_size, n_epochs, learning_rate)

        expected_keys = [
            MODEL_STATE_KEY,
            HIDDEN_SIZE_KEY,
            N_DIMS_KEY,
            N_CLASSES_KEY,
            LEARNING_RATE_KEY,
            N_EPOCHS_KEY,
            BEST_EPOCH_KEY,
            F1_MACRO_KEY,
            OPTIMIZER_NAME_KEY,
            LOSS_FN_NAME_KEY
        ]
        expected_keys = sorted(expected_keys)
        actual_keys = sorted(list(self.trainer.best_model.keys()))
        self.assertListEqual(expected_keys, actual_keys)

    def test_train2(self):

        self.trainer.X_train = self.X_train
        self.trainer.y_train = self.y_train
        self.trainer.X_dev = self.X_dev
        self.trainer.y_dev = self.y_dev
        self.trainer.label_map = self.l_map

        # Define parameters
        n_epochs = 150
        hidden_size = 128
        learning_rate = 0.005

        self.trainer.n_samples = self.X_train.shape[0]
        self.trainer.n_dims = self.X_train.shape[1]
        self.trainer.n_classes = len(self.l_map)
        self.trainer.train(hidden_size, n_epochs, learning_rate)

        expected = hidden_size
        actual = self.trainer.best_model[HIDDEN_SIZE_KEY]
        self.assertEqual(expected, actual)

    def test_train3(self):

        self.trainer.X_train = self.X_train
        self.trainer.y_train = self.y_train
        self.trainer.X_dev = self.X_dev
        self.trainer.y_dev = self.y_dev
        self.trainer.label_map = self.l_map

        # Define parameters
        n_epochs = 150
        hidden_size = 128
        learning_rate = 0.005

        self.trainer.n_samples = self.X_train.shape[0]
        self.trainer.n_dims = self.X_train.shape[1]
        self.trainer.n_classes = len(self.l_map)
        self.trainer.train(hidden_size, n_epochs, learning_rate)

        expected = self.n_dims
        actual = self.trainer.best_model[N_DIMS_KEY]
        self.assertEqual(expected, actual)

    def test_train4(self):

        self.trainer.X_train = self.X_train
        self.trainer.y_train = self.y_train
        self.trainer.X_dev = self.X_dev
        self.trainer.y_dev = self.y_dev
        self.trainer.label_map = self.l_map

        # Define parameters
        n_epochs = 150
        hidden_size = 128
        learning_rate = 0.005

        self.trainer.n_samples = self.X_train.shape[0]
        self.trainer.n_dims = self.X_train.shape[1]
        self.trainer.n_classes = len(self.l_map)
        self.trainer.train(hidden_size, n_epochs, learning_rate)

        expected = self.n_classes
        actual = self.trainer.best_model[N_CLASSES_KEY]
        self.assertEqual(expected, actual)

    def test_train5(self):

        self.trainer.X_train = self.X_train
        self.trainer.y_train = self.y_train
        self.trainer.X_dev = self.X_dev
        self.trainer.y_dev = self.y_dev
        self.trainer.label_map = self.l_map

        # Define parameters
        n_epochs = 150
        hidden_size = 128
        learning_rate = 0.005

        self.trainer.n_samples = self.X_train.shape[0]
        self.trainer.n_dims = self.X_train.shape[1]
        self.trainer.n_classes = len(self.l_map)
        self.trainer.train(hidden_size, n_epochs, learning_rate)

        expected = learning_rate
        actual = self.trainer.best_model[LEARNING_RATE_KEY]
        self.assertEqual(expected, actual)

    def test_train6(self):

        self.trainer.X_train = self.X_train
        self.trainer.y_train = self.y_train
        self.trainer.X_dev = self.X_dev
        self.trainer.y_dev = self.y_dev
        self.trainer.label_map = self.l_map

        # Define parameters
        n_epochs = 150
        hidden_size = 128
        learning_rate = 0.005

        self.trainer.n_samples = self.X_train.shape[0]
        self.trainer.n_dims = self.X_train.shape[1]
        self.trainer.n_classes = len(self.l_map)
        self.trainer.train(hidden_size, n_epochs, learning_rate)

        expected = n_epochs
        actual = self.trainer.best_model[N_EPOCHS_KEY]
        self.assertEqual(expected, actual)

    def test_save_best_model(self):

        self.trainer.X_train = self.X_train
        self.trainer.y_train = self.y_train
        self.trainer.X_dev = self.X_dev
        self.trainer.y_dev = self.y_dev
        self.trainer.label_map = self.l_map

        # Define parameters
        n_epochs = 150
        hidden_size = 128
        learning_rate = 0.005

        self.trainer.n_samples = self.X_train.shape[0]
        self.trainer.n_dims = self.X_train.shape[1]
        self.trainer.n_classes = len(self.l_map)
        self.trainer.train(hidden_size, n_epochs, learning_rate)

        base_name = "UnitTestData/tmp-model"
        model_path = "UnitTestData/tmp-model.pt"
        model_info_path = "UnitTestData/tmp-model-info.json"

        self.trainer.save_best_model(base_name)

        with open("UnitTestData/unittest-model-info.json", 'r') as f:
            expected = json.load(f)

        with open("UnitTestData/tmp-model-info.json", 'r') as f:
            actual = json.load(f)

        # delete tmp files
        os.remove(model_path)
        os.remove(model_info_path)

        self.assertDictEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()
