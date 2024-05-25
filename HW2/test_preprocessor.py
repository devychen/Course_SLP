import unittest

import spacy
import compress_fasttext
import torch.optim
import os
from preprocessor import Preprocessor
from constants import *


class TestPreprocessor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.nlp = spacy.load('de_core_news_sm', disable=['parser', 'ner'])
        cls.embeddings = compress_fasttext.models.CompressedFastTextKeyedVectors.load("fasttext-de-mini")
        cls.label_map_file = "/Users/ychen/Notes-SNLP/Data/label_map.json"
        cls.label_map = {
            "Web": 0,
            "Panorama": 1,
            "International": 2,
            "Wirtschaft": 3,
            "Sport": 4,
            "Inland": 5,
            "Etat": 6,
            "Wissenschaft": 7,
            "Kultur": 8
        }

    @classmethod
    def tearDownClass(cls):
        cls.embeddings = None
        cls.nlp = None

    @classmethod
    def setUp(self) -> None:
        self.emb_dim = self.embeddings.vector_size
        self.prep = Preprocessor(self.nlp, self.embeddings)

    @classmethod
    def tearDown(self) -> None:
        self.prep = None

    def test_constructor1(self):
        expected = self.nlp
        actual = self.prep.nlp
        self.assertEqual(expected, actual)

    def test_constructor2(self):
        expected = self.embeddings
        actual = self.prep.embeddings
        self.assertEqual(expected, actual)

    def test_load_csv_data1(self):
        expected = [
            "21-Jähriger fällt wohl bis Saisonende aus.",
            "'Der neue \"King of Pop\" soll er sein.'",
            "Dem Gründer der geschlossenen Tauschplattform Megaupload drohen 20 Jahre Haft.",
            "Die Europäische Zentralbank bewilligt  keine weiteren Nothilfen mehr.",
            "\"Prekäre Situation\" in Innsbruck, Salzburg fehlen Fachärzte, in Wien warten Patienten.",
            "Rechnungshof-Prüfbericht warnt laut Medien vor Finanzloch.",
            "Vorerst kein Termin für zweite Verhandlungsrunde – UNO sieht Hilfsbedarf von 1,4 Milliarden Euro.",
            "Polizei ermittelt wegen Vandalismus.",
            "In einem Forschungszentrum für Primatologie untersuchen Forscher, wie Stress und Fortpflanzung zusammenhängen."
        ]
        self.prep._load_csv_data("UnitTestData/unittest-train.csv")
        actual = self.prep.X_texts
        self.assertListEqual(expected, actual)

    def test_load_csv_data2(self):
        expected = ["Sport", "Kultur","Web","Wirtschaft","Inland","Etat","International","Panorama","Wissenschaft"]
        self.prep._load_csv_data("UnitTestData/unittest-train.csv")
        actual = self.prep.y_texts
        self.assertListEqual(expected, actual)

    def test_load_label_map(self):
        expected = self.label_map
        self.prep._load_label_map(self.label_map_file)
        actual = self.prep.label_map
        self.assertDictEqual(expected, actual)

    def test_load_data1(self):
        expected = [
            "21-Jähriger fällt wohl bis Saisonende aus.",
            "'Der neue \"King of Pop\" soll er sein.'",
            "Dem Gründer der geschlossenen Tauschplattform Megaupload drohen 20 Jahre Haft.",
            "Die Europäische Zentralbank bewilligt  keine weiteren Nothilfen mehr.",
            "\"Prekäre Situation\" in Innsbruck, Salzburg fehlen Fachärzte, in Wien warten Patienten.",
            "Rechnungshof-Prüfbericht warnt laut Medien vor Finanzloch.",
            "Vorerst kein Termin für zweite Verhandlungsrunde – UNO sieht Hilfsbedarf von 1,4 Milliarden Euro.",
            "Polizei ermittelt wegen Vandalismus.",
            "In einem Forschungszentrum für Primatologie untersuchen Forscher, wie Stress und Fortpflanzung zusammenhängen."
        ]
        self.prep.load_data("UnitTestData/unittest-train.csv", "Data/label_map.json")
        actual = self.prep.X_texts
        self.assertListEqual(expected, actual)

    def test_load_data2(self):
        expected = ["Sport", "Kultur", "Web", "Wirtschaft", "Inland", "Etat", "International", "Panorama", "Wissenschaft"]
        self.prep.load_data("UnitTestData/unittest-train.csv", "Data/label_map.json")
        actual = self.prep.y_texts
        self.assertListEqual(expected, actual)

    def test_load_data3(self):
        expected = self.label_map
        self.prep.load_data("UnitTestData/unittest-train.csv", "Data/label_map.json")
        actual = self.prep.label_map
        self.assertDictEqual(expected, actual)

    def test_preprocess_text(self):
        text = "Vorerst kein Termin für zweite \"Verhandlungsrunde\" – \nUNO sieht Hilfsbedarf von 1,4 Milliarden Euro.\n"
        expected = ["Vorerst", "Termin", "Verhandlungsrunde", "UNO", "sieht", "Hilfsbedarf", "Milliarden", "Euro"]
        actual = self.prep._preprocess_text(text)
        self.assertListEqual(expected, actual)

    def test_calc_mean_embedding1(self):
        expected = self.emb_dim
        actual = self.prep._calc_mean_embedding("21-Jähriger fällt Saisonende").shape[0]
        self.assertEqual(expected, actual, msg="wrong shape")

    def test_calc_mean_embedding2(self):
        expected = torch.tensor([0.0239, 0.0084, 0.0255, 0.0518, 0.0005])
        actual = self.prep._calc_mean_embedding("21-Jähriger fällt Saisonende")[:5]
        self.assertTrue(torch.allclose(expected, actual, atol=0.01))

    def test_calc_mean_embedding3(self):
        expected = torch.float32
        mean_emb = self.prep._calc_mean_embedding("21-Jähriger fällt Saisonende")[:5]
        actual = mean_emb.dtype
        self.assertEqual(expected, actual, msg="wrong dtype")

    def test_generate_X_tensor1(self):
        self.prep.X_texts = [
            "21-Jähriger fällt wohl bis Saisonende aus.",
            "'Der neue \"King of Pop\" soll er sein.'"
        ]
        expected0, expected1 = 2, 300
        self.prep._generate_X_tensor()
        actual0, actual1 = self.prep.X_tensor.shape
        self.assertEqual(expected0, actual0, msg="wrong X_tensor shape[0]")
        self.assertEqual(expected1, actual1, msg="wrong X_tensor shape[1]")

    def test_generate_X_tensor2(self):
        self.prep.X_texts = [
            "21-Jähriger fällt wohl bis Saisonende aus.",
            "'Der neue \"King of Pop\" soll er sein.'"
        ]
        expected = torch.tensor([[ 0.0239,  0.0084,  0.0255,  0.0518,  0.0005],
                                 [-0.0533,  0.0024, -0.0190,  0.0193,  0.0970]])
        self.prep._generate_X_tensor()
        actual = self.prep.X_tensor[:, :5]
        self.assertTrue(torch.allclose(expected, actual, atol=0.01))

    def test_generate_y_tensor(self):
        self.prep.label_map = {'Sport': 0, 'Etat': 1, 'Panorama': 2, 'Kultur': 3}
        self.prep.y_texts = ['Sport', 'Etat', 'Panorama', 'Etat', 'Kultur', 'Panorama']
        expected = torch.tensor([0, 1, 2, 1, 3, 2])
        self.prep._generate_y_tensor()
        actual = self.prep.y_tensor
        self.assertTrue(torch.equal(expected, actual))

    def test_generate_tensors(self):
        self.prep.X_texts = [
            "21-Jähriger fällt wohl bis Saisonende aus.",
            "'Der neue \"King of Pop\" soll er sein.'"
        ]
        self.prep.y_texts = ['Sport', 'Kultur']
        self.prep.label_map = {'Sport': 0, 'Etat': 1, 'Panorama': 2, 'Kultur': 3}

        expected_X = torch.tensor([[0.0239, 0.0084, 0.0255, 0.0518, 0.0005],
                                 [-0.0533, 0.0024, -0.0190, 0.0193, 0.0970]])
        expected_y = torch.tensor([0, 3])
        self.prep.generate_tensors()
        actual_X = self.prep.X_tensor[:, :5]
        actual_y = self.prep.y_tensor
        self.assertTrue(torch.allclose(expected_X, actual_X, atol=0.01))
        self.assertTrue(torch.equal(expected_y, actual_y))

    def test_save_tensors(self):
        tmp_file = "UnitTestData/unittest-tmp.pt"

        # set up preprocessor with fake tensors, label_map
        self.prep.X_tensor = torch.ones((2, 300))
        self.prep.y_tensor = torch.tensor([1, 0])
        self.prep.label_map = {'label1': 0, "label2": 1}

        # save tensors to tmp file
        self.prep.save_tensors(tmp_file)

        # load tensor file
        tensors = torch.load(tmp_file)

        # delete tmp file
        os.remove(tmp_file)

        # returned values should be the fake tensors, label_map
        self.assertTrue(torch.allclose(self.prep.X_tensor, tensors[X_KEY]))
        self.assertTrue(torch.allclose(self.prep.y_tensor, tensors[Y_KEY]))
        self.assertDictEqual(self.prep.label_map, tensors[MAP_KEY])


if __name__ == '__main__':
    unittest.main()
