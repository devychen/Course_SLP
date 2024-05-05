import unittest
import pickle
from a1 import *


class MyTestCase(unittest.TestCase):
    maxDiff = None

    def test_load_data(self):
        expected = {0: {'description': 'Sir Ken Robinson makes an entertaining and profoundly '
                                       + 'moving case for creating an education system that '
                                       + 'nurtures (rather than undermines) creativity.',
                        'url': 'https://www.ted.com/talks/ken_robinson_says_schools_kill_creativity\n'
                        },
                    1: {'description': 'With the same humor and humanity he exuded in "An '
                                       + 'Inconvenient Truth," Al Gore spells out 15 ways that '
                                       + 'individuals can address climate change immediately, from '
                                       + 'buying a hybrid to inventing a new, hotter brand name for '
                                       + 'global warming.',
                        'url': 'https://www.ted.com/talks/al_gore_on_averting_climate_crisis\n'
                        }
                    }
        actual = load_data('/Users/ychen/Lab_NLP2/HW1/test_data.csv')
        self.assertDictEqual(expected, actual)

    def test_preprocess_text1(self):
        sentence = "Please, let the tests pass this time!"
        expected = ['let', 'tests', 'pass', 'time']
        actual = preprocess_text(sentence)
        self.assertEqual(expected, actual)

    def test_preprocess_text2(self):
        sentence = "This is a sentence with a url: https:/google.com."
        expected = ["sentence", "url"]
        actual = preprocess_text(sentence)
        self.assertEqual(expected, actual)

    def test_preprocess_text3(self):
        sentence = "These sentences have whitespaces at the end.\nAnother sentence.\n"
        expected = ["sentences", "whitespaces", "end", "sentence"]
        actual = preprocess_text(sentence)
        self.assertEqual(expected, actual)

    def test_preprocess_text4(self):
        sentence = "This is a simple file. It contains: Two Sentences."
        expected = ["simple", "file", "contains", "sentences"]
        actual = preprocess_text(sentence)
        self.assertEqual(expected, actual)

    def test_preprocess_text5(self):
        sentence = "none here"
        expected = []
        actual = preprocess_text(sentence)
        self.assertEqual(expected, actual)

    def test_preprocess_texts1(self):
        data_dict = {0: {'description': 'Description of a very interesting talk.',
                         'url': 'https://example.com'
                        },
                    1: {'description': 'Another one.',
                        'url': 'https://example.com'
                        }
                    }
        expected = {0: {'description': 'Description of a very interesting talk.',
                        'url': 'https://example.com',
                        'pp_text': ['description', 'interesting', 'talk']
                        },
                    1: {'description': 'Another one.',
                        'url': 'https://example.com',
                        'pp_text': []
                        }
                    }
        actual = preprocess_texts(data_dict)
        self.assertDictEqual(expected, actual)

    def test_preprocess_texts2(self):
        data_dict = {0: {'description': 'Sir Ken Robinson makes an entertaining and profoundly '
                                       + 'moving case for creating an education system that '
                                       + 'nurtures (rather than undermines) creativity.',
                         'url': 'https://www.ted.com/talks/ken_robinson_says_schools_kill_creativity\n'
                        },
                    1: {'description': 'With the same humor and humanity he exuded in "An '
                                       + 'Inconvenient Truth," Al Gore spells out 15 ways that '
                                       + 'individuals can address climate change immediately, from '
                                       + 'buying a hybrid to inventing a new, hotter brand name for '
                                       + 'global warming.',
                        'url': 'https://www.ted.com/talks/al_gore_on_averting_climate_crisis\n'
                        }
                    }
        expected = {0: {'description': 'Sir Ken Robinson makes an entertaining and profoundly '
                                       + 'moving case for creating an education system that '
                                       + 'nurtures (rather than undermines) creativity.',
                        'url': 'https://www.ted.com/talks/ken_robinson_says_schools_kill_creativity\n',
                        'pp_text': ['sir', 'ken', 'robinson', 'makes', 'entertaining', 'profoundly', 'moving',
                                    'case', 'creating', 'education', 'system', 'nurtures', 'undermines', 'creativity']
                        },
                    1: {'description': 'With the same humor and humanity he exuded in "An '
                                       + 'Inconvenient Truth," Al Gore spells out 15 ways that '
                                       + 'individuals can address climate change immediately, from '
                                       + 'buying a hybrid to inventing a new, hotter brand name for '
                                       + 'global warming.',
                        'url': 'https://www.ted.com/talks/al_gore_on_averting_climate_crisis\n',
                        'pp_text': ['humor', 'humanity', 'exuded', 'inconvenient', 'truth', 'al', 'gore',
                                    'spells', '15', 'ways', 'individuals', 'address', 'climate', 'change',
                                    'immediately', 'buying', 'hybrid', 'inventing', 'new', 'hotter', 'brand',
                                    'global', 'warming']
                        }
                    }
        actual = preprocess_texts(data_dict)
        self.assertDictEqual(expected, actual)

    def test_get_vector1(self):
        tokens = []
        expected = None
        actual = get_vector(tokens)
        self.assertEqual(expected, actual)

    def test_get_vector2(self):
        tokens = ["global", "warming"]
        expected = np.array([
            -0.08374023,  0.09472656, -0.02270508,  0.23901367, -0.24609375,  0.01403809,
            -0.07055664, -0.2055664 ,  0.07897949,  0.22875977, -0.16894531, -0.02941895,
             0.1114502 ,  0.00927734, -0.14404297,  0.07141113, -0.00247192,  0.07025146,
             0.06970215, -0.07312012, -0.1328125 ,  0.067276  , -0.09056091, -0.2084961 ,
             0.12036133, -0.01573563, -0.07763672,  0.00048828,  0.09375   ,  0.05615234,
            -0.11657715, -0.06359863, -0.0369873 , -0.16210938, -0.02294922, -0.07208252,
             0.07794189, -0.03179932,  0.11694336,  0.17358398,  0.02410889, -0.10787964,
            -0.04455566,  0.03173828,  0.09179688, -0.03076172, -0.0355835 ,  0.11669922,
            -0.0378418 ,  0.10479736, -0.08587646,  0.1427002 , -0.35302734, -0.09667969,
             0.2241211 ,  0.1875    ,  0.13781738, -0.18115234, -0.01660156, -0.07275391,
            -0.16186523,  0.13330078,  0.09350586, -0.25048828,  0.14428711, -0.05834961,
            -0.0489502 , -0.01681519,  0.05200195, -0.01397705, -0.07421875,  0.16577148,
             0.19799805,  0.07769012, -0.12643814, -0.1574707 ,  0.24731445,  0.2265625 ,
            -0.05889893, -0.00634766,  0.15881348, -0.11425781,  0.08154297,  0.02583313,
            -0.0012207 ,  0.06896973, -0.14208984,  0.08959961, -0.06958008,  0.2133789 ,
             0.199646  , -0.17071533, -0.12573242,  0.03344727, -0.08398438, -0.11682129,
             0.0813446 , -0.13574219,  0.24365234, -0.15466309, -0.2109375 , -0.00878906,
             0.08966064,  0.0234375 , -0.05786133,  0.0244751 , -0.05615234,  0.09059143,
            -0.02832031,  0.0256958 , -0.01885986, -0.02075195, -0.18041992, -0.08251953,
             0.01782227, -0.10119629,  0.03515625, -0.05688477,  0.04492188,  0.10693359,
            -0.23535156, -0.18652344, -0.05062866,  0.09326172, -0.28857422, -0.03112793,
             0.07397461,  0.01953125, -0.17523193,  0.04248047,  0.05871582,  0.25341797,
            -0.1303711 , -0.03527832,  0.03295898, -0.00814819,  0.10827637, -0.17480469,
            -0.18579102,  0.01379395,  0.12158203, -0.09716797,  0.01928711, -0.03527832,
             0.04351807,  0.02539062,  0.19238281,  0.04162598,  0.20874023, -0.07873535,
             0.09362793, -0.12670898,  0.07745361,  0.07763672,  0.12442017, -0.09423828,
             0.14672852, -0.0524292 ,  0.14453125, -0.14770508, -0.05737305, -0.1439209 ,
             0.04295731,  0.01416016,  0.1315918 ,  0.04943848,  0.16711426, -0.06445312,
             0.012146  ,  0.0501709 , -0.10818481,  0.11437988,  0.17651367, -0.01025391,
             0.02331543,  0.02050781, -0.0723877 , -0.00549316, -0.16503906, -0.02050781,
            -0.06384277, -0.08804321,  0.16894531, -0.30566406,  0.18261719, -0.00073242,
            -0.08519363,  0.11578369,  0.08764648,  0.19580078, -0.17382812,  0.01220703,
             0.12695312, -0.08203125, -0.07641602,  0.274292  ,  0.03369141, -0.09320068,
             0.16552734, -0.31054688, -0.04559326, -0.04855347, -0.07006836,  0.03009033,
             0.04608154,  0.05249023,  0.26367188, -0.15551758, -0.09506226,  0.16832733,
            -0.16210938, -0.08935547, -0.02219391,  0.1527996 , -0.01318359, -0.01202393,
             0.03173828, -0.13793945, -0.17700195,  0.05224609,  0.12915039,  0.12036133,
             0.05555725,  0.0094223 , -0.06970215,  0.05517578,  0.0546875 , -0.22314453,
             0.11767578,  0.13104248, -0.10620117,  0.1133194 ,  0.25634766, -0.23876953,
             0.05078125, -0.02282715, -0.20581055,  0.07397461,  0.00610352, -0.15722656,
             0.27368164,  0.08422852, -0.04321289, -0.14404297, -0.13024902,  0.02856445,
             0.10974121, -0.03466797,  0.04992676, -0.2607422 , -0.03601074,  0.13043213,
            -0.07133484,  0.03363037,  0.18121338, -0.13952637,  0.1899414 , -0.1184082 ,
            -0.12261963, -0.02783203, -0.14868164, -0.08862305,  0.00484467, -0.02166748,
             0.14892578,  0.12597656, -0.14941406,  0.01721191,  0.15332031, -0.14111328,
            -0.1899414 ,  0.01196289,  0.00152588, -0.15576172,  0.03787231, -0.1461258 ,
            -0.28027344,  0.21923828,  0.19189453,  0.13598633,  0.1652832 , -0.10302734,
             0.06781006, -0.02197266,  0.20239258,  0.02832031,  0.03564453,  0.00366211,
             0.1743164 , -0.04666138,  0.02923584,  0.07165527,  0.02893066,  0.10681152,
             0.0736084 , -0.02374268,  0.02733612,  0.22045898,  0.23291016, -0.04748535
        ]
)
        actual = get_vector(tokens)
        self.assertTrue(np.allclose(expected, actual),
                        msg=f"expected {expected}\nbut got\n{actual}")

    def test_get_vectors1(self):
        data_dict = {
            0: {'description': 'Another one.',
                'url': 'https://example.com',
                'pp_text': []
                }
        }
        expected = {
            0: {'description': 'Another one.',
                'url': 'https://example.com',
                'pp_text': [],
                'vector': None
                }
        }
        actual = get_vectors(data_dict)

        self.assertIn('description', actual[0])
        self.assertIn('url', actual[0])
        self.assertIn('pp_text', actual[0])
        self.assertIn('vector', actual[0])

    def test_get_vectors2(self):
        data_dict = {
            0: {'description': 'Another one.',
                'url': 'https://example.com',
                'pp_text': []
                }
        }
        expected = {
            0: {'description': 'Another one.',
                'url': 'https://example.com',
                'pp_text': [],
                'vector': None
                }
        }
        actual = get_vectors(data_dict)

        self.assertEqual(None, actual[0]['vector'])

    def test_get_vectors3(self):
        data_dict = {
            0: {'description': 'Description of a very interesting talk.',
                'url': 'https://example.com',
                'pp_text': ['description', 'interesting', 'talk']
                }
        }
        expected = {
            0: {'description': 'Description of a very interesting talk.',
                'url': 'https://example.com',
                'pp_text': ['description', 'interesting', 'talk'],
                'vector': np.array([
                     1.02183027e-02, -8.20312500e-02, -5.99161796e-02,  9.74121094e-02,
                     7.90201798e-02,  3.17993164e-02,  1.79036453e-01, -2.06705723e-02,
                     3.89404297e-02,  3.66210938e-02,  6.91731786e-03, -7.22249374e-02,
                    -4.91739921e-02,  7.40559911e-03, -5.20731620e-02,  2.77018219e-01,
                     6.26627589e-03,  2.39013672e-01,  6.74641952e-02, -2.24609375e-01,
                     9.40958690e-03,  1.08235680e-01, -8.62223282e-02,  1.18693031e-01,
                    -6.42903671e-02,  1.64428711e-01, -4.54711914e-03,  9.09016952e-02,
                     3.36914062e-02,  1.10514320e-01, -5.65592460e-02,  4.06901054e-02,
                     7.48697901e-03, -9.17561818e-03, -7.16145849e-03, -1.91243496e-02,
                     5.42055778e-02, -3.91438790e-02,  1.42252609e-01,  1.18408203e-01,
                     1.85546875e-02,  2.86458340e-02, -5.99772148e-02,  1.19954430e-01,
                    -2.38932297e-01,  9.61914062e-02, -6.62434921e-02, -8.47117081e-02,
                    -5.34261055e-02, -2.05078125e-02, -9.56166610e-02, -7.45849609e-02,
                    -1.25488281e-01,  5.79223633e-02,  3.62141919e-03, -2.75878906e-02,
                     1.44856768e-02, -1.55029297e-01,  2.83528656e-01, -1.22029625e-01,
                     7.50325546e-02,  1.54622393e-02, -1.57714844e-01, -1.27685547e-01,
                     1.34277344e-02, -1.31022139e-02, -7.84098282e-02, -1.62556972e-02,
                     1.14746094e-02, -3.31420898e-02,  6.52669296e-02,  9.84700546e-02,
                     5.90935536e-02,  4.80143242e-02, -9.08406600e-02,  4.75260429e-02,
                     9.15934220e-02,  1.44083664e-01,  1.58691406e-01,  2.38606766e-01,
                     1.61560059e-01,  4.76074219e-02, -2.42513027e-02,  1.39567060e-02,
                    -3.58886719e-02, -1.14583336e-01, -1.03753410e-01,  4.82584648e-02,
                    -1.91243493e-03, -7.91829452e-02,  3.40983085e-02, -6.00179024e-02,
                    -1.62109375e-01,  9.13899764e-02,  2.97037754e-02,  3.36100273e-02,
                    -3.25520843e-04,  5.62337227e-02, -3.49934888e-03, -9.11458302e-03,
                    -2.13297531e-01,  5.55826835e-02, -3.77604179e-02, -7.32421875e-03,
                     4.03442383e-02, -8.13802108e-05,  5.72916679e-02, -1.00179039e-01,
                    -1.98567715e-02, -7.58463517e-02, -2.39257812e-02, -8.96809921e-02,
                     2.78320312e-02, -1.50390625e-01,  9.39127579e-02, -1.81152344e-01,
                     1.79443359e-01, -1.04492188e-01,  2.21679688e-01,  1.03332520e-01,
                    -6.34860992e-02,  7.02718124e-02, -5.87565117e-02,  1.79036465e-02,
                    -1.49449661e-01, -6.51041642e-02, -1.29964188e-01,  5.85937500e-02,
                    -3.53190117e-02,  1.03678383e-01, -1.82210281e-01, -3.19010407e-01,
                    -7.58463517e-02,  1.04654945e-01,  6.06689453e-02,  8.69547501e-02,
                    -2.36816406e-02, -6.97428361e-02, -8.91927108e-02,  1.51301071e-01,
                     1.86848953e-01,  8.51643905e-02, -2.44140625e-02,  4.13411446e-02,
                     3.75162773e-02,  2.49837246e-02, -2.85237636e-02, -1.14095055e-01,
                    -7.57649764e-02, -1.15275063e-01,  2.69114170e-02,  8.55712891e-02,
                    -2.71809906e-01,  6.65073395e-02, -5.97330742e-02, -1.28743485e-01,
                    -1.00107826e-01, -1.20442711e-01, -5.32226562e-02, -3.39355469e-02,
                     5.66406250e-02,  1.87500000e-01, -5.96516915e-02, -1.08215332e-01,
                     1.63411453e-01, -4.76277657e-02,  1.09171547e-01, -1.20808922e-01,
                    -8.96809921e-02,  7.21232072e-02,  5.54097481e-02,  1.17838539e-01,
                     6.00992851e-02, -1.04329430e-01,  5.66609688e-02,  6.32222518e-02,
                    -2.25830078e-03, -1.23860680e-01, -9.60286483e-02,  1.26851397e-02,
                     1.62760410e-02, -1.75781250e-02,  1.19222002e-02, -4.23787422e-02,
                    -4.02018242e-02, -4.58984375e-02, -8.64969864e-02,  1.17187500e-01,
                    -4.19108085e-02, -4.49625663e-02,  4.69156913e-02,  2.77709961e-02,
                     2.92154942e-02, -8.62630233e-02,  1.13769531e-01,  9.72493459e-03,
                    -1.35253906e-01, -6.86848983e-02, -4.97639962e-02, -1.46891281e-01,
                     5.12288399e-02,  1.30371094e-01, -2.35026047e-01, -2.53702793e-02,
                    -8.43912736e-02,  4.98860665e-02, -1.03983559e-01,  6.87255859e-02,
                    -5.20833349e-03,  3.34472656e-02, -2.10545864e-02,  5.02929688e-02,
                    -2.49837246e-02, -6.42344132e-02, -1.24450684e-01,  2.58789062e-02,
                     2.16796875e-01,  1.14664711e-01,  9.60286427e-03, -1.30330399e-01,
                     4.40266915e-02,  1.23697914e-01, -1.45589188e-01, -5.78613281e-02,
                     1.02213539e-01, -1.50502520e-02,  6.27543107e-02,  9.92838517e-02,
                     4.50846367e-02,  2.19726562e-02,  5.07812500e-02, -4.32128906e-02,
                    -3.80045585e-02,  8.80432129e-02,  2.63671875e-01, -9.78597030e-02,
                     6.11979179e-02,  2.73234043e-02,  6.40462264e-02, -2.60416660e-02,
                     4.84212227e-02,  5.46875000e-02, -4.56542969e-02, -6.99462891e-02,
                    -3.80859375e-02,  1.14583336e-01,  9.57845077e-02,  8.28450546e-02,
                     4.17480469e-02, -1.08072914e-01,  2.31119785e-02,  2.45117188e-01,
                    -6.26627579e-02,  9.22851562e-02, -3.61328125e-02, -1.51041672e-01,
                    -1.18652344e-01, -4.08935547e-02,  7.42746964e-02, -1.52397156e-01,
                     6.75455704e-02, -2.08663940e-02, -2.35514328e-01, -1.86360683e-02,
                     1.12223305e-01,  2.57486969e-01, -1.80664062e-02, -1.93359375e-01,
                    -1.85872391e-01, -1.05061851e-01,  1.13932295e-02,  6.49414062e-02,
                     2.28190109e-01,  1.79036465e-02,  1.00911461e-01, -1.57877609e-01,
                     9.82259139e-02, -1.73339844e-01,  8.70869979e-02,  7.22656250e-02,
                     7.38321915e-02, -9.52148438e-02, -3.79231758e-02, -5.42297363e-02,
                     5.22054024e-02, -2.66927090e-02, -6.89697266e-02, -4.23177099e-03,
                     5.11474609e-02, -5.86751290e-02,  5.71289062e-02,  2.76692715e-02,
                     1.18815107e-02, -5.02929688e-02,  9.24072266e-02, -1.99218750e-01,
                     9.44824219e-02, -1.69270828e-01,  5.37109375e-03, -6.81152344e-02])
                }
        }
        actual = get_vectors(data_dict)

        self.assertTrue(np.allclose(expected[0]['vector'], actual[0]['vector']),
                        msg=f"expected {expected[0]['vector']}\nbut got\n{actual[0]['vector']}")

    def test_cosine_similarity(self):
        v1 = np.array([.5, -.3, -.2])
        v2 = np.array([.4, -.2, -.1])
        expected = 0.9911892555667041
        actual = cosine_similarity(v1, v2)

        self.assertAlmostEqual(expected, actual)

    def test_k_most_similar(self):
        with open('/Users/ychen/Lab_NLP2/HW1/test_dict.pkl', 'rb') as f:
            data_dict = pickle.load(f)
        query = "migrating birds"
        expected = [(49, 0.318628), (17, 0.31004193), (32, 0.3005341)]
        actual = k_most_similar(query, data_dict, 3)
        self.assertEqual(len(expected), len(actual))
        for tup1,tup2 in zip(expected, actual):
            for val1,val2 in zip(tup1,tup2):
                if type(val1) is float:
                    self.assertAlmostEqual(val1, val2)
                else:
                    self.assertEqual(val1, val2)


if __name__ == '__main__':
    unittest.main()
