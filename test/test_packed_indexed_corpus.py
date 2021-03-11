import logging
import numpy as np
import tempfile
import unittest

from teds.data import PackedIndexedCorpus

class TestPackedIndexedCorpus(unittest.TestCase):
    def assert_corpus_are_equal(self, corpus_1, corpus_2):
        self.assertEqual(len(corpus_1), len(corpus_2))
        for i in range(len(corpus_1)):
            np.testing.assert_array_equal(corpus_1[i], corpus_2[i])

    def get_random_corpus(self, max_value, max_length, max_num_sentences):
        return [
            np.random.randint(
                max_value,
                size=(np.random.randint(1, max_length + 1)),
            )
            for i in range(np.random.randint(1, max_num_sentences + 1))
        ]

    def run_test_pack_simple(self, max_value, max_length, max_num_sentences):
        corpus = self.get_random_corpus(
            max_value,
            max_length,
            max_num_sentences,
        )
        corpus_packed = PackedIndexedCorpus.pack(corpus)
        self.assert_corpus_are_equal(corpus, corpus_packed)

    def test_pack_simple(self):
        for i in range(100):
            self.run_test_pack_simple(100, 100, 100)

    def run_test_pack_to_file(self, max_value, max_length, max_num_sentences):
        logging.disable(logging.INFO)    
        corpus = self.get_random_corpus(
            max_value,
            max_length,
            max_num_sentences,
        )
        with tempfile.NamedTemporaryFile(delete=False) as corpus_packed_file:
            PackedIndexedCorpus.pack_to_file(corpus, corpus_packed_file)
            self.assert_corpus_are_equal(
                corpus,
                PackedIndexedCorpus.from_file(corpus_packed_file.name)
            )

    def test_pack_to_file(self):
        for i in range(20):
            self.run_test_pack_to_file(100, 100, 100)


if __name__ == '__main__':
    unittest.main()
