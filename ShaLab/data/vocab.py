import codecs
from collections import Counter
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

class Vocabulary(object):

    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.idx2count = []

    def __len__(self):
        return len(self.idx2word)

    def __iter__(self):
        for i in range(len(self)):
            yield self.idx2word[i]

    def __contains__(self, key):
        return key in self.word2idx

    def add_word_safe(self, word, count=None):
        assert word not in self.word2idx, \
            word + ' already exists in the vocaulary'
        return self.add_word(word, count)

    def add_word(self, word, count=None):
        if word not in self.word2idx:
            idx = len(self)
            self.idx2word.append(word)
            self.idx2count.append(count)
            self.word2idx[word] = idx
        return self.word2idx[word]

    def get_word_id(self, word):
        assert word in self.word2idx
        return self.word2idx[word]

    def numberize(self, sentence):
        if isinstance(sentence, str):
            sentence = sentence.split()
        return [self.get_word_id(word) for word in sentence]

    def numberize_corpus(self, corpus):
        return [self.numberize(sentence) for sentence in corpus]

    def denumberize(self, word_ids):
        return [self.idx2word[word_id] for word_id in word_ids]

    def to_file(self, path):
        with codecs.open(path, 'w', 'utf-8') as f:
            for i in range(len(self)):
                if self.idx2count[i] is not None:
                    f.write('%s\t%d\n' % (self.idx2word[i], self.idx2count[i]))
                else:
                    f.write('%s\n' % self.idx2word[i])

    @classmethod
    def from_file(cls, path, allow_empty=False, size=None):
        vocab = cls()
        start_time = time.time()
        with codecs.open(path, 'r', 'utf-8') as f:
            for line in f:
                columns = line[:-1].split()
                count = None
                if len(columns) == 0:
                    if allow_empty:
                        word = ''
                    else:
                        raise Exception('Empty line in the vocabulary')
                elif len(columns) == 1:
                    word = columns[0]
                elif len(columns) == 2:
                    word, count = columns
                else:
                    raise Exception('Cannot parse line: ' + line[:-1])
                vocab.add_word(word, count)
                if size is not None and len(vocab) >= size:
                    break
        logging.info(
            'Loaded dictionary (%d words) from file %s in %d seconds',
            len(vocab),
            path,
            time.time() - start_time,
        )
        return vocab

    @classmethod
    def from_corpus(cls, corpus, size=None):
        vocab = cls()
        start_time = time.time()
        word_counter = Counter()
        for line in corpus:
            if isinstance(line, str):
                words = line.split(' ')
            else:
                words = line
            word_counter.update(words)

        for word, count in word_counter.most_common(size):
            if size is not None and len(vocab) >= size:
                break
            vocab.add_word(word, count)

        logging.info(
            'Built dictionary (%d words) from corpus in %d seconds',
            len(vocab),
            time.time() - start_time,
        )
        return vocab


class WordVocabulary(Vocabulary):

    BIG_COUNT = 1000000001

    PAD_WORD = '<PAD>'
    GO_WORD = '<GO>'
    EOS_WORD = '<EOS>'
    UNK_WORD = '<UNK>'

    def __init__(self):
        super(WordVocabulary, self).__init__()
        self.pad_idx = self.add_word(
            WordVocabulary.PAD_WORD,
            WordVocabulary.BIG_COUNT,
        )
        self.go_idx = self.add_word(
            WordVocabulary.GO_WORD,
            WordVocabulary.BIG_COUNT,
        )
        self.eos_idx = self.add_word(
            WordVocabulary.EOS_WORD,
            WordVocabulary.BIG_COUNT,
        )
        self.unk_idx = self.add_word(
            WordVocabulary.UNK_WORD,
            WordVocabulary.BIG_COUNT,
        )

    def get_word_id(self, word):
        return self.word2idx[word] if word in self.word2idx else self.unk_idx
