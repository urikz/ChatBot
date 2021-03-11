import codecs
import csv
import logging
import numpy as np
import pandas
import time
import tqdm


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)


def read_binary_word2vec(path, restriced_vocab=None):
    start_time = time.time()
    with open(path, 'rb') as f:
        num_vectors, embedding_size = (int(x) for x in f.readline().split())
        binary_length = np.dtype(np.float32).itemsize * embedding_size
        embeddings = np.zeros((num_vectors, embedding_size), dtype=np.float32)
        vocab = {}

        for i in tqdm.tqdm(range(num_vectors), 'Reading word2vec file'):
            characters = []
            while True:
                c = f.read(1)
                if c == b' ':
                    break
                # ignore newlines in front of words (some binary files have)
                if c != b'\n':
                    characters.append(c)
            word = b''.join(characters).decode('utf-8')
            binary = f.read(binary_length)
            if restriced_vocab is None or word in restriced_vocab:
                vocab[word] = len(vocab)
                embeddings[vocab[word]] = np.fromstring(
                    binary,
                    dtype=np.float32,
                )
        logging.info(
            'Loaded %d word embeddings of size %d from file %s in %d seconds',
            num_vectors,
            embedding_size,
            path,
            time.time() - start_time,
        )
    return vocab, embeddings


def load_word2vec_embeddings(path, vocab, embedding_size, binary_format):
    assert binary_format is True
    vocab_from_file, embeddings_from_file = read_binary_word2vec(
        path=path,
        restriced_vocab=vocab,
    )
    assert embeddings_from_file.shape[1] == embedding_size

    start_time = time.time()
    embeddings = np.random.uniform(
        low=-np.sqrt(3),
        high=np.sqrt(3),
        size=(len(vocab), embedding_size),
    ).astype(np.float32)

    words_initialized = 0
    for i, word in enumerate(vocab):
        assert i == vocab.get_word_id(word)
        if word in vocab_from_file:
            index_from_file = vocab_from_file[word]
            embeddings[i] = embeddings_from_file[index_from_file]
            words_initialized += 1
    logging.info(
        (
            'Initialized embeddings for %d words '
            '(%.2f%% of the vocabulary) in %d seconds'
        ),
        words_initialized,
        100.0 * words_initialized / len(vocab),
        time.time() - start_time,
    )
    return embeddings


def load_glove_embeddings(path, vocab, embedding_size):
    start_time = time.time()
    embeddings_from_file = pandas.read_table(
        path,
        sep=' ',
        index_col=0,
        header=None,
        quoting=csv.QUOTE_NONE,
        skiprows=0,
    )
    assert embeddings_from_file.loc.obj.ndim == 2
    logging.info(
        'Loaded %d word embeddings of size %d from file %s in %d seconds',
        embeddings_from_file.loc.obj.shape[0],
        embeddings_from_file.loc.obj.shape[1],
        path,
        time.time() - start_time,
    )
    start_time = time.time()
    assert embeddings_from_file.loc.obj.shape[1] == embedding_size
    embeddings = np.random.uniform(
        low=-np.sqrt(3),
        high=np.sqrt(3),
        size=(len(vocab), embedding_size),
    ).astype(np.float32)
    glove_index = np.zeros(len(vocab), dtype=np.float32)

    words_initialized = 0

    for i, word in enumerate(vocab):
        assert i == vocab.get_word_id(word)
        if word in embeddings_from_file.index:
            embeddings[i] = embeddings_from_file.loc[word].values
            glove_index[i] = embeddings_from_file.index.get_loc(word) + 1
            words_initialized += 1

    logging.info(
        'Initialized embeddings for %d words (%f%% of the vocabulary) in %d seconds',
        words_initialized,
        100.0 * words_initialized / len(vocab),
        time.time() - start_time,
    )
    return embeddings, glove_index
