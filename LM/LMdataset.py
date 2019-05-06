from collections import Counter
import numpy as np
import os,glob
import tensorflow as tf

class Dataset:
    def __init__(self, filepath, wordpath, batch_size, epoch):
        self.filepath = filepath
        self.wordpath = wordpath
        self.batch_size = batch_size
        self.epoch = epoch

    def __call__(self, *args, **kwargs):
        datagen = self.sample_generator(glob.glob(os.path.join(self.filepath, 'wiki_*')),
                                        os.path.join(self.wordpath, 'words.txt'))
        ds = tf.data.TextLineDataset.from_generator(lambda: datagen, (tf.int32, tf.int32, tf.int32, tf.int32),
                                                    (tf.TensorShape([None]), tf.TensorShape([None]),
                                                     tf.TensorShape([None]), tf.TensorShape([])))
        ds = ds.repeat(self.epoch)
        ds = ds.shuffle(buffer_size=100000)
        ds = ds.padded_batch(self.batch_size, (tf.TensorShape([None]),tf.TensorShape([None]),
                                               tf.TensorShape([None]),tf.TensorShape([])))
        ds = ds.map(lambda a, b, c, d: ({'f_wids': a, 'f_len': d}, {'l_fwd': b, 'l_bwd': c}))
        ds = ds.prefetch(10)

        return ds

    @staticmethod
    def sample_generator(filenames, wordspath, maxlen=100, minlen=10):
        def create_vocab(wpath):
            w2id = {}
            id2w = {}

            def add_word(vocab, word):
                vocab[word] = len(vocab)

            add_word(w2id, '<pad>')
            add_word(w2id, '<unk>')
            add_word(w2id, '<bos>')
            add_word(w2id, '<eos>')
            with open(wpath) as fd:
                for line in fd:
                    w, freq = line.strip().split('\t')
                    add_word(w2id, w)
            for w in w2id:
                id2w[w2id[w]] = w
            return w2id, id2w

        def read_one_file(file, vocab, maxlen, minlen):
            with open(file) as fd:
                for line in fd:
                    if not line.isspace():
                        wids = [vocab[w] if w in vocab else vocab['<unk>']
                                for w in list(''.join(line.strip().split()))]
                        if len(wids) < minlen or len(wids) >= maxlen:
                            continue
                        yield ([vocab['<bos>']] + wids + [vocab['<eos>']],
                               wids + [vocab['<eos>'], vocab['<eos>']],
                               [vocab['<bos>'], vocab['<bos>']] + wids,
                               len(wids) + 2)

        w2id, id2w = create_vocab(wordspath)
        if isinstance(filenames, list):
            np.random.shuffle(filenames)
            for f in filenames:
                for d in read_one_file(f, w2id, maxlen, minlen):
                    yield d
        else:
            for d in read_one_file(filenames, w2id, maxlen, minlen):
                yield d