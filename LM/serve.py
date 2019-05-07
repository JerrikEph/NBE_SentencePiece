from pathlib import Path
from tensorflow.contrib import predictor
import numpy as np


class Serve_VBE:
    def __init__(self, wordspath, export_dir):
        self.w2id, self.id2w = self.create_vocab(wordspath)

        subdirs = [x for x in Path(export_dir).iterdir()
                   if x.is_dir() and 'temp' not in str(x)]
        latest = str(sorted(subdirs)[-1])
        self.predict_fn = predictor.from_saved_model(latest)

    def __call__(self, snt):
        vocab = self.w2id
        snt_lst = list(''.join(snt.strip().split()))
        _wids = [vocab[w] if w in vocab else vocab['<unk>'] for w in snt_lst]
        f_wids = [vocab['<bos>']] + _wids + [vocab['<eos>']]
        f_len = len(f_wids)

        out = self.predict_fn({'f_wids': [f_wids], 'f_len': [f_len]})
        fwd_diff = self.compute_diff_h_fwd(out['fwd_entropy'])[0]
        bwd_diff = self.compute_diff_h_bwd(out['bwd_entropy'])[0]
        scrs, track = self.maxscore(fwd_diff, bwd_diff)
        segs = self.get_segs(track)
        segs = list(reversed(segs))
        seg_snt = []
        for seg in segs:
            seg_snt.append(''.join(snt_lst[seg[0]:seg[1] + 1]))
        return ' '.join(seg_snt)

    @staticmethod
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

    @staticmethod
    def compute_diff_h_fwd(fwd_entropy):
        '''
        Args:
            params fwd_entropy: shape(b_sz, xlen) [[0, 1, 2, e, e]]
                                                   [0, 1, 2, e]
        Return:
            params diff: shape(b_sz, xlen-2)    [[0, 1, 2]]
        '''
        diff = fwd_entropy[:, 1:] - fwd_entropy[:, :-1]
        return diff[:, :-1]

    @staticmethod
    def compute_diff_h_bwd(bwd_entropy):
        '''
        Args:
        params fwd_entropy: shape(b_sz, xlen) [[s, s, 0, 1, 2]]
                                               [s, 0, 1, 2]
        Return:
            params diff: shape(b_sz, xlen-2)    [[0, 1, 2]]
        '''
        diff = bwd_entropy[:, :-1] - bwd_entropy[:, 1:]
        return diff[:, 1:]

    @staticmethod
    def maxscore(fwd_diff, bwd_diff):
        '''
        Args:
            param fwd_diff: shape(xlen)
            param bwd_diff: shape(xlen)
        '''

        def factor(x):
            return 1.0

        #             return x**0.7

        lens = len(fwd_diff)

        socre_bwd = np.expand_dims(np.array(bwd_diff), 1)
        score_fwd = np.expand_dims(np.array(fwd_diff), 0)
        score_ = socre_bwd + score_fwd

        max_score = np.zeros(shape=[lens], dtype=np.float32) - np.float('inf')
        backtrack = np.zeros(shape=[lens], dtype=np.int32)

        for j in range(0, lens):
            _s = [score_[0, j] * factor(j + 1)] + [score_[i + 1, j] * factor(j - i) + max_score[i] for i in range(j)]
            max_score[j] = np.max(_s)
            backtrack[j] = np.argmax(_s)
        return max_score, backtrack

    @staticmethod
    def get_segs(backtract):
        idx = np.arange(len(backtract))
        tup = list(zip(backtract, idx))
        end = len(idx) - 1
        seg = []
        while end >= 0:
            seg.append(tup[end])
            end = tup[end][0] - 1
        return seg


serv_vbe = Serve_VBE(wordspath='words.txt', export_dir='./test04_tune/saved_model')

print(serv_vbe('数学透过抽象化和逻辑推理的使用,由计数、计算、量度和对物体形状及运动的观察而产生。'))
print(serv_vbe('数学家们拓展这些概念,为了公式化新的猜想以及从选定的公理及定义中建立起严谨推导出的定理。'))
print(serv_vbe('数学对这些领域的应用通常被称为应用数学,有时亦会激起新的数学发现,并导致全新学科的发展,例如物理学的实质性发展中建立的某些理论激发数学家对于某些问题的不同角度的思考。'))
print(serv_vbe('数学对这些领域的应用通常被称为应用数学,有时亦会激起新的数学发现'))