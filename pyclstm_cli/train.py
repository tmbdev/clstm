from __future__ import division

from itertools import chain
import argparse
import glob
import logging
import os
import random
import re
import sys

import pyclstm
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s\t%(message)s")

class OcrTrainer:

    def __init__(self,
                 imgdir,
                 gtdir=None,
                 model_name=None,
                 train_ratio=.8,
                 max_iterations=2000000,
                 num_hidden=50,
                 learning_rate=0.0001,
                 momentum=0.9):
        """Initialize a trainer.

        :param imgdir:          Basedirectory containing '*/*.bin.png' images
        :type imgdir:           str/unicode
        :param max_iterations:  Maximum iterations
        :type max_iterations:   int
        :param gtdir:           Base directory containing '*/*.gt.txt' transcriptions
        :type imgdir:           str/unicode
        :param train_ratio:     Ratio of ground truth touse for training
        :type train_ratio:      float
        :param model_name:      Model basename. 'foo' => 'foo.clstm' or 'foo-1000.clstm'
        :type model_name:       str/unicode
        :param num_hidden:      See pyclstm.pyx
        :param learning_rate:   See pyclstm.pyx
        :param momentum:        See pyclstm.pyx
        """
        if not gtdir:
            gtdir = imgdir
        self.model_name = model_name if model_name else re.sub(r"[^A-Za-z0-9_-]", "", imgdir)
        self.imgdir = imgdir
        all_imgs = [Image.open(p)
                    for p in sorted(glob.glob("{0}/*/*.bin.png".format(imgdir)))
                   ]
        all_texts = [open(p).read().strip()
                     for p in sorted(glob.glob("{0}/*/*.gt.txt".format(gtdir)))]
        if sys.version_info <= (3,):
            all_texts = [t.decode('utf8') for t in all_texts]
        all_data = list(zip(all_imgs, all_texts))
        nr_train = int(len(all_data) * train_ratio)
        self.train_data = all_data[:nr_train]
        self.test_data = all_data[nr_train:]
        self.graphemes = set(chain.from_iterable(all_texts))
        self.ocr = pyclstm.ClstmOcr()
        self.max_iterations = max_iterations
        self.num_hidden = num_hidden
        self.learning_rate = learning_rate
        self.momentum = momentum

    def save_model(self, suffix=''):
        """
        Save the model

        :param suffix: Optional suffix to save intermediary results.
        :param suffix: str/unicode
        """
        fname = self.model_name
        if suffix != '': fname += '-' + suffix
        fname += '.clstm'
        logging.info("Saving model to {}".format(fname))
        self.ocr.save(fname)

    def calculate_error(self):
        """
        Calculate error rate by recognizing test_data and comparing by
        levenshtein distance
        """
        logging.info("Calculating error")
        errors = 0
        chars = 0
        for img, txt in self.test_data:
            out = self.ocr.recognize(img)
            errors += pyclstm.levenshtein(txt, out)
            chars += len(txt)
        return errors / chars

    def train(self,
              log_out=True,
              log_every=10,
              save_every=-1,
              calculate_every=1000):
        """Train from OCR ground truth data.

        :param log_out:         Whether to log 'OUT'
        :param log_every:       Log 'TRN'/'ALN' every N iterations
        :param save_every:      Save every N iterations with suffix -N.clst
        :param calculate_every: Calculate error rate every N iterations and save
        """
        logprog = lambda i, err, msg: logging.info("[{:6d} / {:2.2f}%] {:s}".format(i, (err) * 100, msg))
        self.ocr.prepare_training(
            self.graphemes,
            num_hidden=self.num_hidden,
            learning_rate=self.learning_rate,
            momentum=self.momentum)

        logging.info("Learning from {} [n-train={},n-test={}]".format(
            self.imgdir, len(self.train_data), len(self.test_data)))
        if save_every > -1: logging.info("Saving every {}th iteration".format(save_every))
        logging.info("Evaluating every {}th iteration".format(calculate_every))
        logging.info("Logging everyy {}th iteration".format(log_every))
        err = 1.
        for i in range(self.max_iterations):
            img, txt = random.choice(self.train_data)
            out = self.ocr.train(img, txt)
            if i == 0: continue
            if not i % log_every:
                aligned = self.ocr.aligned()
                logprog(i, err, "TRN: {}".format(txt))
                logprog(i, err, "ALN: {}".format(aligned))
                if log_out: logprog(i, err, "OUT: {}".format(out))
            if not i % calculate_every:
                cur_err = self.calculate_error()
                if cur_err != err:
                    diff = 100*(err-cur_err)
                    logging.info("=== {}ed by {:.2f}%".format("Improv" if diff>0 else "Degrad", diff))
                    self.save_model()
                    err = cur_err
                else:
                    logging.info("No change in error rate")
            elif save_every > 0 and not i % save_every:
                self.save_model(suffix="{:05d}".format(i))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('imgdir', help='Directory matching "*/*.bin.png"')
    parser.add_argument(
        '--model-name', help='Name of the model file without ".clstm". Default: Generated from input name')
    parser.add_argument(
        '--gtdir', help='Directory matching "*/*.gt.txt". Default: imgdir')
    parser.add_argument(
        '--learning-rate',
        metavar='RATE',
        type=float,
        default=0.0001,
        help="Learning rate for the model training. Default: %(default)s")
    parser.add_argument(
        '--momentum',
        type=float,
        default=0.9,
        help="Momentum for the model training. Default: %(default)s")
    parser.add_argument(
        '--num-hidden',
        metavar='NUM',
        type=int,
        default=100,
        help="Number of hidden units in the LSTM layers, larger "
        "values require more storage/memory and take longer "
        "for training and recognition, so try to find "
        "a good performance/cost tradeoff. Default: %(default)s")
    parser.add_argument(
        '--max-iterations',
        metavar='N',
        type=int,
        default=2000000,
        help="Maximum iterations. Default: %(default)s")
    parser.add_argument(
        '--train-ratio',
        metavar='RATIO',
        type=float,
        default=0.8,
        help="Ratio of ground truth to be used for training. Default: %(default)s")
    parser.add_argument(
        '--log-out',
        action='store_true',
        default=False,
        help="Log 'OUT' messages. Default: %(default)s")
    parser.add_argument(
        '--log-every',
        type=int,
        metavar='N',
        default=10,
        help="Log on every N-th iteration. Default: %(default)s")
    parser.add_argument(
        '--save-every',
        type=int,
        metavar='N',
        default=500,
        help="Save on every N-th iteration. Default: %(default)s")
    parser.add_argument(
        '--calculate-every',
        type=int,
        metavar='N',
        default=1000,
        help="Calculate error rate every N-th iteration. Default: %(default)s")
    #  imgdir = '/home/kb/build/github.com/tmbdev/clstm/book'
    args = parser.parse_args()
    trainer = OcrTrainer(
        args.imgdir,
        gtdir=args.gtdir,
        model_name=args.model_name,
        train_ratio=args.train_ratio,
        max_iterations=args.max_iterations,
        num_hidden=args.num_hidden,
        learning_rate=args.learning_rate,
        momentum=args.momentum)
    trainer.train(
        log_out=args.log_out,
        log_every=args.log_every,
        save_every=args.save_every,
        calculate_every=args.calculate_every)

if __name__ == '__main__':
    main()
