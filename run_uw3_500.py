from __future__ import division

import glob
import random
from itertools import chain

import pyclstm
from PIL import Image

all_imgs = [Image.open(p) for p in sorted(glob.glob("./book/*/*.png"))]
all_texts = [open(p).read().strip().decode('utf8')
             for p in sorted(glob.glob("./book/*/*.gt.txt"))]
all_data = zip(all_imgs, all_texts)

train_data = all_data[:400]
test_data = all_data[400:]

ocr = pyclstm.ClstmOcr()
graphemes = set(chain.from_iterable(all_texts))
ocr.prepare_training(graphemes)

for i in range(2000000):
    best_error = 1.
    img, txt = random.choice(train_data)
    out = ocr.train(img, txt)
    if not i % 10:
        aligned = ocr.aligned()
        print("Truth:   {}".format(txt))
        print("Aligned: {}".format(aligned))
        print("Output:  {}".format(out))
    if not i % 1000:
        errors = 0
        chars = 0
        for img, txt in test_data:
            out = ocr.recognize(img)
            errors += pyclstm.levenshtein(txt, out)
            chars += len(txt)
        error = errors / chars
        print ("=== Test set error after {} iterations: {:.2f}"
               .format(i, error))
        if error < best_error:
            print("=== New best error, saving model to model.clstm")
            ocr.save("./model.clstm")
