#!/usr/bin/python
from numpy import *
from pylab import *
from numpy.random import rand
import clstm
net = clstm.make_net_init("lstm1", "ninput=1:nhidden=4:noutput=2")
net.setLearningRate(1e-4, 0.9)
N = 20
ntrain = 30000
ntest = 1000
print("training 1:4:2 network to learn delay")
for i in range(ntrain):
    xs = array(rand(N) < 0.3, 'f')
    ys = roll(xs, 1)
    ys[0] = 0
    ys = array([1 - ys, ys], 'f').T.copy()
    net.inputs.aset(xs.reshape(N, 1, 1))
    net.forward()
    net.outputs.dset(ys.reshape(N, 2, 1) - net.outputs.array())
    net.backward()
    clstm.sgd_update(net)
print("testing", ntest, "random instances")
maxerr = 0.0
for i in range(ntest):
    xs = array(rand(N) < 0.3, 'f')
    ys = roll(xs, 1)
    ys[0] = 0
    net.inputs.aset(xs.reshape(N, 1, 1))
    net.forward()
    preds = net.outputs.array()[:, 1, 0]
    err = abs(amax(abs(ys - preds)))
    assert (err < 0.1), err
    maxerr = maximum(err, maxerr)
print("OK", maxerr)
