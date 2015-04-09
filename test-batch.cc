#include "clstm.h"
#include <assert.h>
#include <iostream>
#include <vector>
#include <memory>
#include <math.h>
#include <Eigen/Dense>
#include <string>

#include "multidim.h"
#include "pymulti.h"
#include "extras.h"
#include "version.h"

using std_string = std::string;
#define string std_string
using std::vector;
using std::shared_ptr;
using std::unique_ptr;
using std::to_string;
using std::make_pair;
using std::cout;
using std::stoi;
using namespace Eigen;
using namespace ocropus;
using namespace pymulti;

double error_rate(shared_ptr<INetwork> net,const string &testset) {
    int maxeval = getienv("maxeval", 1000000000);
    shared_ptr<IOcrDataset> dataset(make_Dataset(testset));

    mdarray<float> image;
    mdarray<int> transcript;
    Classes classes;

    double total = 0;
    double errs = 0;

    int N = min(dataset->samples(), maxeval);

    for (int sample = 0; sample < N; sample++) {
        dataset->image(image, sample);
        dataset->transcript(transcript, sample);
        assign(net->inputs, image);
        net->forward();
        Classes output_classes;
        trivial_decode(output_classes, net->outputs);
        string gt = dataset->to_string(transcript);;
        string out = dataset->to_string(output_classes);
        total += gt.size();
        double err = levenshtein(gt, out);
        errs += err;
    }
    return errs/total;
}

void batched_of_sequences(Sequence &batched, vector<Sequence> &sequences) {
    int bs = sequences.size();
    assert(bs>0);
    int d = sequences[0][0].rows();
    for (auto s : sequences) assert(s[0].cols()==1);
    assert(sequences[0].size()>0);
    int N = 0;
    for (int b=0; b<bs; b++) { if (sequences[b].size()>N) N = sequences[b].size(); }
    // print("---", bs, d, N, sequences[0].size());
    assert(N>0);
    batched.resize(N);
    for (int i=0;i<N;i++) batched[i] = Mat::Zero(d, bs);
    for (int b=0; b<bs; b++) {
        Sequence &s = sequences[b];
        for (int t=0; t<s.size(); t++) batched[t].col(b) = s[t].col(0);
    }
}

void sequences_of_batched(vector<Sequence> &sequences, Sequence &batched) {
    int bs = batched[0].cols();
    int d = batched[0].rows();
    int N = batched.size();
    sequences.resize(bs);
    for (int b=0; b<bs; b++) {
        sequences[b].resize(N);
        for (int t=0;t<N;t++) {
            sequences[b][t].resize(d,1);
            sequences[b][t].col(0) = batched[t].col(b);
        }
    }
}

struct BatchBuilder {
    vector<Sequence> inputs;
    vector<Classes> classes;
    vector<Sequence> outputs;
    vector<Sequence> aligned;
};

int main(int argc, char **argv) {
    srandomize();

    const char *h5file = argc > 1 ? argv[1] : "mnist_seq.h5";
    string load_name = getsenv("load", "");
    int ntrain = getienv("ntrain", 1000000);
    double lrate = getrenv("lrate", 1e-4);
    int batchsize = getrenv("batch", 16);
    double momentum = getuenv("momentum", 0.9);
    //int display_every = getienv("display_every", 0);
    int report_every = getienv("report_every", 100);
    //bool randomize = getienv("randomize", 1);
    string lrnorm = getsenv("lrnorm", "batch");
    string dewarp = getsenv("dewarp", "none");
    string lstm_type = getsenv("lstm", "bidi");

    string testset = getsenv("testset", "mnist_test_seq.h5");
    int test_every = getienv("test_every", 10000);
    string after_test = getsenv("after_test", "");

    shared_ptr<IOcrDataset> dataset;
    dataset.reset(make_Dataset(h5file));
    print("dataset", dataset->samples(), dataset->dim(), dewarp);

    shared_ptr<INetwork> net;
    int nclasses = -1;
    int dim = dataset->dim();
    if (load_name != "") {
        net = load_net(load_name);
        nclasses = net->codec.size();
    } else {
        vector<int> codec;
        dataset->getCodec(codec);
        nclasses = codec.size();
        net = make_net_init(lstm_type, nclasses, dim);
    }
    net->setLearningRate(lrate, momentum);
    dataset->getCodec(net->codec);
    // if (load_name != "") net->load(load_name.c_str());
    INetwork::Normalization norm = INetwork::NORM_DFLT;
    if (lrnorm=="len") norm = INetwork::NORM_LEN;
    if (lrnorm=="none") norm = INetwork::NORM_NONE;
    if (norm!=INetwork::NORM_DFLT) print("nonstandard lrnorm: ", lrnorm);
    net->networks("", [norm](string s, INetwork *net) {net->normalization = norm;});

    //double start_time = now();
    //double best_erate = 1e38;

    int next_test = test_every;
    int start = stoi(getdef(net->attributes, "trial", getsenv("start", "-1")))+1;
    if (start>0) print("start", start);
    for (int trial = start; trial < ntrain; trial+=batchsize) {
        BatchBuilder batch;
        for (int i=0;i<batchsize; i++) {
            int sample = irandom() % dataset->samples();
            mdarray<float> image;
            mdarray<int> transcript;
            dataset->image(image, sample);
            Sequence inputs;
            assign(inputs, image);
            batch.inputs.push_back(inputs);
            dataset->transcript(transcript, sample);
            Classes classes;
            assign(classes, transcript);
            batch.classes.push_back(classes);
        }

        batched_of_sequences(net->inputs, batch.inputs);
        net->forward();
        sequences_of_batched(batch.outputs, net->outputs);
        batch.aligned.resize(batch.outputs.size());
        for (int b=0; b<batch.aligned.size(); b++) {
            Sequence targets;
            mktargets(targets, batch.classes[b], net->noutput());
            ctc_align_targets(batch.aligned[b], batch.outputs[b], targets);
        }
        batched_of_sequences(net->d_outputs, batch.aligned);
        for (int t = 0; t < net->d_outputs.size(); t++)
            net->d_outputs[t] -= net->outputs[t];
        net->backward();
        net->update();

        Classes output_classes, aligned_classes;
        trivial_decode(output_classes, batch.outputs[0]);
        trivial_decode(aligned_classes, batch.aligned[0]);
        string gt = dataset->to_string(batch.classes[0]);
        string out = dataset->to_string(output_classes);
        string aln = dataset->to_string(aligned_classes);

        if (int(trial/batchsize) % report_every == 0) {
            print(trial);
            print("TRU:", "'"+gt+"'");
            print("OUT:", "'"+out+"'");
            print("ALN:", "'"+aln+"'");
            print(levenshtein(gt,out));
        }
        if (trial > next_test) {
            double erate = error_rate(net, testset);
            print("TESTERR ", trial, erate);
            next_test += test_every;
        }
    }
    return 0;
}
