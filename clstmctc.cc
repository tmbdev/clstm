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

using std_string = std::string;
#define string std_string
using std::vector;
using std::shared_ptr;
using std::unique_ptr;
using std::to_string;
using std::cout;
using namespace Eigen;
using namespace ocropus;
using namespace pymulti;

namespace {
template <class S, class T>
void assign(S &dest, T &src) {
    dest.resize_(src.dims);
    int n = dest.size();
    for (int i = 0; i < n; i++) dest.data[i] = src.data[i];
}

template <class S, class T>
void transpose(S &dest, T &src) {
    dest.resize(src.dim(1), src.dim(0));
    for (int i = 0; i < dest.dim(0); i++)
        for (int j = 0; j < dest.dim(1); j++)
            dest(i, j) = src(j, i);
}

template <class T>
void transpose(T &a) {
    T temp;
    transpose(temp, a);
    assign(a, temp);
}

template <class T>
void assign(Sequence &seq, T &a) {
    assert(a.rank() == 2);
    seq.resize(a.dim(0));
    for (int t = 0; t < a.dim(0); t++) {
        seq[t].resize(a.dim(1));
        for (int i = 0; i < a.dim(1); i++) seq[t](i) = a(t, i);
    }
}

template <class T>
void assign(T &a, Sequence &seq) {
    a.resize(int(seq.size()), int(seq[0].size()));
    for (int t = 0; t < a.dim(0); t++) {
        for (int i = 0; i < a.dim(1); i++) a(t, i) = seq[t](i);
    }
}

void assign(Classes &classes, mdarray<int> &transcript) {
    classes.resize(transcript.size());
    for (int i = 0; i < transcript.size(); i++)
        classes[i] = transcript(i);
}

template <class A, class T>
int indexof(A &a, const T &t) {
    for (int i = 0; i < a.size(); i++)
        if (a[i] == t) return i;
    return -1;
}

}

void debug_decode(Sequence &outputs, Sequence &aligned) {
    for (int t = 0; t < outputs.size(); t++) {
        int oindex, aindex;
        outputs[t].maxCoeff(&oindex);
        aligned[t].maxCoeff(&aindex);
        print(t,
              "outputs", outputs[t](0), outputs[t](1),
              oindex, outputs[t](oindex),
              "aligned", aligned[t](0), aligned[t](1),
              aindex, aligned[t](aindex));
    }
}

void trivial_decode(Classes &cs, Sequence &outputs) {
    int N = outputs.size();
    int t = 0;
    float mv = 0;
    int mc = -1;
    while (t < N) {
        int index;
        float v = outputs[t].maxCoeff(&index);
        if (index == 0) {
            // NB: there should be a 0 at the end anyway
            if (mc != -1) cs.push_back(mc);
            mv = 0; mc = -1; t++;
            continue;
        }
        if (v > mv) {
            mv = v;
            mc = index;
        }
        t++;
    }
}

bool anynan(mdarray<float> &a) {
    for (int i = 0; i < a.size(); i++)
        if (isnan(a[i])) return true;
    return false;
}

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

int main_ocr(int argc, char **argv) {
    const char *h5file = argc > 1 ? argv[1] : "uw3-dew.h5";
    string load_name = getsenv("load", "");
    double lrate = getdenv("lrate", 1e-4);

    int save_every = getienv("save_every", 0);
    string save_name = getsenv("save_name", "");
    if (save_every>=0 && save_name=="") throw "must give save_name=";
    if (save_every>0 && save_name.find('%')==string::npos)
        save_name += "-%08d";
    else
        save_name += ".h5";
    string after_save = getsenv("after_save", "");

    int start = getienv("start", 0);
    int ntrain = getienv("ntrain", 1000000);
    int nhidden = getienv("hidden", 100);
    int nhidden2 = getienv("hidden2", -1);
    int display_every = getienv("display_every", 0);
    int report_every = getienv("report_every", 1);
    int randseed = getienv("seed", int(fmod(now()*1e6, 1e9)));
    bool randomize = getienv("randomize", 1);
    string normalization = getsenv("normalization", "batch");

    string testset = getsenv("testset", "");
    int test_every = getienv("test_every", -1);
    string after_test = getsenv("after_test", "");

    srand48(randseed);

    unique_ptr<PyServer> py;
    if (display_every > 0) {
        py.reset(new PyServer());
        if (display_every > 0) py->open();
        py->eval("ion()");
        py->eval("matplotlib.rc('xtick',labelsize=7)");
        py->eval("matplotlib.rc('ytick',labelsize=7)");
        py->eval("matplotlib.rcParams.update({'font.size':7})");
    }

    shared_ptr<IOcrDataset> dataset;
    dataset.reset(make_Dataset(h5file));
    print("dataset", dataset->samples(), dataset->dim());

    shared_ptr<INetwork> net;
    if (nhidden2<0) {
        net.reset(make_BIDILSTM());
        net->init(dataset->classes(), nhidden, dataset->dim());
    } else {
        net.reset(make_BIDILSTM2());
        net->init(dataset->classes(), nhidden2, nhidden, dataset->dim());
    }
    net->setLearningRate(lrate, 0.9);
    dataset->getCodec(net->codec);
    if (load_name != "") net->load(load_name.c_str());
    INetwork::Normalization norm = normalization=="len" ? INetwork::NORM_LEN : INetwork::NORM_DFLT;
    if (norm!=INetwork::NORM_DFLT) print("nonstandard normalization: ", normalization);
    net->networks("", [norm](string s, INetwork *net) {net->normalization = norm;});

    mdarray<float> raw_image, image, outputs, aligned;
    mdarray<int> transcript;
    Sequence targets;
    Sequence saligned;
    Classes classes;

    double start_time = now();
    double best_erate = 1e38;

    for (int trial = start; trial < ntrain; trial++) {
        bool report = (report_every>0) && (trial % report_every == 0);
        int sample = trial % dataset->samples();
        if (randomize) sample = lrand48() % dataset->samples();
        if (trial > 0 && save_every > 0 && trial%save_every == 0) {
            char fname[4096];
            sprintf(fname, save_name.c_str(), trial);
            print("saving", fname);
            net->save(fname);
            if (after_save!="") system(after_save.c_str());
        }
        if (trial > 0 && test_every > 0 && trial%test_every == 0 && testset != "") {
            double erate = error_rate(net, testset);
            print("TESTERR", now()-start_time, save_name, trial, erate,
                  "lrate", lrate, "hidden", nhidden, nhidden2);
            if (save_every==0 && erate < best_erate) {
                best_erate = erate;
                print("saving", save_name, "at", erate);
                net->save(save_name.c_str());
                if (after_save!="") system(after_save.c_str());
            }
            if (after_test!="") system(after_test.c_str());
        }
        dataset->image(image, sample);
        dataset->transcript(transcript, sample);
        if (report) {
            print(trial, sample,
                  "dim", image.dim(0), image.dim(1),
                  "time", now()-start_time,
                  "lrate", lrate, "hidden", nhidden, nhidden2);
            print("TRU:", "'"+dataset->to_string(transcript)+"'");
        }
        assign(net->inputs, image);
        net->forward();
        assign(classes, transcript);
        assign(outputs, net->outputs);
        mktargets(targets, classes, dataset->classes());
        ctc_align_targets(saligned, net->outputs, targets);
        assert(saligned.size() == net->outputs.size());
        net->d_outputs.resize(net->outputs.size());
        for (int t = 0; t < saligned.size(); t++)
            net->d_outputs[t] = saligned[t] - net->outputs[t];
        net->backward();
        assign(aligned, saligned);
        if (anynan(outputs) || anynan(aligned)) {
            print("got nan");
            break;
        }
        Classes output_classes, aligned_classes;
        trivial_decode(output_classes, net->outputs);
        trivial_decode(aligned_classes, saligned);
        string gt = dataset->to_string(transcript);;
        string out = dataset->to_string(output_classes);
        string aln = dataset->to_string(aligned_classes);
        if (report) {
            print("OUT:", "'"+out+"'");
            print("ALN:", "'"+aln+"'");
            print(levenshtein(gt,out));
        }

        if (display_every > 0 && trial%display_every == 0) {
            net->d_outputs.resize(saligned.size());
            py->eval("clf()");
            py->subplot(4, 1, 1);
            py->evalf("title('%s')", gt.c_str());
            py->imshowT(image, "cmap=cm.gray,interpolation='bilinear'");
            py->subplot(4, 1, 2);
            py->evalf("title('%s')", out.c_str());
            py->imshowT(outputs, "cmap=cm.hot,interpolation='bilinear'");
            py->subplot(4, 1, 3);
            py->evalf("title('%s')", aln.c_str());
            py->imshowT(aligned, "cmap=cm.hot,interpolation='bilinear'");
            py->subplot(4, 1, 4);
            mdarray<float> v;
            v.resize(outputs.dim(0));
            for (int t = 0; t < outputs.dim(0); t++)
                v(t) = outputs(t, 0);
            py->plot(v, "color='b'");
            int sp = 1;
            for (int t = 0; t < outputs.dim(0); t++)
                v(t) = outputs(t, sp);
            py->plot(v, "color='g'");
            int nclass = net->outputs[0].size();
            for (int t = 0; t < outputs.dim(0); t++)
                v(t) = net->outputs[t].segment(2, nclass-2).maxCoeff();
            py->evalf("xlim(0,%d)", outputs.dim(0));
            py->plot(v, "color='r'");
            py->eval("ginput(1,1e-3)");
        }
    }
    return 0;
}

int main_eval(int argc, char **argv) {
    const char *h5file = argc > 1 ? argv[1] : "uw3-dew-test.h5";
    string mode = getsenv("mode","errs");
    string load_name = getsenv("load", "");
    shared_ptr<IOcrDataset> dataset(make_Dataset(h5file));
    shared_ptr<INetwork> net(make_BIDILSTM());
    int nhidden = 17;
    net->init(dataset->classes(), nhidden, dataset->dim());
    if(load_name=="") throw "must give load=";
    net->load(load_name.c_str());

    mdarray<float> image;
    mdarray<int> transcript;
    Classes classes;

    double total = 0;
    double errs = 0;

    for (int sample = 0; sample < dataset->samples(); sample++) {
        dataset->image(image, sample);
        dataset->transcript(transcript, sample);
        assign(net->inputs, image);
        net->forward();
        Classes output_classes;
        trivial_decode(output_classes, net->outputs);
        string gt = dataset->to_string(transcript);;
        string out = dataset->to_string(output_classes);
        total += gt.size();
        double err = levenshtein(gt,out);
        errs += err;
        if  (mode=="quiet") {
            // do nothing
        } else if(mode=="errs") {
            print(to_string(int(err))+"\t"+out);
        } else if(mode=="text") {
            print(to_string(sample)+"\t"+out);
        } else if(mode=="full") {
            cout << int(err) << "\t";
            cout << int(sample) << "\t";
            cout << out << "\t";
            cout << gt << "\n";
        }
        cout.flush();
    }
    print("errs",errs,"total",total,"rate",errs*100.0/total);
    cout.flush();
    return 0;
}

int main_dump(int argc, char **argv) {
    const char *h5file = argc > 1 ? argv[1] : "uw3-dew-test.h5";
    shared_ptr<IOcrDataset> dataset(make_Dataset(h5file));
    for (int sample = 0; sample < dataset->samples(); sample++) {
        mdarray<int> transcript;
        dataset->transcript(transcript, sample);
        string gt = dataset->to_string(transcript);;
        print(to_string(sample)+"\t"+gt);
        cout.flush();
    }
    return 0;
}

int main_testdewarp(int argc, char **argv) {
    if (argc!=3) throw "usage: ... hdf5 index";
    shared_ptr<IOcrDataset> dataset(make_HDF5Dataset(argv[1],true));
    int index = atoi(argv[2]);
    mdarray<float> image, dewarped;
    unique_ptr<PyServer> py;
    py.reset(new PyServer());
    py->open();
    py->eval("ion()");
    py->eval("clf()");
    dataset->image(image, index);
    unique_ptr<INormalizer> normalizer;
    normalizer.reset(make_MeanNormalizer());
    normalizer->measure(image);
    py->eval("subplot(211)");
    py->imshowT(image, "cmap=cm.gray,interpolation='nearest'");
    py->eval("subplot(212)");
    normalizer->normalize(dewarped, image);
    py->imshowT(dewarped, "cmap=cm.gray,interpolation='nearest'");
    return 0;
}

const char *usage = /*program+*/ R"(data.h5

data.h5 is an HDF5 file containing:

float images(N,*): text line images (or sequences of vectors)
int images_dims(N,2): shape of the images
int transcripts(N,*): corresponding transcripts
)";

int main(int argc, char **argv) {
    if (argc < 2) {
        print(string(argv[0])+" "+usage);
        exit(1);
    }
    try {
        string mode = getsenv("mode", "train");
        if (getienv("eval", 0)) { // for old scripts
            return main_eval(argc, argv);
        }
        if (mode=="dump") {
            return main_dump(argc, argv);
        } else if (mode=="train") {
            return main_ocr(argc, argv);
        } else if (mode=="testdewarp") {
            return main_testdewarp(argc, argv);
        } else {
            return main_eval(argc, argv);
        }
    } catch(const char *msg) {
        print("EXCEPTION", msg);
    } catch(...) {
        print("UNKNOWN EXCEPTION");
    }
}
