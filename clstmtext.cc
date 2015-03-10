#include "clstm.h"
#include <assert.h>
#include <iostream>
#include <vector>
#include <memory>
#include <math.h>
#include <Eigen/Dense>
#include <string>
#include <boost/locale.hpp>
#include <sstream>
#include <string>
#include <fstream>
#include <iostream>
#include <set>

#include "multidim.h"
#include "pymulti.h"
#include "extras.h"
#include "version.h"

using std_string = std::string;
#define string std_string
using std::stoi;
using std::to_string;
using std::vector;
using std::map;
using std::make_pair;
using std::shared_ptr;
using std::unique_ptr;
using std::to_string;
using std::cout;
using std::wstring;
using std::ifstream;
using std::set;
using boost::locale::conv::to_utf;
using boost::locale::conv::from_utf;
using boost::locale::conv::utf_to_utf;
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

void assign(mdarray<int> &dest, vector<int> &src) {
    int n = src.size();
    dest.resize(n);
    for (int i = 0; i < n; i++) dest[i] = src[i];
}

void assign(vector<int> &dest, mdarray<int> &src) {
    int n = src.dim(0);
    dest.resize(n);
    for (int i = 0; i < n; i++) dest[i] = src[i];
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

#if 0
void assign(Classes &classes, mdarray<int> &transcript) {
    classes.resize(transcript.size());
    for (int i = 0; i < transcript.size(); i++)
        classes[i] = transcript(i);
}
#endif

template <class A, class T>
int indexof(A &a, const T &t) {
    for (int i = 0; i < a.size(); i++)
        if (a[i] == t) return i;
    return -1;
}

}

struct Sample {
    wstring in,out;
};

wstring utf32(string s) {
    return to_utf<wchar_t>(s, "UTF-8");
}

string utf8(wstring s) {
    return utf_to_utf<char>(s);
}

void read_samples(vector<Sample> &samples,const string &fname) {
    ifstream stream(fname);
    string line;
    wstring in,out;;
    samples.clear();
    while (getline(stream, line)) {
        int where = line.find("\t");
        if (where<0) throw "no tab found in input line";
        in = utf32(line.substr(0,where));
        out = utf32(line.substr(where+1));
        if (in.size()==0) continue;
        if (out.size()==0) continue;
        samples.push_back(Sample{in, out});
    }
}

void get_codec(mdarray<int> &codec, vector<Sample> &samples, wstring Sample::* p) {
    set<int> codes;
    codes.insert(0);
    for (auto e : samples) {
        for (auto c : e.*p) codes.insert(int(c));
    }
    int n = codes.size();
    codec.resize(n);
    int i = 0;
    for (auto c : codes) codec[i++] = c;
    assert(i==n);
}

void encoder_of_codec(map<wchar_t,int> &encoder, mdarray<int> &codec) {
    for (int i=0; i<codec.dim(0); i++) {
        encoder.insert(make_pair(wchar_t(codec[i]), i));
    }
}

void sequence_of_wstring(Sequence &seq, wstring &s, map<wchar_t,int> &encoder, int d, int neps) {
    seq.clear();
    for(int i=0;i<neps;i++) seq.push_back(Vec::Zero(d));
    for(int pos=0; pos<s.size(); pos++) {
        int c = encoder[s[pos]];
        Vec v = Vec::Zero(d);
        v[c] = 1.0;
        seq.push_back(v);
        for(int i=0;i<neps;i++) seq.push_back(Vec::Zero(d));
    }
}

void classes_of_wstring(Classes &classes, wstring &s, map<wchar_t,int> &encoder) {
    classes.clear();
    for(int pos=0; pos<s.size(); pos++) {
        int c = encoder[s[pos]];
        classes.push_back(c);
    }
}

wstring wstring_of_classes(Classes &classes, mdarray<int> &codec) {
    wstring s;
    for(int pos=0; pos<classes.size(); pos++) {
        s.push_back(wchar_t(codec[classes[pos]]));
    }
    return s;
}

string utfdecode(Classes &classes, mdarray<int> &codec) {
    // FIXME: not utf yet
    string result;
    for(int i=0; i<classes.size(); i++) {
        int c = codec[classes[i]];
        if (c==0) continue;
        if (c>126) c = int('~');
        result.push_back(char(c));
    }
    return result;
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

string string_of_wstring(const wstring &s) {
    string result;
    for (auto c : s) {
        if (int(c)>=127) result.push_back('~');
        else result.push_back(char(int(c)));
    }
    return result;
}

double error_rate(shared_ptr<INetwork> net,const string &testset, map<wchar_t,int> &iencoder, mdarray<int> &codec, int nclasses, int neps) {
    int maxeval = getienv("maxeval", 1000000000);
    vector<Sample> samples;
    read_samples(samples, testset);

    int N = fmin(samples.size(), maxeval);
    double errs = 0.0;
    double total = 0;

    for (int sample = 0; sample < N; sample++) {
        sequence_of_wstring(net->inputs, samples[sample].in, iencoder, nclasses, neps);
        mdarray<float> image;
        assign(image, net->inputs);
        net->forward();
        mdarray<float> outputs;
        assign(outputs, net->outputs);
        Classes output_classes;
        trivial_decode(output_classes, net->outputs);
        wstring gt = samples[sample].out;
        wstring out = wstring_of_classes(output_classes, codec);
        double err = levenshtein(gt, out);
        errs += err;
    }
    return errs/total;
}

int main_train(int argc, char **argv) {
    int randseed = getienv("seed", int(fmod(now()*1e6, 1e9)));
    srand48(randseed);

    const char *textfile = argc > 1 ? argv[1] : "training.txt";
    vector<Sample> samples;
    read_samples(samples, textfile);
    print("got", samples.size(), "lines");
    int nsamples = samples.size();

    string load_name = getsenv("load", "");
    int save_every = getienv("save_every", 0);
    string save_name = getsenv("save_name", "");
    if (save_every>=0 && save_name=="") throw "must give save_name=";
    if (save_every>0 && save_name.find('%')==string::npos)
        save_name += "-%08d.h5";
    else
        save_name += ".h5";
    string after_save = getsenv("after_save", "");

    int ntrain = getienv("ntrain", 1000000);
    double lrate = getrenv("lrate", 1e-4);
    int nhidden = getrenv("hidden", 100);
    int nhidden2 = getrenv("hidden2", -1);
    int batch = getrenv("batch", 1);
    double momentum = getuenv("momentum", 0.9);
    int display_every = getienv("display_every", 0);
    int report_every = getienv("report_every", 1);
    bool randomize = getienv("randomize", 1);
    string lrnorm = getsenv("lrnorm", "batch");
    int neps = int(getuenv("neps", 3));
    string lstm_type = getsenv("lstm", "bidi");

    string testset = getsenv("testset", "");
    int test_every = getienv("test_every", -1);
    string after_test = getsenv("after_test", "");

    print("params",
          "hg_version", hg_version(),
          "lrate", lrate,
          "hidden", nhidden,
          "hidden2", nhidden2,
          "batch", batch,
          "momentum", momentum);

    unique_ptr<PyServer> py;
    if (display_every > 0) {
        py.reset(new PyServer());
        if (display_every > 0) py->open();
        py->eval("ion()");
        py->eval("matplotlib.rc('xtick',labelsize=7)");
        py->eval("matplotlib.rc('ytick',labelsize=7)");
        py->eval("matplotlib.rcParams.update({'font.size':7})");
    }

    shared_ptr<INetwork> net;

    mdarray<int> codec, icodec;
    int nclasses = -1, iclasses = -1;
    if (load_name != "") {
        net = load_net(load_name);
        assign(icodec, net->icodec);
        assign(codec, net->codec);
        nclasses = codec.dim(0);
        iclasses = icodec.dim(0);
        neps = stoi(net->attributes["neps"]);
    } else {
        get_codec(icodec, samples, &Sample::in);
        get_codec(codec, samples, &Sample::out);
        nclasses = codec.dim(0);
        iclasses = icodec.dim(0);
        net = make_net(lstm_type);
        if (lstm_type=="bidi2") {
            net->init(nclasses, nhidden2, nhidden, iclasses);
            print("init-bidi2", nclasses, nhidden2, nhidden, iclasses);
        } else {
            net->init(nclasses, nhidden, iclasses);
            print("init", nclasses, nhidden, iclasses);
        }
        assign(net->icodec, icodec);
        assign(net->codec, codec);
        net->attributes["neps"] = to_string(int(neps));
    }
    net->setLearningRate(lrate, momentum);
    map<wchar_t, int> encoder, iencoder;
    encoder_of_codec(iencoder, icodec);
    encoder_of_codec(encoder, codec);
    print("codec", codec.dim(0), "icodec", icodec.dim(0));
    INetwork::Normalization norm = INetwork::NORM_DFLT;
    if (lrnorm=="len") norm = INetwork::NORM_LEN;
    if (lrnorm=="none") norm = INetwork::NORM_NONE;
    if (norm!=INetwork::NORM_DFLT) print("nonstandard lrnorm: ", lrnorm);
    net->networks("", [norm](string s, INetwork *net) {net->normalization = norm;});

    Sequence targets;
    Sequence saligned;
    Classes classes;

    double start_time = now();
    double best_erate = 1e38;

    int start = stoi(getdef(net->attributes, "trial", getsenv("start", "0")))+1;
    if (start>0) print("start", start);
    for (int trial = start; trial < ntrain; trial++) {
        bool report = (report_every>0) && (trial % report_every == 0);
        int sample = trial % nsamples;
        if (randomize) sample = lrand48() % nsamples;
        if (trial > 0 && save_every > 0 && trial%save_every == 0) {
            char fname[4096];
            sprintf(fname, save_name.c_str(), trial);
            print("saving", fname);
            net->attributes["trial"] = to_string(trial);
            save_net(fname, net);
            if (after_save!="") system(after_save.c_str());
        }
        if (trial > 0 && test_every > 0 && trial%test_every == 0 && testset != "") {
            double erate = error_rate(net, testset, iencoder, codec, nclasses, neps);
            print("TESTERR", now()-start_time, save_name, trial, erate,
                  "lrate", lrate, "hidden", nhidden, nhidden2,
                  "batch", batch, "momentum", momentum);
            if (save_every==0 && erate < best_erate) {
                best_erate = erate;
                print("saving", save_name, "at", erate);
                net->attributes["trial"] = to_string(trial);
                net->attributes["err"] = to_string(best_erate);
                save_net(save_name, net);
                if (after_save!="") system(after_save.c_str());
            }
            if (after_test!="") system(after_test.c_str());
        }
        sequence_of_wstring(net->inputs, samples[sample].in, iencoder, iclasses, neps);
        mdarray<float> image;
        assign(image, net->inputs);
        Classes transcript;
        classes_of_wstring(transcript, samples[sample].out, encoder);
        net->forward();
        classes = transcript;
        mdarray<float> outputs;
        assign(outputs, net->outputs);
        mktargets(targets, classes, nclasses);
        ctc_align_targets(saligned, net->outputs, targets);
        assert(saligned.size() == net->outputs.size());
        net->d_outputs.resize(net->outputs.size());
        for (int t = 0; t < saligned.size(); t++)
            net->d_outputs[t] = saligned[t] - net->outputs[t];
        net->backward();
        if (trial%batch==0) net->update();
        mdarray<float> aligned;
        assign(aligned, saligned);
        if (anynan(outputs) || anynan(aligned)) {
            print("got nan");
            break;
        }
        Classes output_classes, aligned_classes;
        trivial_decode(output_classes, net->outputs);
        trivial_decode(aligned_classes, saligned);
        wstring gt = wstring_of_classes(transcript, codec);
        wstring out = wstring_of_classes(output_classes, codec);
        wstring aln = wstring_of_classes(aligned_classes, codec);
        if (report) {
            wstring s = samples[sample].in;
            print("trial", trial);
            print("INP:", "'"+utf8(s)+"'");
            print("TRU:", "'"+utf8(gt)+"'");
            print("OUT:", "'"+utf8(out)+"'");
            print("ALN:", "'"+utf8(aln)+"'");
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

int main_filter(int argc, char **argv) {
    if (argc!=2) throw "give text file as an argument";
    const char *fname = argv[1];
    string load_name = getsenv("load", "");
    if (load_name=="") throw "must give load= parameter";
    shared_ptr<INetwork> net;
    net = load_net(load_name);
    int neps = stoi(net->attributes["neps"]);
    mdarray<int> codec, icodec;
    assign(icodec, net->icodec);
    assign(codec, net->codec);
    int nclasses = codec.dim(0), iclasses = icodec.dim(0);
    map<wchar_t, int> encoder, iencoder;
    encoder_of_codec(iencoder, icodec);
    encoder_of_codec(encoder, codec);
    dprint("codec", codec.dim(0), "icodec", icodec.dim(0));

    string line;
    wstring in,out;;
    ifstream stream(fname);
    while (getline(stream, line)) {
        in = utf32(line);
        sequence_of_wstring(net->inputs, in, iencoder, iclasses, neps);
        net->forward();
        Classes output_classes;
        trivial_decode(output_classes, net->outputs);
        wstring out = wstring_of_classes(output_classes, codec);
        print(utf8(out));
    }
}

const char *usage = /*program+*/ R"(training.txt

training.txt is a text file consisting of lines of the form:

input\toutput\n

UTF-8 encoding is assumed.

)";

int main(int argc, char **argv) {
    if (argc < 2) {
        print(string(argv[0])+" "+usage);
        exit(1);
    }
    try {
        string mode = getsenv("mode", "train");
        if (mode=="train") {
            return main_train(argc, argv);
        } else if (mode=="filter") {
            return main_filter(argc, argv);
        }
    } catch(const char *msg) {
        print("EXCEPTION", msg);
    } catch(...) {
        print("UNKNOWN EXCEPTION");
    }
}
