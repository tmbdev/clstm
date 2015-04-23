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

double error_rate(shared_ptr<INetwork> net, const string &testset) {
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
    srandomize();

    const char *h5file = argc > 1 ? argv[1] : "uw3-dew.h5";
    string load_name = getsenv("load", "");

    int save_every = getienv("save_every", 0);
    string save_name = getsenv("save_name", "");
    if (save_every >= 0 && save_name == "") throw "must give save_name=";
    if (save_every > 0 && save_name.find('%') == string::npos)
        save_name += "-%08d";
    else
        save_name += ".h5";
    string after_save = getsenv("after_save", "");
    string after_start = getsenv("after_start", "");

    int ntrain = getienv("ntrain", 1000000);
    double lrate = getrenv("lrate", 1e-4);
    int nhidden = getrenv("nhidden", getrenv("hidden", 100));
    int nhidden2 = getrenv("nhidden2", getrenv("hidden2", -1));
    int pseudo_batch = getrenv("pseudo_batch", 1);
    double momentum = getuenv("momentum", 0.9);
    int display_every = getienv("display_every", 0);
    int report_every = getienv("report_every", 1);
    bool randomize = getienv("randomize", 1);
    string lrnorm = getoneof("lrnorm", "batch");
    string dewarp = getoneof("dewarp", "none");
    string net_type = getoneof("lstm", "BIDILSTM");
    string lstm_type = getoneof("lstm_type", "LSTM");
    string output_type = getoneof("output_type", "SoftmaxLayer");
    string target_name = getoneof("target_name", "ctc");

    string testset = getsenv("testset", "");
    int test_every = getienv("test_every", -1);
    string after_test = getsenv("after_test", "");

    print("params",
          "hg_version", hg_version(),
          "lrate", lrate,
          "nhidden", nhidden,
          "nhidden2", nhidden2,
          "pseudo_batch", pseudo_batch,
          "momentum", momentum,
          "type", net_type, lstm_type, output_type);

    if (getienv("params_only", 0)) return 0;

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
        net = make_net(net_type);
        net->set("ninput", dim);
        net->set("noutput", nclasses);
        net->set("nhidden", nhidden);
        net->set("nhidden2", nhidden2);
        net->set("lstm_type", lstm_type);
        net->set("output_type", output_type);
        net->initialize();
    }
    net->setLearningRate(lrate, momentum);
    if (getienv("info", 0)) net->info("");
    dataset->getCodec(net->codec);
    // if (load_name != "") net->load(load_name.c_str());
    INetwork::Normalization norm = INetwork::NORM_DFLT;
    if (lrnorm == "len") norm = INetwork::NORM_LEN;
    if (lrnorm == "none") norm = INetwork::NORM_NONE;
    if (norm != INetwork::NORM_DFLT) print("nonstandard lrnorm: ", lrnorm);
    net->networks("", [norm](string s, INetwork *net) {net->normalization = norm; });

    mdarray<float> raw_image, image, outputs, aligned;
    mdarray<int> transcript;
    Sequence targets;
    Sequence saligned;
    Classes classes;

    double start_time = now();
    double best_erate = 1e38;

    int start = stoi(getdef(net->attributes, "trial", getsenv("start", "-1")))+1;
    if (start > 0) print("start", start);
    if (after_start != "") system(after_start.c_str());
    for (int trial = start; trial < ntrain; trial++) {
        bool report = (report_every > 0) && (trial % report_every == 0);
        int sample = trial % dataset->samples();
        if (randomize) sample = irandom() % dataset->samples();
        if (trial > 0 && save_every > 0 && trial%save_every == 0) {
            char fname[4096];
            sprintf(fname, save_name.c_str(), trial);
            print("saving", fname);
            net->attributes["trial"] = to_string(trial);
            save_net(fname, net);
            if (after_save != "") system(after_save.c_str());
        }
        if (trial > 0 && test_every > 0 && trial%test_every == 0 && testset != "") {
            double erate = error_rate(net, testset);
            net->attributes["trial"] = to_string(trial);
            net->attributes["last_err"] = to_string(best_erate);
            print("TESTERR", now()-start_time, save_name, trial, erate,
                  "lrate", lrate, "hidden", nhidden, nhidden2,
                  "pseudo_batch", pseudo_batch, "momentum", momentum);
            if (save_every == 0 && erate < best_erate) {
                best_erate = erate;
                print("saving", save_name, "at", erate);
                save_net(save_name, net);
                if (after_save != "") system(after_save.c_str());
            }
            if (after_test != "") system(after_test.c_str());
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
        assign(outputs, net->outputs);
        if (target_name == "ctc") {
            assign(classes, transcript);
            mktargets(targets, classes, dataset->classes());
            ctc_align_targets(saligned, net->outputs, targets);
            assign(aligned, saligned);
        } else {
            // Use a pre-aligned sequence; this is intended for testing
            // CTC vs non-CTC performance. For general sequence training,
            // use clstmseq
            dataset->seq(aligned, sample, target_name);
            assign(saligned, aligned);
        }
        if (anynan(outputs) || anynan(aligned)) {
            print("got nan");
            break;
        }
        assert(saligned.size() == net->outputs.size());
        net->d_outputs.resize(net->outputs.size());
        for (int t = 0; t < saligned.size(); t++)
            net->d_outputs[t] = saligned[t] - net->outputs[t];
        net->backward();
        if (trial%pseudo_batch == 0) net->update();
        Classes output_classes, aligned_classes;
        trivial_decode(output_classes, net->outputs);
        trivial_decode(aligned_classes, saligned);
        string gt = dataset->to_string(transcript);;
        string out = dataset->to_string(output_classes);
        string aln = dataset->to_string(aligned_classes);
        if (report) {
            print("OUT:", "'"+out+"'");
            print("ALN:", "'"+aln+"'");
            print(levenshtein(gt, out));
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
                v(t) = net->outputs[t].col(0).segment(2, nclass-2).maxCoeff();
            py->evalf("xlim(0,%d)", outputs.dim(0));
            py->plot(v, "color='r'");
            py->eval("ginput(1,1e-3)");
        }
    }
    return 0;
}

int main_eval(int argc, char **argv) {
    const char *h5file = argc > 1 ? argv[1] : "uw3-dew-test.h5";
    string mode = getsenv("mode", "errs");
    string load_name = getsenv("load", "");
    shared_ptr<IOcrDataset> dataset(make_Dataset(h5file));
    shared_ptr<INetwork> net;
    if (load_name == "") throw "must give load=";
    net = load_net(load_name);

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
        double err = levenshtein(gt, out);
        errs += err;
        if  (mode == "quiet") {
            // do nothing
        } else if (mode == "errs") {
            print(to_string(int(err))+"\t"+out);
        } else if (mode == "text") {
            print(to_string(sample)+"\t"+out);
        } else if (mode == "full") {
            cout << int(err) << "\t";
            cout << int(sample) << "\t";
            cout << out << "\t";
            cout << gt << "\n";
        }
        cout.flush();
    }
    print("errs", errs, "total", total, "rate", errs*100.0/total, "%");
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
    srandomize();
    if (argc != 2) throw "usage: ... image.png";
    mdarray<unsigned char> raw;
    mdarray<float> image, dewarped;
    read_png(raw, argv[1]);
    print("raw", raw.dim(0), raw.dim(1));
    image.resize(raw.dim(0), raw.dim(1));
    for (int i = 0; i < raw.dim(0); i++) {
        for (int j = 0; j < raw.dim(1); j++) {
            int jj = raw.dim(1)-j-1;
            if (raw.rank() == 2) image(i, jj) = 1.0-raw(i, j)/255.0;
            else image(i, jj) = 1.0-raw(i, j, 0)/255.0;
        }
    }
    PyServer *py = new PyServer();
    py->open();
    unique_ptr<INormalizer> normalizer;
    normalizer.reset(make_CenterNormalizer());
    normalizer->target_height = int(getrenv("target_height", 48));
    normalizer->getparams(true);
    // normalizer->setPyServer(py);
    py->eval("ion()");
    py->eval("clf()");
    normalizer->measure(image);
    py->eval("subplot(211)");
    py->imshowT(image, "cmap=cm.gray,interpolation='nearest'");
    py->eval("subplot(212)");
    normalizer->normalize(dewarped, image);
    py->imshowT(dewarped, "cmap=cm.gray,interpolation='nearest'");
    return 0;
}

const char *usage =
    "data.h5\n\n"
    "data.h5 is an HDF5 file containing:\n"
    "float images(N,*): text line images (or sequences of vectors)\n"
    "int images_dims(N,2): shape of the images\n"
    "int transcripts(N,*): corresponding transcripts\n";

int main(int argc, char **argv) {
    if (argc < 2) {
        print(string(argv[0])+" "+usage);
        exit(1);
    }
    try {
        string mode = getsenv("mode", "train");
        if (getienv("eval", 0)) {  // for old scripts
            return main_eval(argc, argv);
        }
        if (mode == "dump") {
            return main_dump(argc, argv);
        } else if (mode == "train") {
            return main_ocr(argc, argv);
        } else if (mode == "testdewarp") {
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

