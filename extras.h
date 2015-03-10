#ifndef clstm_extras_
#define clstm_extras_

#include "multidim.h"
#include <string>
#include <sys/time.h>
#include <math.h>
#include <stdlib.h>
#include <map>

extern "C" { double drand48(); }

namespace ocropus {
using std::string;
using std::shared_ptr;
using std::vector;
using std::cout;
using std::cerr;
using std::endl;
using std::min;
using namespace multidim;

// get current time down to usec precision as a double

inline double now() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + 1e-6 * tv.tv_usec;
}

// print the arguments to cout

template <class T>
inline void print(const T &arg) {
    cout << arg << endl;
}

template <class T, typename ... Args>
inline void print(T arg, Args ... args) {
    cout << arg << " ";
    print(args ...);
}

inline string getdef(std::map<string, string> &m, const string &key, const string &dflt) {
    auto it = m.find(key);
    if (it==m.end()) return dflt;
    return it->second;
}

// print the arguments to cerr

template <class T>
inline void dprint(const T &arg) {
    cerr << arg << endl;
}

template <class T, typename ... Args>
inline void dprint(T arg, Args ... args) {
    cerr << arg << " ";
    dprint(args ...);
}

// get values from the environment, with defaults

inline const char *getsenv(const char *name, const char *dflt) {
    const char *result = getenv(name);
    if (result) return result;
    return dflt;
}

inline int getienv(const char *name, int dflt=0) {
    const char *result = getenv(name);
    if (result) return atoi(result);
    return dflt;
}

inline double getdenv(const char *name, double dflt=0) {
    const char *result = getenv(name);
    if (result) return atof(result);
    return dflt;
}

// get a value or random value from the environment (var=7.3 or var=2,8)

inline double getrenv(const char *name, double dflt=0, bool logscale=true) {
    const char *s = getenv(name);
    if (!s) return dflt;
    float lo,hi;
    if (sscanf(s, "%g,%g", &lo, &hi)==2) {
        double x = exp(log(lo)+drand48()*(log(hi)-log(lo)));
        return x;
    } else if (sscanf(s, "%g", &lo)==1) {
        return lo;
    } else {
        throw "bad format for getrenv";
    }
}

inline double getuenv(const char *name, double dflt=0) {
    const char *s = getenv(name);
    if (!s) return dflt;
    float lo,hi;
    if (sscanf(s, "%g,%g", &lo, &hi)==2) {
        double x = lo+drand48()*(hi-lo);
        return x;
    } else if (sscanf(s, "%g", &lo)==1) {
        return lo;
    } else {
        throw "bad format for getuenv";
    }
}

// array minimum, maximum

template <class T>
T amin(mdarray<T> &a) {
    T m = a[0];
    for (int i = 1; i < a.size(); i++) if (a[i] < m) m = a[i];
    return m;
}

template <class T>
T amax(mdarray<T> &a) {
    T m = a[0];
    for (int i = 1; i < a.size(); i++) if (a[i] > m) m = a[i];
    return m;
}

// text line normalization

struct INormalizer {
    int target_height = 48;
    float smooth2d = 1.0;
    float smooth1d = 0.3;
    float range = 4.0;
    float vscale = 1.0;
    virtual ~INormalizer() {}
    virtual void getparams(bool verbose=false) {}
    virtual void measure(mdarray<float> &line) = 0;
    virtual void normalize(mdarray<float> &out, mdarray<float> &in) = 0;
    virtual void setPyServer(void *p) {}
};

INormalizer *make_Normalizer(const string &);
INormalizer *make_NoNormalizer();
INormalizer *make_MeanNormalizer();
INormalizer *make_CenterNormalizer();

// OCR dataset access, including datasets that are normalized
// on the fly

struct IOcrDataset {
    virtual ~IOcrDataset() {}
    virtual void image(mdarray<float> &a, int index) = 0;
    virtual void transcript(mdarray<int> &a, int index) = 0;
    virtual string to_string(mdarray<int> &transcript) = 0;
    virtual string to_string(vector<int> &transcript) = 0;
    virtual void getCodec(vector<int> &codec) = 0;
    virtual int samples() = 0;
    virtual int dim() = 0;
    virtual int classes() = 0;
};

IOcrDataset *make_HDF5Dataset(const string &fname, bool varsize=false);
IOcrDataset *make_NormalizedDataset(shared_ptr<IOcrDataset> &dataset,
                                    shared_ptr<INormalizer> &normalizer);
IOcrDataset *make_Dataset(const string &fname);

void read_png(mdarray<unsigned char> &image,FILE *fp,bool gray=false);
void write_png(FILE *fp,mdarray<unsigned char> &image);
void read_png(mdarray<unsigned char> &image,const char *name,bool gray=false);
void write_png(const char *name,mdarray<unsigned char> &image);

}

#endif
