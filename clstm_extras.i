namespace ocropus {

template <class F, class G, class H, bool PEEP = true>
struct GenericLSTM : INetwork {
    // NB: verified gradients against Python implementation; this
    // code yields identical numerical results
#define SEQUENCES gix, gfx, gox, cix, gi, gf, go, ci, state
#define DSEQUENCES gierr, gferr, goerr, cierr, stateerr, outerr
#define WEIGHTS WGI, WGF, WGO, WCI
#define PEEPS WIP, WFP, WOP
#define DWEIGHTS DWGI, DWGF, DWGO, DWCI
#define DPEEPS DWIP, DWFP, DWOP
    Sequence source, SEQUENCES, sourceerr, DSEQUENCES;
    Mat WEIGHTS, DWEIGHTS;
    Vec PEEPS, DPEEPS;
    Float gradient_clipping = 10.0;
    int ni, no, nf;
    int nsteps = 0;
    int nseq = 0;
    string mykind = string("LSTM_")+F::kind+G::kind+H::kind+(PEEP ? "" : "_NP");
    GenericLSTM() {
        name = "lstm";
    }
    const char *kind() {
        return mykind.c_str();
    }
    int noutput() {
        return no;
    }
    int ninput() {
        return ni;
    }
    void postLoad() {
        no = ROWS(WGI);
        nf = COLS(WGI);
        assert(nf > no);
        ni = nf-no-1;
        clearUpdates();
    }
    void initialize() {
        int ni = irequire("ninput");
        int no = irequire("noutput");
        int nf = 1+ni+no;
        this->ni = ni;
        this->no = no;
        this->nf = nf;
        each([no, nf](Mat &w) {randinit(w, no, nf, 0.01);}, WEIGHTS);
        if (PEEP) {
            each([no](Vec &w) {randinit(w, no, 0.01);}, PEEPS);
        };
        clearUpdates();
    }
    void clearUpdates() {
        each([this](Mat &d) { d = Mat::Zero(no, nf); }, DWEIGHTS);
        if (PEEP) each([this](Vec &d) { d = Vec::Zero(no); }, DPEEPS);
    }
    void resize(int N) {
        each([N](Sequence &s) {
                 s.resize(N);
                 for (int t = 0; t < N; t++) s[t].setConstant(NAN);
             }, source, sourceerr, outputs, SEQUENCES, DSEQUENCES);
        assert(source.size() == N);
        assert(gix.size() == N);
        assert(goerr.size() == N);
    }
#define A array()
    void forward() {
        int N = inputs.size();
        resize(N);
        for (int t = 0; t < N; t++) {
            int bs = COLS(inputs[t]);
            source[t].resize(nf, bs);
            BLOCK(source[t], 0, 0, 1, bs).setConstant(1);
            BLOCK(source[t], 1, 0, ni, bs) = inputs[t];
            if (t == 0) BLOCK(source[t], 1+ni, 0, no, bs).setConstant(0);
            else BLOCK(source[t], 1+ni, 0, no, bs) = outputs[t-1];
            gix[t] = MATMUL(WGI, source[t]);
            gfx[t] = MATMUL(WGF, source[t]);
            gox[t] = MATMUL(WGO, source[t]);
            cix[t] = MATMUL(WCI, source[t]);
            if (t > 0) {
                int bs = COLS(state[t-1]);
                for (int b = 0; b < bs; b++) {
                    if (PEEP) COL(gix[t], b) += EMUL(WIP, COL(state[t-1], b));
                    if (PEEP) COL(gfx[t], b) += EMUL(WFP, COL(state[t-1], b));
                }
            }
            gi[t] = nonlin<F>(gix[t]);
            gf[t] = nonlin<F>(gfx[t]);
            ci[t] = nonlin<G>(cix[t]);
            state[t] = ci[t].A * gi[t].A;
            if (t > 0) {
                state[t] += EMUL(gf[t], state[t-1]);
                if (PEEP) {
                    int bs = COLS(state[t]);
                    for (int b = 0; b < bs; b++)
                        COL(gox[t], b) += EMULV(WOP, COL(state[t], b));
                }
            }
            go[t] = nonlin<F>(gox[t]);
            outputs[t] = nonlin<H>(state[t]).A * go[t].A;
        }
    }
    void backward() {
        int N = inputs.size();
        d_inputs.resize(N);
        for (int t = N-1; t >= 0; t--) {
            int bs = COLS(d_outputs[t]);
            outerr[t] = d_outputs[t];
            if (t < N-1) outerr[t] += BLOCK(sourceerr[t+1], 1+ni, 0, no, bs);
            goerr[t] = EMUL(EMUL(yprime<F>(go[t]), nonlin<H>(state[t])), outerr[t]);
            stateerr[t] = EMUL(EMUL(xprime<H>(state[t]),  go[t].A), outerr[t]);
            if (PEEP) {
                for (int b = 0; b < bs; b++)
                    COL(stateerr[t], b) += EMULV(COL(goerr[t], b), WOP);
            }
            if (t < N-1) {
                if (PEEP) for (int b = 0; b < bs; b++) {
                        COL(stateerr[t], b) += EMULV(COL(gferr[t+1], b), WFP);
                        COL(stateerr[t], b) += EMULV(COL(gierr[t+1], b), WIP);
                    }
                stateerr[t] += EMUL(stateerr[t+1], gf[t+1]);
            }
            if (t > 0) gferr[t] = EMUL(EMUL(yprime<F>(gf[t]), stateerr[t]), state[t-1]);
            gierr[t] = EMUL(EMUL(yprime<F>(gi[t]), stateerr[t]), ci[t]);
            cierr[t] = EMUL(EMUL(yprime<G>(ci[t]), stateerr[t]), gi[t]);
            sourceerr[t] = MATMUL_TR(WGI, gierr[t]);
            if (t > 0) sourceerr[t] += MATMUL_TR(WGF, gferr[t]);
            sourceerr[t] += MATMUL_TR(WGO, goerr[t]);
            sourceerr[t] += MATMUL_TR(WCI, cierr[t]);
            d_inputs[t].resize(ni, bs);
            d_inputs[t] = BLOCK(sourceerr[t], 1, 0, ni, bs);
        }
        if (gradient_clipping > 0 || gradient_clipping < 999) {
            gradient_clip(gierr, gradient_clipping);
            gradient_clip(gferr, gradient_clipping);
            gradient_clip(goerr, gradient_clipping);
            gradient_clip(cierr, gradient_clipping);
        }
        for (int t = 0; t < N; t++) {
            int bs = COLS(state[t]);
            if (PEEP) {
                for (int b = 0; b < bs; b++) {
                    if (t > 0) DWIP += EMULV(COL(gierr[t], b), COL(state[t-1], b));
                    if (t > 0) DWFP += EMULV(COL(gferr[t], b), COL(state[t-1], b));
                    DWOP += EMULV(COL(goerr[t], b), COL(state[t], b));
                }
            }
            DWGI += MATMUL_RT(gierr[t], source[t]);
            if (t > 0) DWGF += MATMUL_RT(gferr[t], source[t]);
            DWGO += MATMUL_RT(goerr[t], source[t]);
            DWCI += MATMUL_RT(cierr[t], source[t]);
        }
        nsteps += N;
        nseq += 1;
    }
#undef A
    void gradient_clip(Sequence &s, Float m=1.0) {
        for (int t = 0; t < s.size(); t++) {
            s[t] = MAPFUNC(s[t],
                           [m](Float x) {
                               return x > m ? m : x < -m ? -m : x;
                           });
        }
    }
    void update() {
        float lr = learning_rate;
        if (normalization == NORM_BATCH) lr /= nseq;
        else if (normalization == NORM_LEN) lr /= nsteps;
        else if (normalization == NORM_NONE) /* do nothing */;
        else throw "unknown normalization";
        WGI += lr * DWGI;
        WGF += lr * DWGF;
        WGO += lr * DWGO;
        WCI += lr * DWCI;
        if (PEEP) {
            WIP += lr * DWIP;
            WFP += lr * DWFP;
            WOP += lr * DWOP;
        }
        DWGI *= momentum;
        DWGF *= momentum;
        DWGO *= momentum;
        DWCI *= momentum;
        if (PEEP) {
            DWIP *= momentum;
            DWFP *= momentum;
            DWOP *= momentum;
        }
    }
    void myweights(const string &prefix, WeightFun f) {
        f(prefix+".WGI", &WGI, &DWGI);
        f(prefix+".WGF", &WGF, &DWGF);
        f(prefix+".WGO", &WGO, &DWGO);
        f(prefix+".WCI", &WCI, &DWCI);
        if (PEEP) {
            f(prefix+".WIP", &WIP, &DWIP);
            f(prefix+".WFP", &WFP, &DWFP);
            f(prefix+".WOP", &WOP, &DWOP);
        }
    }
    virtual void mystates(const string &prefix, StateFun f) {
        f(prefix+".inputs", &inputs);
        f(prefix+".d_inputs", &d_inputs);
        f(prefix+".outputs", &outputs);
        f(prefix+".d_outputs", &d_outputs);
        f(prefix+".state", &state);
        f(prefix+".stateerr", &stateerr);
        f(prefix+".gi", &gi);
        f(prefix+".gierr", &gierr);
        f(prefix+".go", &go);
        f(prefix+".goerr", &goerr);
        f(prefix+".gf", &gf);
        f(prefix+".gferr", &gferr);
        f(prefix+".ci", &ci);
        f(prefix+".cierr", &cierr);
    }
    Sequence *getState() {
        return &state;
    }
};

typedef GenericLSTM<SigmoidNonlin, TanhNonlin, TanhNonlin> LSTM;
REGISTER(LSTM);
typedef GenericLSTM<SigmoidNonlin, TanhNonlin, NoNonlin> LINLSTM;
REGISTER(LINLSTM);
typedef GenericLSTM<SigmoidNonlin, ReluNonlin, TanhNonlin> RELUTANHLSTM;
REGISTER(RELUTANHLSTM);
typedef GenericLSTM<SigmoidNonlin, ReluNonlin, NoNonlin> RELULSTM;
REGISTER(RELULSTM);
typedef GenericLSTM<SigmoidNonlin, ReluNonlin, ReluNonlin> RELU2LSTM;
REGISTER(RELU2LSTM);

}
