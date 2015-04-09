
inline const char *hg_version() {
    return "e8845c8e6e6f+";
}
inline const char *hg_log() {
    return R"(changeset:   73:e8845c8e6e6f
tag:         tip
user:        Tom <tmb@google>
date:        Sun Apr 05 16:58:26 2015 -0700
files:       clstm.cc clstm.h clstmctc.cc clstmtext.cc
description:
bug fixes


changeset:   72:8e878ec34ccc
user:        Tom <tmb@google>
date:        Thu Apr 02 21:06:03 2015 -0700
files:       extras.cc extras.h
description:
bug fix in random number generator


changeset:   71:a8586bf81db6
user:        Tom <tmb@google>
date:        Thu Apr 02 19:39:04 2015 -0700
files:       clstmctc.cc clstmtext.cc test-batch.cc
description:
replaced more rand48 functions with builtins


)";
}
