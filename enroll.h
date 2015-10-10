#ifndef enroll_h__
#define enroll_h__
#define VA_NUM_ARGS(...) \
  VA_NUM_ARGS_IMPL(__VA_ARGS__, 9, 8, 7, 6, 5, 4, 3, 2, 1)
#define VA_NUM_ARGS_IMPL(_1, _2, _3, _4, _5, _6, _7, _8, _9, N, ...) N
#define ENROLL(...) ENROLL_(VA_NUM_ARGS(__VA_ARGS__), __VA_ARGS__)
#define ENROLL_(count, ...) ENROLL0(count, __VA_ARGS__)
#define ENROLL0(count, ...) ENROLL##count(__VA_ARGS__)
#define ENROLL1(a) enroll(a, #a)
#define ENROLL2(a, ...) \
  enroll(a, #a);        \
  ENROLL1(__VA_ARGS__)
#define ENROLL3(a, ...) \
  enroll(a, #a);        \
  ENROLL2(__VA_ARGS__)
#define ENROLL4(a, ...) \
  enroll(a, #a);        \
  ENROLL3(__VA_ARGS__)
#define ENROLL5(a, ...) \
  enroll(a, #a);        \
  ENROLL4(__VA_ARGS__)
#define ENROLL6(a, ...) \
  enroll(a, #a);        \
  ENROLL5(__VA_ARGS__)
#define ENROLL7(a, ...) \
  enroll(a, #a);        \
  ENROLL6(__VA_ARGS__)
#define ENROLL8(a, ...) \
  enroll(a, #a);        \
  ENROLL7(__VA_ARGS__)
#define ENROLL9(a, ...) \
  enroll(a, #a);        \
  ENROLL8(__VA_ARGS__)
#endif
