// -*- C++ -*-

#ifndef ocropus_clstm_pstring_
#define ocropus_clstm_pstring_

#include <assert.h>
#include <wchar.h>
#include <iostream>
#include <string>

namespace {
std::wstring utf8_to_utf32(std::string s) {
  int i = 0;
  std::wstring result;
  while (i < s.size()) {
    unsigned c = unsigned(s[i]);
    unsigned w = '~';
    if ((c & 0x80) == 0) {
      w = c;
      i += 1;
    } else if ((c & 0xe0) == 0xc0) {
      if (i + 1 >= s.size()) THROW("bad encoding");
      unsigned c1 = unsigned(s[i + 1]);
      w = (((c & 0x1f) << 6) | (c1 & 0x3f));
      i += 2;
    } else if ((c & 0xf0) == 0xe0) {
      if (i + 2 >= s.size()) THROW("bad encoding");
      unsigned c1 = unsigned(s[i + 1]);
      unsigned c2 = unsigned(s[i + 2]);
      w = (((c & 0x0f) << 12) | ((c1 & 0x3f) << 6) | (c2 & 0x3f));
      i += 3;
    } else if ((c & 0xf8) == 0xf0) {
      if (i + 3 >= s.size()) THROW("bad encoding");
      unsigned c1 = unsigned(s[i + 1]);
      unsigned c2 = unsigned(s[i + 2]);
      unsigned c3 = unsigned(s[i + 3]);
      w = (((c & 0x0f) << 18) | ((c1 & 0x3f) << 12) | ((c2 & 0x3f) << 6) |
           (c3 & 0x3f));
      i += 4;
    } else {
      THROW("unicode character out of range");
    }
    assert(w != 0);
    result.push_back(wchar_t(w));
  }
  return result;
}

std::string utf32_to_utf8(std::wstring s) {
  std::string result;
  for (int i = 0; i < s.size(); i++) {
    unsigned c = s[i];
    if (c < 0x80) {
      result.push_back(char(c));
    } else if (c <= 0x7ff) {
      result.push_back(char((c >> 6) | 0xc0));
      result.push_back(char((c & 0x3f) | 0x80));
    } else if (c <= 0xffff) {
      result.push_back(char((c >> 12) | 0xe0));
      result.push_back(char(((c >> 6) & 0x3f) | 0x80));
      result.push_back(char((c & 0x3f) | 0x80));
    } else if (c <= 0x10ffff) {
      result.push_back(char((c >> 18) | 0xf0));
      result.push_back(char(((c >> 12) & 0x3f) | 0x80));
      result.push_back(char(((c >> 6) & 0x3f) | 0x80));
      result.push_back(char((c & 0x3f) | 0x80));
    } else {
      THROW("unicode character out of range");
    }
  }
  return result;
}
}

#endif
