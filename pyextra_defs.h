#include <Python.h>

/** This extra header is needed to make the code Python 3 compatible.
 * The reason for this is that the signature of `PyUnicode_AsWideChar` changed
 * due to the removal of the `PyUnicodeObject` (`unicode`) type in Python 3.
 *
 * Unfortunately Cython doesn't support conditional compilation out of the
 * box, so we need to use this workaround.
 **/
#if PY_MAJOR_VERSION >= 3
typedef PyObject UnicodeObject;
#else
typedef PyUnicodeObject UnicodeObject;
#endif

Py_ssize_t Unicode_AsWideChar(PyObject* str, Py_ssize_t length, wchar_t *wchars) {
    return PyUnicode_AsWideChar((UnicodeObject*) str, wchars, length);
}
