// A test harness small enough to read in one sitting. No dependency, because this runtime is
// meant to build from source on a pinned box with nothing but a compiler.
#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace burnisher_test {

inline int& failures() { static int f = 0; return f; }
inline int& checks() { static int c = 0; return c; }

inline void report(bool ok, const char* expr, const char* file, int line,
                   const std::string& extra = "") {
    ++checks();
    if (ok) return;
    ++failures();
    std::fprintf(stderr, "FAIL %s:%d  %s%s%s\n", file, line, expr,
                 extra.empty() ? "" : "  --  ", extra.c_str());
}

inline int summary(const char* name) {
    std::fprintf(stderr, "%s: %d checks, %d failures\n", name, checks(), failures());
    return failures() ? 1 : 0;
}

}  // namespace burnisher_test

#define CHECK(expr) ::burnisher_test::report((expr), #expr, __FILE__, __LINE__)
#define CHECK_MSG(expr, msg) ::burnisher_test::report((expr), #expr, __FILE__, __LINE__, (msg))
#define CHECK_NEAR(a, b, tol)                                                              \
    ::burnisher_test::report(std::fabs((a) - (b)) <= (tol), #a " ~= " #b, __FILE__,         \
                             __LINE__, "got " + std::to_string((double)(a)) + " vs " +      \
                                       std::to_string((double)(b)))
#define CHECK_THROWS(stmt)                                                                 \
    do {                                                                                   \
        bool threw = false;                                                                \
        try { stmt; } catch (const std::exception&) { threw = true; }                       \
        ::burnisher_test::report(threw, "throws: " #stmt, __FILE__, __LINE__);              \
    } while (0)
