#include "Halide.h"

#if defined(_MSC_VER) && !defined(NOMINMAX)
#define NOMINMAX
#endif
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

using namespace Halide;

int main(int argc, char **argv) {
    std::string lib;
#ifdef _WIN32
    if (argc > 1) {
        lib = argv[1];
    } else {
        lib = "gradient_autoscheduler.dll";
    }

    if (!LoadLibraryA(lib.c_str())) {
        DWORD last_err = GetLastError();
        LPVOID last_err_msg;
        FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
                           FORMAT_MESSAGE_IGNORE_INSERTS,
                       nullptr, last_err, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                       reinterpret_cast<LPSTR>(&last_err_msg), 0, nullptr);
        std::cerr << "Failed to load: " << lib << "\n";
        std::cerr << "LoadLibraryA failed with error " << last_err << ": "
                  << static_cast<char *>(last_err_msg) << "\n";
        LocalFree(last_err_msg);
        return 1;
    }
#else
    if (argc > 1) {
        lib = argv[1];
    } else {
        lib = "libgradient_autoscheduler.so";
    }

    if (dlopen(lib.c_str(), RTLD_LAZY) == nullptr) {
        std::cerr << "Failed to load: " << lib << ": " << dlerror() << "\n";
        return 1;
    }
#endif

    MachineParams params(32, 16000000, 40);
    Target target;

    Var x("x"), y("y");

    { // Simple 1D pointwise operations. Should inline.
        Func in("in");
        in(x) = cast<float>(x);
        Func f0("f0");
        f0(x) = 2.f * in(x);
        Func f1("f1");
        f1(x) = sin(f0(x));
        Func f2("f2");
        f2(x) = f1(x) * f1(x);

        f2.set_estimate(x, 0, 10000);

        AutoSchedulerResults result =
            Pipeline(f2).auto_schedule(target, params);
        std::cout << "Schedule for 1D pointwise operations:" << std::endl;
        std::cout << result.schedule_source << std::endl;
        std::cout << std::endl;
    }

    { // Simple 2D pointwise operations. Should inline.
        Func in("in");
        in(x, y) = cast<float>(x + y);
        Func f0("f0");
        f0(x, y) = 2.f * in(x, y);
        Func f1("f1");
        f1(x, y) = sin(f0(x, y));
        Func f2("f2");
        f2(x, y) = f1(x, y) * f1(x, y);

        f2.set_estimate(x, 0, 1000)
          .set_estimate(y, 0, 1000);

        AutoSchedulerResults result =
            Pipeline(f2).auto_schedule(target, params);
        std::cout << "Schedule for 2D pointwise operations:" << std::endl;
        std::cout << result.schedule_source << std::endl;
        std::cout << std::endl;
    }

    { // 1D Convolution.
        Func in("in");
        in(x) = cast<float>(x);
        RDom r(0, 5);
        Func f0("f0");
        f0(x) += in(x + r) / 5.f;

        f0.set_estimate(x, 0, 1000);

        AutoSchedulerResults result =
            Pipeline(f0).auto_schedule(target, params);
        std::cout << "Schedule for 1D convolution:" << std::endl;
        std::cout << result.schedule_source << std::endl;
        std::cout << std::endl;
    }

    { // 2D Convolution.
        Func in("in");
        in(x, y) = cast<float>(x + y);
        RDom r(0, 5, 0, 5);
        Func f0("f0");
        f0(x, y) += in(x + r.x, y + r.y) / 25.f;

        f0.set_estimate(x, 0, 1000)
          .set_estimate(y, 0, 1000);

        AutoSchedulerResults result =
            Pipeline(f0).auto_schedule(target, params);
        std::cout << "Schedule for 2D convolution:" << std::endl;
        std::cout << result.schedule_source << std::endl;
        std::cout << std::endl;
    }

    { // 1D Histogram.
        Func in("in");
        in(x) = x % 10;
        RDom r(0, 1000);
        Func hist("hist");
        hist(x) = 0;
        hist(clamp(in(r), 0, 10)) += 1;

        hist.set_estimate(x, 0, 10);

        AutoSchedulerResults result =
            Pipeline(hist).auto_schedule(target, params);
        std::cout << "Schedule for 1D histogram:" << std::endl;
        std::cout << result.schedule_source << std::endl;
        std::cout << std::endl;
    }

    { // 2D Histogram.
        Func in("in");
        in(x, y) = (x + y) % 10;
        RDom r(0, 1000, 0, 1000);
        Func hist("hist");
        hist(x) = 0;
        hist(clamp(in(r.x, r.y), 0, 10)) += 1;

        hist.set_estimate(x, 0, 10);

        AutoSchedulerResults result =
            Pipeline(hist).auto_schedule(target, params);
        std::cout << "Schedule for 2D histogram:" << std::endl;
        std::cout << result.schedule_source << std::endl;
        std::cout << std::endl;
    }

    { // 2D Histogram, but the domain is much larger.
        Func in("in");
        in(x, y) = (x + y) % 10000;
        RDom r(0, 1000, 0, 1000);
        Func hist("hist");
        hist(x) = 0;
        hist(clamp(in(r.x, r.y), 0, 10000)) += 1;

        hist.set_estimate(x, 0, 10000);

        AutoSchedulerResults result =
            Pipeline(hist).auto_schedule(target, params);
        std::cout << "Schedule for 2D histogram with larger domain:" << std::endl;
        std::cout << result.schedule_source << std::endl;
        std::cout << std::endl;
    }

    return 0;
}
