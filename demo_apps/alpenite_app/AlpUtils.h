#pragma once

// Defer using RAII to call a lambda on scope exit
namespace Alp {
struct __defer {};
// Obj
template <class F>
struct __Deferrer {
    F f;
    constexpr ~__Deferrer() {
        f();
    }
};

template <class F>
__Deferrer<F> constexpr operator+(__defer, F f) {
    return {f};
}
}  // namespace Alp
#define ALP_DEFER_1(x, y) x##y
#define ALP_DEFER_2(x, y) ALP_DEFER_1(x, y)
#define ALP_DEFER_3(x)    ALP_DEFER_2(x, __COUNTER__)
#define defer [[maybe_unused]] auto ALP_DEFER_3(_defer_) = Alp::__defer{} + [&](void) -> void

// Takes a const array and returns the number of elements
template <typename T, int N>
inline constexpr int countOf(const T (&)[N]) noexcept {
    return N;
}

