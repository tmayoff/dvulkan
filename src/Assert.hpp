#pragma once
#ifndef ASSERT_HPP_
#define ASSERT_HPP_

#ifdef ENABLE_ASSERTS

#define INTERNAL_ASSERT_IMPL(type, check, msg, ...)                            \
  {                                                                            \
    if (!(check)) {                                                            \
      DV##type##ERROR(msg, __VA_ARGS__);                                       \
      DEBUGBREAK;                                                              \
    }                                                                          \
  }

#define INTERNAL_ASSERT_WITH_MSG(type, check, ...)                             \
  INTERNAL_ASSERT_IMPL(type, check, "Assertion failed: {0}", __VA_ARGS__)

#define INTERNAL_ASSERT_NO_MSG(type, check)                                    \
  INTERNAL_ASSERT_IMPL(type, check, "Assertion '{0}', failed at {1}:{2}",      \
                       STRINGIFY_MACRO(check),                                 \
                       std::filesystem::path(__FILE__).filename().string(),    \
                       __LINE__)

#define INTERNAL_ASSERT_GET_MACRO_NAME(arg1, arg2, macro, ...) macro
#define INTERNAL_ASSERT_GET_MACRO(...)                                         \
  EXPAND_MACRO(INTERNAL_ASSERT_GET_MACRO_NAME(                                 \
      __VA_ARGS__, INTERNAL_ASSERT_WITH_MSG, INTERNAL_ASSERT_NO_MSG))

#define ASSERT(...)                                                            \
  EXPAND_MACRO(INTERNAL_ASSERT_GET_MACRO(__VA_ARGS__)(_, __VA_ARGS__))
#define CORE_ASSERT(...)                                                       \
  EXPAND_MACRO(INTERNAL_ASSERT_GET_MACRO(__VA_ARGS__)(_CORE_, __VA_ARGS__))
#else
#define ASSERT(...)
#define CORE_ASSERT(...)
#endif

#endif // ASSERT_HPP_
