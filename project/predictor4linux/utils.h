#ifndef __UTIL_H__
#define __UTIL_H__

#include <stdint.h>
#include <stddef.h>
#include <sys/time.h>

// int64_t GetCurrentTime();
// double GetElapsedTime(int64_t time);


inline int64_t GetCurrentTime() {
    struct timeval time;
    gettimeofday(&time, NULL);
    return 1000000LL * (int64_t)time.tv_sec + (int64_t)time.tv_usec;
}

inline double GetElapsedTime(int64_t time) {
    return (GetCurrentTime() - time) / 1000.0f;
}

#endif // __UTIL_H__

