#ifndef __TIMER_H__
#define __TIMER_H__

#ifdef __cplusplus
#define EXTERN_C extern "C"
#endif // __cplusplus

typedef int error_t;

const int success = 0;
const int error_invalid_argument = 1;
const int error_invalid_memory = 2;
const int error_not_implemented = 3;

struct profile_entry;
struct profile;

#endif // __TIMER_H__