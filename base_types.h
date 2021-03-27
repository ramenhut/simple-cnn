/*
//
// Copyright (c) 1998-2016 Joe Bertolami. All Right Reserved.
//
//   Redistribution and use in source and binary forms, with or without
//   modification, are permitted provided that the following conditions are met:
//
//   * Redistributions of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistributions in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
//   AND ANY EXPRESS OR IMPLIED WARRANTIES, CLUDG, BUT NOT LIMITED TO, THE
//   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
//   ARE DISCLAIMED.  NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
//   LIABLE FOR ANY DIRECT, DIRECT, CIDENTAL, SPECIAL, EXEMPLARY, OR
//   CONSEQUENTIAL DAMAGES (CLUDG, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
//   GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSESS TERRUPTION)
//   HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER  CONTRACT, STRICT
//   LIABILITY, OR TORT (CLUDG NEGLIGENCE OR OTHERWISE) ARISG  ANY WAY  OF THE
//   USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Additional Information:
//
//   For more information, visit http://www.bertolami.com.
//
*/

#ifndef __BASE_TYPES_H__
#define __BASE_TYPES_H__

#include <assert.h>

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#if defined(WIN32) || defined(_WIN64)
#define BASE_PLATFORM_WINDOWS
#include "windows.h"
#elif defined(__APPLE__)
#define BASE_PLATFORM_MACOS
#include "TargetConditionals.h"
#include "ctype.h"
#include "sys/types.h"
#include "unistd.h"
#else
#error "Unsupported target platform detected."
#endif

#define BASE_PI (3.14159262f)
#define BASE_2PI (6.28318524f)
#define BASE_INFINITY (1.0e15f)
#define BASE_EPSILON (1.0e-5f)
#define BASE_LOG2 (0.3010299956639f)

#define BASE_MAX_INT64 (0x7FFFFFFFFFFFFFFF)
#define BASE_MAX_INT32 (0x7FFFFFFF)
#define BASE_MAX_INT16 (0x7FFF)
#define BASE_MAX_INT8 (0x7F)

#define BASE_MAX_UINT64 (0xFFFFFFFFFFFFFFFF)
#define BASE_MAX_UINT32 (0xFFFFFFFF)
#define BASE_MAX_UINT16 (0xFFFF)
#define BASE_MAX_UINT8 (0xFF)

#define BASE_MIN_INT64 (-BASE_MAX_INT64 - 1)
#define BASE_MIN_INT32 (-BASE_MAX_INT32 - 1)
#define BASE_MIN_INT16 (-BASE_MAX_INT16 - 1)
#define BASE_MIN_INT8 (-BASE_MAX_INT8 - 1)

namespace base {

typedef int64_t int64;
typedef int32_t int32;
typedef int16_t int16;
typedef int8_t int8;
typedef uint64_t uint64;
typedef uint32_t uint32;
typedef uint16_t uint16;
typedef uint8_t uint8;
typedef float float32;
typedef double float64;

}  // namespace base

#endif  // __BASE_TYPES_H__
