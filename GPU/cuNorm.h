#ifndef CUNORM_H
#define CUNORM_H
#include "head.h"
#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>
#include <time.h>
#include <cuda_runtime.h>

void cuNorm(int l);
void cuNorm2(int l);
#endif
