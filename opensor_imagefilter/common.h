#ifndef COMMON_H
#define COMMON_H

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <memory.h>
#include <math.h>

// Align up n to the nearest multiple of 

// round up n/m
inline int iDivUp(int n, int m)
{
	return (n + m - 1) / m;
}
#endif
