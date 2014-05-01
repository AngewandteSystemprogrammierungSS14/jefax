/*
 * atomic.c
 *
 * Created: 01.05.2014 11:55:28
 *  Author: Fabian
 */ 

#include <util/atomic.h>
#include "atomic.h"

void atomicSet(atomic_t *p_atomic, const int p_value)
{
	ATOMIC_BLOCK(ATOMIC_RESTORESTATE)
	{
		*(p_atomic) = p_value;
	}
}

int atomicRead(atomic_t *p_atomic)
{
	int result;
	ATOMIC_BLOCK(ATOMIC_RESTORESTATE)
	{
		result = *(p_atomic);
	}
	
	return result;
}

void atomicInc(atomic_t *p_atomic)
{
	ATOMIC_BLOCK(ATOMIC_RESTORESTATE)
	{
		*(p_atomic) += 1;
	}
}

void atomicDec(atomic_t *p_atomic)
{
	ATOMIC_BLOCK(ATOMIC_RESTORESTATE)
	{
		*(p_atomic) -= 1;
	}
}

void atomicAdd(atomic_t *p_atomic, const int p_value)
{
	ATOMIC_BLOCK(ATOMIC_RESTORESTATE)
	{
		*(p_atomic) += p_value;
	}
}

void atomicSub(atomic_t *p_atomic, const int p_value)
{
	ATOMIC_BLOCK(ATOMIC_RESTORESTATE)
	{
		*(p_atomic) -= p_value;
	}
}

int atomicTestSet(atomic_t *p_atomic)
{
	int result;
	ATOMIC_BLOCK(ATOMIC_RESTORESTATE)
	{
		result = *(p_atomic);
		*(p_atomic) = 1;
	}
	
	return result;
}