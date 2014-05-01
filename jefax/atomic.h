/*
 * atomic.h
 *
 * Created: 01.05.2014 11:55:16
 *  Author: Fabian
 */ 

#pragma once

typedef int atomic_t;

void atomicSet(atomic_t *p_atomic, const int p_value);
int atomicRead(atomic_t *p_atomic);
void atomicInc(atomic_t *p_atomic);
void atomicDec(atomic_t *p_atomic);
void atomicAdd(atomic_t *p_atomic, const int p_value);
void atomicSub(atomic_t *p_atomic, const int p_value);
int atomicTestSet(atomic_t *p_atomic);