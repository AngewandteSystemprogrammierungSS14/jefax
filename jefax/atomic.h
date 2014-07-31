/* The atomic component provides integers with atomic access.
 * The function enterAtomicBlock() disables interrupts and returns
 * true if the interrupt flag was set before disabling them, else false.
 * The return value is used by exitAtomicBlock() to decide if the flag
 * should be set again or not. */ 

#pragma once

#include <stdint.h>

typedef int atomic_t;

uint8_t enterAtomicBlock();
void exitAtomicBlock(const uint8_t p_state);

void atomicSet(atomic_t *p_atomic, const int p_value);
int atomicRead(atomic_t *p_atomic);
void atomicInc(atomic_t *p_atomic);
void atomicDec(atomic_t *p_atomic);
void atomicAdd(atomic_t *p_atomic, const int p_value);
void atomicSub(atomic_t *p_atomic, const int p_value);
int atomicTestSet(atomic_t *p_atomic);