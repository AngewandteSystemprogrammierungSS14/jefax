/*
 * atomic.c
 *
 * Created: 01.05.2014 11:55:28
 *  Author: Fabian
 */ 

#include "atomic.h"
#include <avr/interrupt.h>

uint8_t enterAtomicBlock()
{
	uint8_t result = SREG & 0x80;
	cli();
	return result;
}

void exitAtomicBlock(const uint8_t p_interruptsEnabled)
{
	if(p_interruptsEnabled)
		sei();
	else
		cli();
}

void atomicSet(atomic_t *p_atomic, const int p_value)
{
	uint8_t irEnabled = enterAtomicBlock();
	
	*(p_atomic) = p_value;
	
	exitAtomicBlock(irEnabled);
}

int atomicRead(atomic_t *p_atomic)
{
	int result;
	uint8_t irEnabled = enterAtomicBlock();
	
	result = *(p_atomic);
	
	exitAtomicBlock(irEnabled);
	
	return result;
}

void atomicInc(atomic_t *p_atomic)
{
	uint8_t irEnabled = enterAtomicBlock();
	
	*(p_atomic) += 1;
	
	exitAtomicBlock(irEnabled);
}

void atomicDec(atomic_t *p_atomic)
{
	uint8_t irEnabled = enterAtomicBlock();
	
	*(p_atomic) -= 1;
	
	exitAtomicBlock(irEnabled);
}

void atomicAdd(atomic_t *p_atomic, const int p_value)
{
	uint8_t irEnabled = enterAtomicBlock();
	
	*(p_atomic) += p_value;
	
	exitAtomicBlock(irEnabled);
}

void atomicSub(atomic_t *p_atomic, const int p_value)
{
	uint8_t irEnabled = enterAtomicBlock();
	
	*(p_atomic) -= p_value;
	
	exitAtomicBlock(irEnabled);
}

int atomicTestSet(atomic_t *p_atomic)
{
	int result;
	uint8_t irEnabled = enterAtomicBlock();
	
	result = *(p_atomic);
	*(p_atomic) = 1;
	
	exitAtomicBlock(irEnabled);
	
	return result;
}