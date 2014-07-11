#include <avr/interrupt.h>
#include "atomic.h"

uint8_t enterAtomicBlock()
{
	// check if interrupt flag is set
	uint8_t result = SREG & 0x80;
	cli();
	return result;
}

void exitAtomicBlock(const uint8_t p_interruptsEnabled)
{
	// only set interrupt flag if it was set before entering the atomic block
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