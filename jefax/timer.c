/*
 * timer.c
 *
 * Created: 19.05.2014 15:48:31
 *  Author: Fabian
 */ 

#include "timer.h"
#include "task.h"
#include "atomic.h"
#include <avr/interrupt.h>

#define DEF_TIMER_COUNT 20
#define MS_TO_TIMER_PER(ms) ((int) (((unsigned long) ms) * 32000000000UL / 256UL))
#define TIMER_PER_TO_MS(per) ((int) (((unsigned long) per) * 256UL / 32000000000UL))
#define TIMER_PRESCALER TC_CLKSEL_DIV256_gc
#define DISABLE_TIMER() TCC0.CTRLA = TC_CLKSEL_OFF_gc
#define ENABLE_TIMER() TCC0.CTRLA = TIMER_PRESCALER

static timer_t timers[DEF_TIMER_COUNT];
static int timerCount;

static void updatePeriod();
static void timerElapsed(const int p_index);

int initTimerSystem()
{
	// Set 16 bit timer
	TCD0.CTRLA = TC_CLKSEL_OFF_gc; // timer off
	TCD0.CTRLB = 0x00; // select Modus: Normal -> Event/Interrupt at top
	TCD0.PER = MS_TO_TIMER_PER(100);
	TCD0.CNT = 0x00;
	TCD0.INTCTRLA = TC_OVFINTLVL_LO_gc; // Enable overflow interrupt level low
	
	return 0;
}

int initTimer(timer_t *p_timer, int p_ms, void (*p_callBack) (void*), void * p_arg)
{
	p_timer->ms = p_ms;
	p_timer->callBack = p_callBack;
	p_timer->arg = p_arg;
	
	return 0;
}

int addTimer(timer_t p_timer)
{
	uint8_t irEnabled = enterAtomicBlock();
	
	if(timerCount >= DEF_TIMER_COUNT)
		return -1;
		
	timers[timerCount] = p_timer;
	++timerCount;
	updatePeriod();
	
	if(timerCount == 1)
		ENABLE_TIMER();
	
	int result = timerCount - 1;
	
	exitAtomicBlock(irEnabled);
	
	return result;
}

static void updatePeriod()
{
	int nextMS = 100;
	int i;
	for(i = 0; i < timerCount; ++i)
	{
		if(timers[i].ms < nextMS)
			nextMS = timers[i].ms;
	}
	
	TCD0.CNT = 0;
	TCD0.PER = MS_TO_TIMER_PER(nextMS);
}

ISR(TCD0_OVF_vect)
{
	int ms = TIMER_PER_TO_MS(TCD0.PER);
	int i;
	
	for(i = 0; i < timerCount; ++i)
		timers[i].ms -= ms;
	
	for(i = 0; i < timerCount; ++i)
	{
		while(i < timerCount && timers[i].ms <= 0)
			timerElapsed(i);
	}
}

static void timerElapsed(const int p_index)
{
	timers[p_index].callBack(timers[p_index].arg);
	int i;
	for(i = p_index; i < timerCount - 1; ++i)
		timers[i] = timers[i + 1];
	--timerCount;
	
	if(timerCount == 0)
		DISABLE_TIMER();
}