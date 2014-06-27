#include <limits.h>
#include <avr/interrupt.h>
#include "timer.h"
#include "task.h"
#include "atomic.h"
#include "utils.h"
#include "scheduler.h"

#define DEF_TIMER_COUNT 10
#define TIMER_PRESCALER TC_CLKSEL_DIV256_gc

static volatile timer_t timers[DEF_TIMER_COUNT];
static volatile int timerCount;

static void updatePeriod();
static void timerElapsed(const int p_index);
static void decreaseTimers();

int initTimerSystem()
{
	// Set 16 bit timer
	TCD0.CTRLA = TC_CLKSEL_OFF_gc; // timer off
	TCD0.CTRLB = 0x00; // select Modus: Normal -> Event/Interrupt at top
	TCD0.PER = MS_TO_TIMER(100, TIMER_PRESCALER);
	TCD0.CNT = 0x00;
	TCD0.INTCTRLA = TC_OVFINTLVL_LO_gc; // Enable overflow interrupt level low
	
	return 0;
}

int initTimer(timer_t *p_timer, unsigned int p_ms, void (*p_callBack) (void*), void * p_arg)
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
	
	// enable hardware timer if there are timers in the list
	if(timerCount >= 1)
		ENABLE_TIMER(TCD0, TIMER_PRESCALER);
	
	int result = timerCount - 1;
	
	exitAtomicBlock(irEnabled);
	
	return result;
}

static void updatePeriod()
{
	unsigned int nextMS = UINT_MAX;
	int i;
	// find shortest relative value
	for(i = 0; i < timerCount; ++i) {
		if(timers[i].ms < nextMS)
			nextMS = timers[i].ms;
	}
	
	// reset hardware timer
	TCD0.CNT = 0;
	TCD0.PER = MS_TO_TIMER(nextMS, TIMER_PRESCALER);
}

ISR(TCD0_OVF_vect,ISR_NAKED)
{
	SAVE_CONTEXT();
	getRunningTask()->stackpointer = (uint8_t *) SP;
	
	ENTER_SYSTEM_STACK();
	
	decreaseTimers();
	
	SP = (uint16_t) (getRunningTask()->stackpointer);
	RESTORE_CONTEXT();
	reti();
}

static void decreaseTimers(const int p_ms)
{
	// get elapsed time
	unsigned int ms = TIMER_TO_MS(TCD0.PER, TIMER_PRESCALER);
	unsigned int toDec;
	int i;
	
	// decrease timer values
	for(i = 0; i < timerCount; ++i) {
		// prevent timer[i].ms from getting lower than 0
		toDec = (timers[i].ms >= ms ? ms : timers[i].ms);
		timers[i].ms -= toDec;
	}
	
	// check for all timers if they elapsed
	for(i = 0; i < timerCount; ++i) {
		while(i < timerCount && timers[i].ms <= 0)
			timerElapsed(i);
	}
	if(timerCount > 0)
		updatePeriod();
}

static void timerElapsed(const int p_index)
{
	timers[p_index].callBack(timers[p_index].arg);
	int i;
	for(i = p_index; i < timerCount - 1; ++i)
		timers[i] = timers[i + 1];
	--timerCount;
	
	if(timerCount == 0)
		DISABLE_TIMER(TCD0);
}