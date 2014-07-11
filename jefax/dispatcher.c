#include "dispatcher.h"
#include "scheduler.h"
#include "schedulerRR.h"
#include "atomic.h"
#include "utils.h"
#include "jefax_xmega128.h"
#include "usart.h"
#include <avr/interrupt.h>

#define TIMER_PRESCALER TC_CLKSEL_DIV256_gc

/* The global, extern defined task list with all tasks to dispatch. */
extern task_t TASKS[];

// Prototypes
static void initTimeSliceTimer();
static void init32MHzClock();
static void init32MHzClock2();

void initDispatcher()
{
	initScheduler(getRRScheduler());
	
	init32MHzClock();
	initUsart();
	initTimeSliceTimer();
	
	// Save the main context
	SAVE_CONTEXT();
	main_stackpointer = (uint8_t *) SP;
	
	// Switch to dispatcher task
	/*initTask(&dispatcherTask);
	SP = (uint16_t) (dispatcherTask.stackpointer);*/
	
	//DISABLE_TIMER(TCC0);
	//RESTORE_CONTEXT();
	
	SP = (uint16_t) (getRunningTask()->stackpointer);
	enableInterrupts();
	RESTORE_CONTEXT();
	return;
}

static void init32MHzClock2()
{
	OSC_CTRL |= OSC_RC32MEN_bm;
	while(!(OSC_STATUS & OSC_RC32MRDY_bm))
		;
	CCP = 0xD8;
	CLK_CTRL = (1 << CLK_SCLKSEL_gp);
}

static void init32MHzClock()
{
	uint8_t clkCtrl;
	
	// Enable 32Mhz internal clock source
	OSC.CTRL |= OSC_RC32MEN_bm;
	
	// Wait until clock source is stable
	while (!(OSC.STATUS & OSC_RC32MRDY_bm));
	
	// Select main clock source
	clkCtrl = (CLK.CTRL & ~CLK_SCLKSEL_gm) | CLK_SCLKSEL_RC32M_gc;
	
	asm volatile(
	"movw r30,  %0"	      "\n\t"
	"ldi  r16,  %2"	      "\n\t"
	"out   %3, r16"	      "\n\t"
	"st     Z,  %1"       "\n\t"
	:
	: "r" (&CLK.CTRL), "r" (clkCtrl), "M" (CCP_IOREG_gc), "i" (&CCP)
	: "r16", "r30", "r31"
	);
	
	// Disable 2Mhz default clock source
	//OSC.CTRL &= ~OSC_RC2MEN_bm;
}

/* Initializes timer for time slices.*/
static void initTimeSliceTimer()
{
	// Set 16 bit timer
	TCC0.CTRLA = TIMER_PRESCALER; // 256 prescaler -> 3900 / sec -> 65536 max.
	TCC0.CTRLB = 0x00; // select Modus: Normal -> Event/Interrupt at top
	TCC0.PER = 40;//MS_TO_TIMER(100, TIMER_PRESCALER);
	TCC0.CNT = 0x00;
	TCC0.INTCTRLA = TC_OVFINTLVL_LO_gc; // Enable overflow interrupt level low
}

void setInterruptTime(unsigned int p_msec)
{
	uint8_t irEnabled = enterAtomicBlock();
	TCC0.PER = MS_TO_TIMER(p_msec, TIMER_PRESCALER); // Top-Value (period)
	exitAtomicBlock(irEnabled);
}

/* Function for dispatcher task. */
/*static int runDispatcher()
{
	
	dispatch(toDispatch);
	return 0;
}

/* Change to the given task. */
/*static void dispatch(task_t *p_task)
{
	SP = (uint16_t) (p_task->stackpointer);
	
	ENABLE_TIMER(TCC0, TIMER_PRESCALER);
	RESTORE_CONTEXT();
	reti();
}*/

ISR(TCC0_OVF_vect, ISR_NAKED)
{
	SAVE_CONTEXT();
	getRunningTask()->stackpointer = (uint8_t *) SP;
	
	// set stackpointer to default task
	ENTER_SYSTEM_STACK();
	
	//!! call scheduler
	task_t* toDispatch = schedule();
	SP = (uint16_t) (toDispatch->stackpointer);
	//initTask(&dispatcherTask);
	//SP = (uint16_t) (dispatcherTask.stackpointer);
	
	//DISABLE_TIMER(TCC0);
	RESTORE_CONTEXT();
	reti();
}