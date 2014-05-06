#include "dispatcher.h"
#include <avr/interrupt.h>
#include <util/atomic.h>

/**
 * Disables interrupts and saves the working registers and the sreg on the stack.
 */
#define SAVE_CONTEXT()										\
	asm volatile (	"cli							\n\t"	\
					"push	r0						\n\t"	\
					"push	r1						\n\t"	\
					"push	r2						\n\t"	\
					"push	r3						\n\t"	\
					"push	r4						\n\t"	\
					"push	r5						\n\t"	\
					"push	r6						\n\t"	\
					"push	r7						\n\t"	\
					"push	r8						\n\t"	\
					"push	r9						\n\t"	\
					"push	r10						\n\t"	\
					"push	r11						\n\t"	\
					"push	r12						\n\t"	\
					"push	r13						\n\t"	\
					"push	r14						\n\t"	\
					"push	r15						\n\t"	\
					"push	r16						\n\t"	\
					"push	r17						\n\t"	\
					"push	r18						\n\t"	\
					"push	r19						\n\t"	\
					"push	r20						\n\t"	\
					"push	r21						\n\t"	\
					"push	r22						\n\t"	\
					"push	r23						\n\t"	\
					"push	r24						\n\t"	\
					"push	r25						\n\t"	\
					"push	r26						\n\t"	\
					"push	r27						\n\t"	\
					"push	r28						\n\t"	\
					"push	r29						\n\t"	\
					"push	r30						\n\t"	\
					"push	r31						\n\t"	\
					"in		r0, __SREG__			\n\t"	\
					"push	r0						\n\t"	\
					"clr	r1						\n\t"	\
);

/**
 * Restores the working registers and the sreg from the stack and
 * enables interrupts.
 */
#define RESTORE_CONTEXT()									\
	asm volatile (	"pop	r0						\n\t"	\
					"out	__SREG__, r0			\n\t"	\
					"pop	r31						\n\t"	\
					"pop	r30						\n\t"	\
					"pop	r29						\n\t"	\
					"pop	r28						\n\t"	\
					"pop	r27						\n\t"	\
					"pop	r26						\n\t"	\
					"pop	r25						\n\t"	\
					"pop	r24						\n\t"	\
					"pop	r23						\n\t"	\
					"pop	r22						\n\t"	\
					"pop	r21						\n\t"	\
					"pop	r20						\n\t"	\
					"pop	r19						\n\t"	\
					"pop	r18						\n\t"	\
					"pop	r17						\n\t"	\
					"pop	r16						\n\t"	\
					"pop	r15						\n\t"	\
					"pop	r14						\n\t"	\
					"pop	r13						\n\t"	\
					"pop	r12						\n\t"	\
					"pop	r11						\n\t"	\
					"pop	r10						\n\t"	\
					"pop	r9						\n\t"	\
					"pop	r8						\n\t"	\
					"pop	r7						\n\t"	\
					"pop	r6						\n\t"	\
					"pop	r5						\n\t"	\
					"pop	r4						\n\t"	\
					"pop	r3						\n\t"	\
					"pop	r2						\n\t"	\
					"pop	r1						\n\t"	\
					"pop	r0						\n\t"	\
					"sei							\n\t"	\
);

#define MS_TO_TIMER_PER(ms) (((unsigned long) ms) * 32000000UL / 256UL)

/**
 * The global, extern defined task list with all tasks to dispatch.
 */
extern task_t TASKS[];

extern int (*idleTaskFunction) ();

// Prototypes
static void setTimer();

/**
 * Which task is currently running:
 *
 * -1: Dispatchers idle task
 *  0: Task 0
 *  1: Task 1
 */
int currentTaskNumber = -1;

uint8_t *main_stackpointer;

void startDispatcher()
{
	initLED();
	enableInterrupts();
	
	initTask(getIdleTask());
	// Save the main context
	SAVE_CONTEXT();
	main_stackpointer = (uint8_t *) SP;
	
	// Switch to dispatcher idle task context
	SP = (uint16_t) (getIdleTask()->stackpointer);
	
	setTimer();
	
	// Start idle task
	idleTaskFunction();
}

/**
 * Sets the timer interrupt. At each interrupt the dispatcher changes
 * the running task. (Timer overflow IR is used).
 */
static void setTimer()
{
	// Set 16 bit timer
	TCC0.CTRLA = TC_CLKSEL_DIV256_gc; // 256 prescaler -> 3900 / sec -> 65536 max.
	TCC0.INTCTRLA = TC_OVFINTLVL_LO_gc; // Enable overflow interrupt level low
	TCC0.CTRLB = 0x00; // select Modus: Normal -> Event/Interrupt at top
	TCC0.CNT = 0x00;
}

void setInterruptTime(unsigned int p_msec)
{
	ATOMIC_BLOCK(ATOMIC_RESTORESTATE)
	{
		TCC0.PER = MS_TO_TIMER_PER(p_msec); // Top-Value (period)
	}
}

ISR(TCC0_OVF_vect, ISR_NAKED)
{
	SAVE_CONTEXT();
	
	if (currentTaskNumber == -1) {
		// Idle task
		
		// Save stackpointer
		getIdleTask()->stackpointer = (uint8_t *) SP;
		
		// Begin with the tasks
		currentTaskNumber = 0;
	} else {
		// Save stackpointer
		TASKS[currentTaskNumber].stackpointer = (uint8_t *) SP;
		
		// TODO: Is this okay?
		
		// Switch to dispatcher idle task context
		SP = (uint16_t) (getIdleTask()->stackpointer);
		
		// Got to next task: Switch between 0 and 1
		currentTaskNumber = (~currentTaskNumber & 0b00000001);
	}
	
	// Restore stackpointer
	SP = (uint16_t) (TASKS[currentTaskNumber].stackpointer);
	
	RESTORE_CONTEXT();
	reti();
}