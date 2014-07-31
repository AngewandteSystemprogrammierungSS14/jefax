/* The interrupt component of jefax provides macros and functions to work
 * with interrupts.
 * The usage of these functions is recommend to ensure consistency in jefax. */

#pragma once

#include <avr/interrupt.h>
#include "system.h"
#include "scheduler.h"

/* Disables interrupts and saves the working registers and the sreg on the stack. */
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
)

/* Restores the working registers and the sreg from the stack
 * and enables interrupts. */
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
					"sei							\n\t"   \
)

#define DISABLE_TIMER(timer) timer.CTRLA; timer.CTRLA = TC_CLKSEL_OFF_gc
#define ENABLE_TIMER(timer, prescaler) timer.CTRLA = prescaler
#define MS_PER_SEC 1000
#define MS_TO_TIMER(ms, prescaler) ((uint16_t) (ms * ((F_CPU / MS_PER_SEC) / getPrescalerValue(prescaler))))
#define TIMER_TO_MS(cnt, prescaler) ((uint16_t) ((cnt * getPrescalerValue(prescaler)) / (F_CPU / MS_PER_SEC)))
#define FORCE_INTERRUPT(timer) timer.CNT = timer.PER - 1

/* The jefax ISR saves the context of the preempted (running) task, executes
 * the given function and restores the context of the running task. */
#define JEFAX_ISR(vect, func)								\
ISR(vect, ISR_NAKED)										\
{															\
	SAVE_CONTEXT();											\
	getRunningTask()->stackpointer = (uint8_t *) SP;		\
	ENTER_SYSTEM_STACK();									\
	func();													\
	SP = (uint16_t) (getRunningTask()->stackpointer);		\
	RESTORE_CONTEXT();										\
	reti();													\
}

void enableInterrupts();
unsigned int getPrescalerValue(uint8_t p_prescaler);