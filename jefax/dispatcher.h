/* The dispatcher exchanges the running task with another task.
 * The decision which task shall be running next is done by the scheduler.
 * The time slice interrupt is also handled in this component. If a task is
 * interrupted its context (registers, SREG) is saved to its stack. On
 * restore the contex is popped from the stack. */

#pragma once

void initDispatcher();

/* Sets the time slice after which a task is exchanged. */
void setInterruptTime(unsigned int p_msec);
