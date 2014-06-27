/* The timer components allows actions to be called at some point in the future.
 * initTimer() takes a relative time value, a callback and alternatively an
 * argument for the callback.
 * int initTimerSystem() has to be called before any timer can be used
 * correctly. */

#pragma once

typedef struct{
	void (*callBack) (void*);
	void *arg;
	volatile unsigned int ms;
} timer_t;

int initTimerSystem();
int initTimer(timer_t *p_timer, unsigned int p_ms, void (*p_callBack) (void*), void * p_arg);
int addTimer(timer_t p_timer);
