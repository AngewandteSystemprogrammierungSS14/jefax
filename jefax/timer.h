/*
 * timer.h
 *
 * Created: 19.05.2014 15:34:07
 *  Author: Fabian
 */ 

#pragma once

typedef struct{
	void (*callBack) (void*);
	void *arg;
	int ms;
} timer_t;

int initTimerSystem();
int initTimer(timer_t *p_timer, int p_ms, void (*p_callBack) (void*), void * p_arg);
int addTimer(timer_t p_timer);