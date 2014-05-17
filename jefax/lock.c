/*
 * lock.c
 *
 * Created: 01.05.2014 09:52:02
 *  Author: Fabian
 */ 
#include "lock.h"
#include "scheduler.h"
#include <util/atomic.h>
#include <stddef.h>

int initSignal(signal_t *p_signal)
{
	return initTaskList(&(p_signal->queue));
}

void waitSignal(signal_t *p_signal)
{
	ATOMIC_BLOCK(ATOMIC_RESTORESTATE)
	{
		pushTaskBack(&(p_signal->queue), getRunningTask());
		setTaskState(getRunningTask(), BLOCKING);
	}
}

void signalOne(signal_t *p_signal)
{
	ATOMIC_BLOCK(ATOMIC_RESTORESTATE)
	{
		task_t *task = popTaskFront(&(p_signal->queue));
		if(task != NULL)
			setTaskState(task, READY);
	}
}

void signalAll(signal_t *p_signal)
{
	ATOMIC_BLOCK(ATOMIC_RESTORESTATE)
	{
		task_t *task;
		while(p_signal->queue.count > 0)
		{
			task = popTaskFront(&(p_signal->queue));
			if(task != NULL)
				setTaskState(task, READY);
		}
	}
}

int initSemaphore(semaphore_t *p_semaphore, unsigned int p_maxValue)
{
	int ret = initSignal(&(p_semaphore->signal));
	if(ret != 0)
		return ret;
	
	p_semaphore->value = 0;
	p_semaphore->maxValue = p_maxValue;
	
	return ret;
}

void lockSemaphore(semaphore_t *p_semaphore)
{
	ATOMIC_BLOCK(ATOMIC_RESTORESTATE)
	{
		p_semaphore->value += 1;
		if(p_semaphore->value > p_semaphore->maxValue)
			waitSignal(&(p_semaphore->signal));
	}
}

void unlockSemaphore(semaphore_t *p_semaphore)
{
	ATOMIC_BLOCK(ATOMIC_RESTORESTATE)
	{
		p_semaphore->value -= 1;
		signalOne(&(p_semaphore->signal));
	}
}

int initMutex(mutex_t *p_mutex)
{
	return initSemaphore(&(p_mutex->lock), 1);
}

void lockMutex(mutex_t *p_mutex)
{
	lockSemaphore(&(p_mutex->lock));
}

void unlockMutex(mutex_t *p_mutex)
{
	unlockSemaphore(&(p_mutex->lock));
}

int initCondition(condition_t *p_cond)
{
	return initSignal(&(p_cond->signal));
}

void waitCondition(condition_t *p_cond, mutex_t *p_mutex)
{
	ATOMIC_BLOCK(ATOMIC_RESTORESTATE)
	{
		unlockMutex(p_mutex);
		waitSignal(&(p_cond->signal));
	}
	lockMutex(p_mutex);
}

void signalCondition(condition_t *p_cond)
{
	signalOne(&(p_cond->signal));
}