/*
 * lock.c
 *
 * Created: 01.05.2014 09:52:02
 *  Author: Fabian
 */ 
#include "lock.h"
#include <util/atomic.h>
#include <stddef.h>
#include "scheduler.h"

void enqueueTask(struct taskQueue *p_queue, task_t* p_task);
task_t* dequeueTask(struct taskQueue *p_queue);

int initSignal(signal_t *p_signal)
{
	p_signal->queue.count = 0;
	p_signal->queue.first = 0;
	p_signal->queue.size = TASK_QUEUE_SIZE;
	
	return 0;
}

void waitSignal(signal_t *p_signal)
{
	ATOMIC_BLOCK(ATOMIC_RESTORESTATE)
	{
		enqueueTask(&(p_signal->queue), getRunningTask());
		setTaskState(getRunningTask(), BLOCKING);
	}
}

void signalOne(signal_t *p_signal)
{
	ATOMIC_BLOCK(ATOMIC_RESTORESTATE)
	{
		task_t *task = dequeueTask(&(p_signal->queue));
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
			task = dequeueTask(&(p_signal->queue));
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
		int hasLock = 0;
		while(!hasLock)
		{
			hasLock = p_semaphore->value < p_semaphore->maxValue;
			if(hasLock)
				++p_semaphore->value;
			else
				waitSignal(&(p_semaphore->signal));
		}
	}
}

void unlockSemaphore(semaphore_t *p_semaphore)
{
	ATOMIC_BLOCK(ATOMIC_RESTORESTATE)
	{
		--p_semaphore->value;
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

void enqueueTask(struct taskQueue *p_queue, task_t* p_task)
{
	uint8_t index = (p_queue->first + p_queue->count) % p_queue->size;
	p_queue->list[index] = p_task;
	++(p_queue->count);
}

task_t* dequeueTask(struct taskQueue *p_queue)
{
	task_t *result = NULL;
	if(p_queue->count > 0)
	{
		result = p_queue->list[p_queue->first];
		--(p_queue->count);
		++(p_queue->first);
	}
	
	return result;
}