#include <stddef.h>
#include "lock.h"
#include "scheduler.h"
#include "atomic.h"

int initSignal(signal_t *p_signal)
{
	return initTaskList(&(p_signal->queue));
}

void waitSignal(signal_t *p_signal)
{
	uint8_t irEnabled = enterAtomicBlock();
	
	// insert task with highest priority on highest index
	insertTaskPriorityAsc(&(p_signal->queue), getRunningTask());
	// change to blocking state and wait until wakeup
	setTaskState(getRunningTask(), BLOCKING);
	
	exitAtomicBlock(irEnabled);
}

static void wakeUpFirst(signal_t *p_signal)
{
	// wake up the first in the queue
	task_t *task = popTaskBack(&(p_signal->queue));
	if(task != NULL)
		setTaskState(task, READY);
}

void signalOne(signal_t *p_signal)
{
	uint8_t irEnabled = enterAtomicBlock();
	
	wakeUpFirst(p_signal);
			
	exitAtomicBlock(irEnabled);
}

void signalAll(signal_t *p_signal)
{
	uint8_t irEnabled = enterAtomicBlock();
	
	while(p_signal->queue.count > 0)
		wakeUpFirst(p_signal);
	
	exitAtomicBlock(irEnabled);
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
	uint8_t irEnabled = enterAtomicBlock();
	
	p_semaphore->value += 1;
	if(p_semaphore->value > p_semaphore->maxValue)
		waitSignal(&(p_semaphore->signal));
		
	exitAtomicBlock(irEnabled);
}

void unlockSemaphore(semaphore_t *p_semaphore)
{
	uint8_t irEnabled = enterAtomicBlock();
	
	p_semaphore->value -= 1;
	signalOne(&(p_semaphore->signal));
		
	exitAtomicBlock(irEnabled);
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
	uint8_t irEnabled = enterAtomicBlock();
	
	unlockMutex(p_mutex);
	waitSignal(&(p_cond->signal));
		
	exitAtomicBlock(irEnabled);
	lockMutex(p_mutex);
}

void signalCondition(condition_t *p_cond)
{
	signalOne(&(p_cond->signal));
}