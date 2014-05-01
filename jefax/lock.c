/*
 * lock.c
 *
 * Created: 01.05.2014 09:52:02
 *  Author: Fabian
 */ 

#include <avr/interrupt.h>
#include <stddef.h>
#include "lock.h"

int safeCLI();
void safeSEI(const uint8_t p_state);

void enqueueTask(struct taskQueue *p_queue, task_t* p_task);
task_t* dequeueTask(struct taskQueue *p_queue);

int initMutex(mutex_t *p_mutex)
{
	p_mutex->lock = 0;
	p_mutex->queue.count = 0;
	p_mutex->queue.first = 0;
	p_mutex->queue.size = TASK_QUEUE_SIZE;
	return 0;
}

void lockMutex(mutex_t *p_mutex)
{
	uint8_t interruptStatus = safeCLI();
	
	uint8_t locked = p_mutex->lock;
	if(locked)
	{
		//TODO enqueueTask(currentTask);
		//TODO setTaskState(currentTask, blockingMode);
	}
	
	p_mutex->lock = 1;
	safeSEI(interruptStatus);
	
	if(locked)
		;//TODO scheduleTask(currentTask);
}

void unlockMutex(mutex_t *p_mutex)
{
	uint8_t interruptStatus = safeCLI();
	
	task_t *task = dequeueTask(&(p_mutex->queue));
	
	//nobody is waiting for mutex anymore
	if(task == NULL)
		p_mutex->lock = 0;
	else
		;//TODO setTaskState(task, readyMode);
	
	
	safeSEI(interruptStatus);
	if(task != NULL)
		;//TODO scheduleTask(task)
}

int initCondition(condition_t *p_cond)
{
	p_cond->queue.count = 0;
	p_cond->queue.first = 0;
	p_cond->queue.size = TASK_QUEUE_SIZE;
	
	return 0;
}

void waitCondition(condition_t *p_cond, mutex_t *p_mutex)
{
	uint8_t interruptStatus = safeCLI();
	
	//TODO enqueueTask(&(p_cond->queue), currentTask);
	unlockMutex(p_mutex);
	//TODO setTaskState(currentTask, blockingMode);
	safeSEI(interruptStatus);
	
	//TODO schedule(currentTask);
	
	//if condition gets signaled, task resumes here
	lockMutex(p_mutex);
}

void signalCondition(condition_t *p_cond)
{
	uint8_t interruptStatus = safeCLI();
	
	task_t *task = dequeueTask(&(p_cond->queue));
	if(task != NULL)
		;//TODO setTaskState(task, readyMode);
	
	safeSEI(interruptStatus);
}

int safeCLI()
{
	uint8_t result = SREG & 0x80;
	cli();
	return result;
}

void safeSEI(const uint8_t p_state)
{
	if(p_state & 0x80)
		sei();
	/*else		TODO is this needed?
		cli();*/
}

void enqueueTask(struct taskQueue *p_queue, task_t* p_task)
{
	uint8_t index = (p_queue->first + p_queue->count) % p_queue->size;
	p_queue->list[index] = p_task;
	++p_queue->count;
}

task_t* dequeueTask(struct taskQueue *p_queue)
{
	task_t *result = NULL;
	if(p_queue->count > 0)
	{
		result = p_queue->list[p_queue->first];
		--p_queue->count;
		++p_queue->first;
	}
	
	return result;
}