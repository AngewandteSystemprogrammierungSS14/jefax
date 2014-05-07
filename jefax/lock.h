/*
 * lock.h
 *
 * Created: 01.05.2014 09:51:34
 *  Author: Fabian
 */ 

#pragma once

#include "task.h"

#define TASK_QUEUE_SIZE 10

struct taskQueue
{
	task_t *list[TASK_QUEUE_SIZE];
	uint8_t first;
	uint8_t count;
	uint8_t size;
};

typedef struct  
{
	struct taskQueue queue;
} signal_t;

typedef struct  
{
	unsigned int value;
	unsigned int maxValue;
	signal_t signal;
} semaphore_t;

typedef struct  
{
	semaphore_t lock;
} mutex_t;

typedef struct
{
	signal_t signal;
} condition_t;

int initSignal(signal_t *p_signal);
void waitSignal(signal_t *p_signal);
void signalOne(signal_t *p_signal);
void signalAll(signal_t *p_signal);

int initSemaphore(semaphore_t *p_semaphore, unsigned int p_maxValue);
void lockSemaphore(semaphore_t *p_semaphore);
void unlockSemaphore(semaphore_t *p_semaphore);

int initMutex(mutex_t *p_mutex);
void lockMutex(mutex_t *p_mutex);
void unlockMutex(mutex_t *p_mutex);

int initCondition(condition_t *p_cond);
void waitCondition(condition_t *p_cond, mutex_t *p_mutex);
void signalCondition(condition_t *p_cond);