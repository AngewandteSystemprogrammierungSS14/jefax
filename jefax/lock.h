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
	uint8_t lock;
	struct taskQueue queue;
	
} mutex_t;

typedef struct
{
	struct taskQueue queue;
} condition_t;

int initMutex(mutex_t *p_mutex);
void lockMutex(mutex_t *p_mutex);
void unlockMutex(mutex_t *p_mutex);

int initCondition(condition_t *p_cond);
void waitCondition(condition_t *p_cond, mutex_t *p_mutex);
void signalCondition(condition_t *p_cond);