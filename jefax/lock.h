/* The lock component provides severals structs and functions
 * for synchronisation purpose. */

#pragma once

#include "tasklist.h"

#define DECLARE_SIGNAL(name) signal_t name = { { {0}, DEF_TASK_LIST_SIZE, 0 } }
#define DECLARE_SEMAPHORE(name, val) semaphore_t name = { 0, val, { { {0}, DEF_TASK_LIST_SIZE, 0 } } }
#define DECLARE_MUTEX(name) mutex_t name = { { 0, 1, { { {0}, DEF_TASK_LIST_SIZE, 0 } } } }
#define DECLARE_CONDITION(name) condition_t name = { { { {0}, DEF_TASK_LIST_SIZE, 0 } } }

typedef struct {
	taskList_t queue;
} signal_t;

typedef struct {
	volatile unsigned int value;
	volatile unsigned int maxValue;
	signal_t signal;
} semaphore_t;

typedef struct {
	semaphore_t lock;
} mutex_t;

typedef struct {
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