/**
 * @file task.h
 */

#pragma once

#include <stdint.h>

#ifndef STACK_SIZE
	#define STACK_SIZE 300
#endif

typedef enum { READY, RUNNING, BLOCKING } taskState_t;

#define CMP_PRIORITY(t1, t2) (((int) t2->priority) - ((int) t1->priority))
	
/**
 * This struct represents a task. To use a task, the function
 * initTask must be called.
 */
typedef struct {
	int (*function)();
	
	unsigned int priority;
	volatile taskState_t state;
	
	/**
	 * Points to the next free memory on the stack.
	 */
	uint8_t *stackpointer;
	
	/**
	 * The stack is used for local variables and for saving the
	 * context of a task:
	 *
	 * - Return address
	 * - Registers
	 * - SREG	 
	 *
	 * @note This stack is not really in the .stack section.
	 */
	uint8_t stack[STACK_SIZE];
	
	
} task_t;

/**
 * This function initializes the given task_t struct.
 * It sets the stackpointer and fills the stack with initial
 * values. 
 */
void initTask(task_t *task);

/**
 * This function counts the number of tasks in the global
 * task struct array.
 */
int countTasks();