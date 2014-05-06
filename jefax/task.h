/**
 * @file task.h
 */

#pragma once

#include <stdint.h>

#define STACK_SIZE 100

typedef enum { READY, RUNNING, BLOCKING } taskState_t;
	
/**
 * This struct represents a task. To use a task, the function
 * initTask must be called.
 */
typedef struct {
	int (*function)();
	
	/**
	 * Indicates for example if the task is running.
	 */
	uint8_t status;
	
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
	
	unsigned int priority;
	taskState_t state;
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

task_t *getIdleTask();