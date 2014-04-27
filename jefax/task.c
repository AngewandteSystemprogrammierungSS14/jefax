#include "task.h"

extern task_t TASKS[];

void initTask(task_t *task)
{
	int i;

	// Set the stackpointer to the top of the stack.
	task->stackpointer = task->stack + STACK_SIZE - 1;
	
	// Push the function address on the tasks stack -> Program counter (3 Byte).
	*(task->stackpointer) = ((uint8_t *) (&task->function))[0];
	task->stackpointer--;
	*(task->stackpointer) = ((uint8_t *) (&task->function))[1];
	task->stackpointer--;
	*(task->stackpointer) = 0;
	task->stackpointer--;
	
	// Working registers
	for (i = 31; i >= 0; i--) {
		*(task->stackpointer) = 0;
		task->stackpointer--;
	}

	// SREG
	*(task->stackpointer) = 0;
	task->stackpointer--;
}

int countTasks()
{	
	int i = 0;
	while (TASKS[i].function()) {
		++i;
	}
	return i;
}
