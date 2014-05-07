#include "task.h"
#include "scheduler.h"
#include "user_counterTask.h"
#include "task.h"
#include "jefax_xmega128.h"

/**
 * The task list with all the tasks the dispatcher
 * should dispatch. The last entry hast to be a task
 * with 0 as the first entry.
 */
task_t TASKS[] = {
	{counterTask1, 1, READY, 0, {0}},
	{counterTask2, 1, READY, 0, {0}},
	{0, 0, READY, 0, {0}}
};

void jefax()
{	
	initTask(&TASKS[0]);
	initTask(&TASKS[1]);
	
	initScheduler();
}