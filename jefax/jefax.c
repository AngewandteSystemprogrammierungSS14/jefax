#include "task.h"
#include "schedulerRR.h"
#include "user_counterTask.h"
#include "task.h"
#include "jefax_xmega128.h"
#include "scheduler_test.h"

/**
 * The task list with all the tasks the dispatcher
 * should dispatch. The last entry hast to be a task
 * with 0 as the first entry.
 */
task_t TASKS[] = {
	{schedTestTask5, 1, READY, 0, {0}},
	{schedTestTask6, 1, READY, 0, {0}},
	{0, 0, READY, 0, {0}}
};

void jefax()
{	
	int i;
	for(i = 0; i < countTasks(); ++i)
		initTask(&TASKS[i]);
	
	initScheduler(getRRScheduler());
}