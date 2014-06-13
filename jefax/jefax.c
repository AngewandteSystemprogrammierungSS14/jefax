#include "task.h"
#include "dispatcher.h"
#include "user_counterTask.h"
#include "jefax_xmega128.h"
#include "scheduler_test.h"
#include "shell.h"
#include "car_control.h"

/**
 * The task list with all the tasks the dispatcher
 * should dispatch. The last entry hast to be a task
 * with 0 as the first entry.
 */
task_t TASKS[] = {
    /*{schedTestTask1, 2, READY, 0, {0}},
    {schedTestTask2, 2, READY, 0, {0}},
    {schedTestTask3, 1, READY, 0, {0}},
    {schedTestTask4, 2, READY, 0, {0}},*/
    //{counterTask1, 1, READY, 0, {0}},
    {counterTask2, 1, READY, 0, {0}},
    //CAR_TASK(1),
    SHELL_TASK(1),
    {0, 0, READY, 0, {0}}
};

void jefax()
{
    int i;
    for(i = 0; i < countTasks(); ++i)
        initTask(&TASKS[i]);

    initDispatcher();
}