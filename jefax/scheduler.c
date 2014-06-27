#include <avr/interrupt.h>
#include "scheduler.h"
#include "atomic.h"
#include "timer.h"
#include "utils.h"
#include "jefax_xmega128.h"

static int initTaskLists();
static void sleepTimerCallback(void *arg);
static int idleTaskFunction();

extern task_t TASKS[];

static task_t *runningTask;
static taskList_t readyList;
static taskList_t blockingList;
static scheduler_t *scheduler;

/* idleTask runs if no other task can be found to be scheduled. */
static task_t idleTask = { idleTaskFunction, 255, READY, 0, {0} };

int initScheduler(scheduler_t *p_defaultScheduler)
{
	int ret;
	initTask(&idleTask);
	
	ret = initTaskLists();
	if(ret)
		return ret;
		
	ret = initTimerSystem();
	if(ret)
		return ret;
		
	setScheduler(p_defaultScheduler);
	
	return 0;
}

static int initTaskLists()
{
	if(initTaskList(&readyList))
		return -1;
	if(initTaskList(&blockingList))
		return -1;
	
	int taskCount = countTasks();
	if(taskCount <= 0)
		return -1;
	
	// first task running is the idleTask, but exchanged right after initialization
	runningTask = &idleTask;
	
	// insert all tasks in ready list, so they are registered for scheduling
	int i;
	for(i = 0; i < taskCount; ++i) {
		TASKS[i].state = READY;
		pushTaskBack(&readyList, &TASKS[i]);
	}
	
	return 0;
}

task_t* schedule()
{
	runningTask = scheduler->getNextTask();
	// no task was found, so schedule idleTask
	if(runningTask == NULL)
		runningTask = &idleTask;
	runningTask->state = RUNNING;
	return runningTask;
}

void yield()
{
	setTaskState(runningTask, READY);
}

void sleep(const int p_ms)
{
	uint8_t irEnabled = enterAtomicBlock();
	
	timer_t timer;
	initTimer(&timer, p_ms, sleepTimerCallback, getRunningTask());
	addTimer(timer);
	setTaskState(getRunningTask(), BLOCKING);
	
	exitAtomicBlock(irEnabled);
}

/* runs in interrupt context */
static void sleepTimerCallback(void *arg)
{
	task_t *task = (task_t*) arg;
	task->state = READY;
	scheduler->taskWokeUp(task);
}

void setTaskState(task_t *p_task, taskState_t p_state)
{
	uint8_t irEnabled = enterAtomicBlock();
	
	p_task->state = p_state;
	scheduler->taskStateChanged(p_task);
		
	exitAtomicBlock(irEnabled);	
}

void forceContextSwitch()
{
	// save interrupt enable state
	uint8_t state = SREG & 0x80;
	// create interrupt
	sei();
	FORCE_INTERRUPT(TCC0);
	
	// wait to be exchanged
	while(!TASK_IS_RUNNING(runningTask))
	{ }
	
	if(!state)
		cli();
}

void setScheduler(scheduler_t *p_scheduler)
{
	scheduler = p_scheduler;
	scheduler->readyList = &readyList;
	scheduler->blockingList = &blockingList;
	scheduler->init();
}

task_t *getRunningTask()
{
	return runningTask;
}

int hasRunningTask()
{
	return runningTask != &idleTask;
}

static int idleTaskFunction()
{
	uint8_t led = 0;
	
	while (1) {
		setLED(~(1 << led++));
		//_delay_ms(500);
		if (led == 8)
			led = 0;
	}
	
	return 0;
}

