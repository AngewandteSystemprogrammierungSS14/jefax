/*
 * scheduler_test.c
 *
 * Created: 16.05.2014 14:06:52
 *  Author: Fabian
 */ 

#include "lock.h"
#include "scheduler.h"

#define TIME_STEP 10
#define MAX_X 1024
#define MAX_Y 1024
#define MAX_THETA 360

static mutex_t posMutex;
static condition_t posUpdate;
static int x = 0;
static int y = 0;
static int theta = 0;

static mutex_t velMutex;
static int x_vel = 1;
static int y_vel = 2;
static int ang_vel = 6;

static int resultGlob;

static mutex_t simpleTestMutex;

void initSchedTest()
{
	initMutex(&posMutex);
	initMutex(&velMutex);
	initMutex(&simpleTestMutex);
	initCondition(&posUpdate);
}

int schedTestTask1()
{
	while(1)
	{
		lockMutex(&posMutex);
		lockMutex(&velMutex);
		
		//koppelnav
		x = (x + x_vel * TIME_STEP) % MAX_X;
		y = (y + y_vel * TIME_STEP) % MAX_Y;
		theta = (theta + ang_vel * TIME_STEP) % MAX_THETA;
		
		unlockMutex(&velMutex);
		unlockMutex(&posMutex);
		
		signalCondition(&posUpdate);
	}
}

int schedTestTask2()
{
	int lastX = 0;
	int lastY = 0;
	int lastTheta = 0;
	
	while(1)
	{
		lockMutex(&posMutex);
		while(lastX == x && lastY == y && lastTheta == theta)
			waitCondition(&posUpdate, &posMutex);
			
		lastX = x;
		lastY = y;
		lastTheta = theta;
		
		unlockMutex(&posMutex);
	}
}

int schedTestTask3()
{
	while(1)
	{
		
	}
}

int schedTestTask4()
{
	volatile int tmp;
	while(1)
	{
		lockMutex(&simpleTestMutex);
		
		tmp = resultGlob;
		//yield
		setTaskState(getRunningTask(), READY);
		
		unlockMutex(&simpleTestMutex);
	}
}

int schedTestTask5()
{
	int i, result;
	
	while(1)
	{
		lockMutex(&simpleTestMutex);
		
		result = resultGlob;
		for(i = 0; i < 1000; ++i)
		{
			result += 4 * i + 20;
		}
		
		resultGlob = result;	
		unlockMutex(&simpleTestMutex);
	}
}