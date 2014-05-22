/*
 * scheduler_test.c
 *
 * Created: 16.05.2014 14:06:52
 *  Author: Fabian
 */ 

#include "lock.h"
#include "scheduler.h"
#include <math.h>

#define TIME_STEP 10
#define MAX_X 50
#define MAX_Y 50
#define MAX_THETA 360

DECLARE_MUTEX(posMutex);
DECLARE_CONDITION(posUpdate);
static int x = 0;
static int y = 0;
static int theta = 0;

DECLARE_MUTEX(velMutex);
static int x_vel = 1;
static int y_vel = 2;
static int ang_vel = 6;

static int resultGlob;

DECLARE_SIGNAL(measureSignal);

DECLARE_MUTEX(simpleTestMutex);
static int measureDistance;
static int field[MAX_X][MAX_Y];

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
		
		yield();
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
		sleep(1);
		measureDistance = 10;
		signalOne(&measureSignal);
	}
}

int schedTestTask4()
{
	int dx, dy;
	while(1)
	{
		waitSignal(&measureSignal);
		lockMutex(&posMutex);
		
		dx = cos(theta) * measureDistance;
		dy = sin(theta) * measureDistance;
		
		field[(x + dx) % MAX_X][(y + dy) % MAX_Y] += 1;
		
		unlockMutex(&posMutex);
	}
}

int schedTestTask5()
{
	volatile int tmp;
	while(1)
	{
		lockMutex(&simpleTestMutex);
		
		tmp = resultGlob;
		
		yield();
		
		unlockMutex(&simpleTestMutex);
	}
}

int schedTestTask6()
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