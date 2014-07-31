#include <stdint.h>
#include <stdlib.h>
#include "car_control.h"
#include "../../motor_api/PWM_driver.h"
#include "shell.h"
#include "scheduler.h"
#include "usart.h"

#define MIN_SPEED 350
#define MAX_SPEED 600
#define SPEED_STEP 50
#define MIN(a,b) ((a) < (b) ? (a) : (b))
static int16_t speed = 0;

void processCarMessage(char *data)
{
    int16_t toChange;
	char str[10];
	
    if (*data == 'w') {
		if(speed == 0)
			toChange = MIN_SPEED;
		else
			toChange = MIN(SPEED_STEP, MAX_SPEED - speed);
        speed += toChange;
    } else if (*data == 's') {
        toChange = MIN(SPEED_STEP, speed);
        speed -= toChange;
		if(speed < MIN_SPEED)
			speed = 0;
    } else if (*data == 'i') {
		print("Speed: ");
		itoa(speed, str, 10);
		print(str);
		print("\r\n");
	}

    carSetSpeed(speed);
}

int carTask()
{
    motorInitDriver();
    setMessageCallback(processCarMessage);

    while (1) {
        sleep(1000);
    }

    return 0;
}