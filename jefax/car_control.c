#include "car_control.h"
//#include "../../motor_api/PWM_driver.h"
#include "shell.h"
#include "scheduler.h"
#include <stdint.h>
#include <stdlib.h>

#define MIN(a,b) ((a) < (b) ? (a) : (b))
int16_t speed = 0;

void processCarMessage(char *data)
{
    int16_t toChange;
    if (*data == 'w') {
        toChange = MIN(INT16_MAX / 10, INT16_MAX - abs(speed));
        speed += toChange;
    } else if (*data == 's') {
        toChange = MIN(INT16_MAX / 10, INT16_MAX - abs(speed));
        speed -= toChange;
    }

    //carSetSpeed(speed);
}

int carTask()
{
    //motorInitDriver();
    //setMessageCallback(processCarMessage);

    while (1) {
        sleep(1000);
    }

    return 0;
}