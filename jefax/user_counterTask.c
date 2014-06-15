#include "user_counterTask.h"
#include "jefax_xmega128.h"
#include "scheduler.h"

#include <util/delay.h>

#define LED2 0x02
#define LED3 0x04

int counterTask1()
{
    volatile int counter;

    while (1) {
        ++counter;

        setLEDPin(LED3);

        sleep(50);
        //_delay_ms(500);

        clearLEDPin(LED3);

        //_delay_ms(200);
    }

    return 0;
}

int counterTask2()
{
    volatile int counter;

    while (1) {
        ++counter;

        setLEDPin(LED2);

        sleep(500);
        //_delay_ms(500);

        clearLEDPin(LED2);

        sleep(500);
        //_delay_ms(200);
    }

    return 0;
}