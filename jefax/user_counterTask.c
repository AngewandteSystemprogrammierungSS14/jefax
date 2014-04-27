#include "user_counterTask.h"
#include "jefax_xmega128.h"
#include <util/delay.h>

/**
 * Use LED 1 on board.
 */
#define LED_MASK1 0xFE

/**
 * Use LED 2 on board.
 */
#define LED_MASK2 0xFD

int counterTask1()
{
	volatile int counter;
	
	while (1) {
		++counter;
		
		setLED(LED_MASK1);
		//_delay_ms(100);
		setLED(0xFF);
	}
	
	return 0;
}

int counterTask2()
{
	volatile int counter;
	
	while (1) {
		++counter;
		
		setLED(LED_MASK2);
		//_delay_ms(100);
		setLED(0xFF);
	}
	
	return 0;
}