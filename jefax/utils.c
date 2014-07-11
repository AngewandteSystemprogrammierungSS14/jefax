#include "utils.h"
#include <avr/interrupt.h>

uint8_t *main_stackpointer;

void enableInterrupts()
{
	PMIC.CTRL |= PMIC_LOLVLEN_bm;
	sei();
}

unsigned int getPrescalerValue(const uint8_t p_prescaler)
{
	switch(p_prescaler)
	{
		case TC_CLKSEL_OFF_gc:
			return 0;
		case TC_CLKSEL_DIV1_gc:
			return 1;
		case TC_CLKSEL_DIV2_gc:
			return 2;
		case TC_CLKSEL_DIV4_gc:
			return 4;
		case TC_CLKSEL_DIV8_gc:
			return 8;
		case TC_CLKSEL_DIV64_gc:
			return 64;
		case TC_CLKSEL_DIV256_gc:
			return 256;
		case TC_CLKSEL_DIV1024_gc:
			return 1024;
		default:
			return 0;
	}
}