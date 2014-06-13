#include "jefax_xmega128.h"
#include <avr/io.h>

void initLED()
{
    PORTE.DIR = 0xFF;
    PORTE.OUT = 0xFF;
}

void setLED(uint8_t status)
{
    PORTE.OUT = status;
}

void setLEDPin(uint8_t pin)
{
    PORTE.OUTCLR = pin;
}

void clearLEDPin(uint8_t pin)
{
    PORTE.OUTSET = pin;
}
