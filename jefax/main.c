#ifndef F_CPU
#define F_CPU 32000000UL
#endif

#include "jefax.h"

int main()
{
	jefax(); // Dispatcher takes control
	
	// Never reached
	while (1) {}
	
	return 0;
}