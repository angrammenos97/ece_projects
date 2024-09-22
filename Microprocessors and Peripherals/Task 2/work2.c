#include <stdio.h>
#include <stdlib.h>
#include "platform.h"
#include "gpio.h"

#define LED PA_5        // Use on board LED
#define BUTTON PC_13    // Use on board Button

#define MODE 0			// 0 -> User must press button when LED turns on
						// 1 -> User must press button when LED turns off
						
#define ITERATIONS 5	// Number of iterations 
												
#define cycles2ms(a) (((float)(a * 1000.0f)) / SystemCoreClock)     // Inline function to calculate ms from CPU cycles
#define ms2cysles(a) ((unsigned int)((a / 1000) * SystemCoreClock)) // Inline function to calculate CPU cycles from ms

volatile uint32_t cycles;   // Variable to hold past cycles

void button_callback_NOP(int status)    // Callback function to disable button press when not used
{
	__NOP();    //do nothing
}

void button_callback(int status)        // Callback interrupt 
{
	cycles = DWT->CYCCNT - 1;   //save past cycles
	gpio_toggle(LED);           //change state of the LED
	gpio_set_callback(BUTTON, button_callback_NOP);     //disable button press
}

int main()      // Main code
{
    // Initialize
	CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;		// These two lines unlock debug registers when not running in debug mode, 
	ITM->LAR = 0xC5ACCE55;								// otherwise program stucks at counting cycles
	DWT->CTRL |= 1;     //enable Cycle Count Register
	gpio_set_mode(LED, Output);     //LED's pin is Output
	gpio_set(LED, MODE);            //set LED initial state according to MODE
	gpio_set_mode(BUTTON, Input);       //BUTTON's pin is Input
	gpio_set_trigger(BUTTON, Rising);   //trigger interrupt at the Rising edge of the Button
	gpio_set_callback(BUTTON, button_callback_NOP);     //button is disable initially
	srand(DWT->CYCCNT);		//generate random numbers using counter value as a seed
	
	// Measure human's reflex
	unsigned int sum = 0;   // numerator of mean value
	for(int i = 0; i < ITERATIONS; i++) {   //test human for number of ITERATIONS
		unsigned int randomCycles = (((unsigned int)rand() % 3000) + 1000); //generate random ms value between 1000 and 4000
		randomCycles = ms2cysles(randomCycles);                             //and convert them to cycles   
		DWT->CYCCNT = 0;                    //reset Cycle Count Register
		while(DWT->CYCCNT < randomCycles)   //and wait until randomCycles past
			__NOP();
		gpio_set_callback(BUTTON, button_callback); //enable button
		gpio_toggle(LED);                           //change led state
		DWT->CYCCNT = 0;                    //reset Cycle Count Register
		while( (!gpio_get(LED)) == MODE) {  //wait until human press the button
			__WFI();    //aka Wait For Interrupt
		}
		sum += cycles;      //add to numerator current's test response time
	}
	float meanTime = cycles2ms(sum / ITERATIONS);   //calculate mean cycles and convert to ms
	return 0;
}
