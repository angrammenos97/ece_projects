#include <stdint.h>
#include "platform.h"
#include "onewire.h"
#include "ds18b20.h"
#include "stopwatch.h"
#include "hcsr04.h"
#include "lcd.h"
#include "gpio.h"
#include "delay.h"
#include "queue.h"

#define DS18B20_DATA_PIN PC_7
#define HCSR04_TRIGGER_PIN PB_9
#define HCSR04_ECHO_PIN PB_8

#define LOWER_THRESHOLD 20.0f
#define UPPER_THRESHOLD 29.0f
#define RELAY_THRESHOLD 30.5f

#define LCD_TRIGGER_DISTANCE 50.0f			//closer than 50cm
#define MESSAGE_THRESHOLD_DISPLAY_TIME 15	// 15 = 3sec/200ms

volatile unsigned int counter;
unsigned int remainingTimes[5];
float lastTemp, previousMeanTemp;
Queue jobs;

uint8_t init_system(void);		// Function that initializes variables and sensors
void what_to_print(void);		// Function that decides what to print on LCD and is called with every tick of the timer
void float2char(char* str, float a, uint8_t decimalDigits, uint8_t celciusFlag);	// Function that converts float number to string for printing to LCD

void SysTick_Handler() {
    counter++;
    if(!queue_enqueue(&jobs, 0))    						// Enqueue ultrasonic job
        lcd_print("Enqueue 0 failed\n");
    if((counter % 25 == 0) && (!queue_enqueue(&jobs, 1)))   // Enqueue get temperature job
        lcd_print("Enqueue 1 failed\n");
    if(counter % 600 == 0) {    							// Enqueue print mean temperature
        if(!queue_enqueue(&jobs, 2))
            lcd_print("Enqueue 2 failed\n");
        counter = 0;
    }
	what_to_print();		// Print something for the next  200ms
}

int main() {
	if(!init_system())	//initialize all modules
		return -1;		//failed
	float meanTemp = 0.0f, distance;
	uint8_t job;
    while(1) {
        if(queue_dequeue(&jobs, &job)) {	//check if there is a new job
            if(job == 0) {	//check the proximity sensor
                distance = hcsr04_get_distance();
                if(distance < LCD_TRIGGER_DISTANCE)
					remainingTimes[0] = 5;	// 5 = 1sec/200ms
            }
            else if(job == 1) {	//get current temperature
                lastTemp = readTemperature(DS18B20_DATA_PIN);
                meanTemp += lastTemp;
                if(lastTemp < LOWER_THRESHOLD) {
					if(!gpio_get(P_LED_G))	//print only once
						remainingTimes[1] = MESSAGE_THRESHOLD_DISPLAY_TIME;
                    gpio_set(P_LED_G, 1);					
				}
                else {
                    gpio_set(P_LED_G, 0);
                    remainingTimes[1] = 0;
                }
                if(lastTemp > UPPER_THRESHOLD) {
					if(!gpio_get(P_LED_R))	//print only once
						remainingTimes[2] = MESSAGE_THRESHOLD_DISPLAY_TIME;
                    gpio_set(P_LED_R, 1);
				}
                else {
                    gpio_set(P_LED_R, 0);
                    remainingTimes[2] = 0;
                }
                if(lastTemp > RELAY_THRESHOLD) {
					if(!gpio_get(P_LED_B))	//print only once
						remainingTimes[3] = MESSAGE_THRESHOLD_DISPLAY_TIME;
                    gpio_set(P_LED_B, 1);
				}
                else {
                    gpio_set(P_LED_B, 0);
                    remainingTimes[3] = 0;
                }
            }
            else {	//print the mean temperature
				previousMeanTemp = meanTemp / 24.0f;
				meanTemp = 0.0f;
				remainingTimes[4] = 50;
            }
        }
    }
    //return 0;
}

uint8_t init_system() {
	CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
	ITM->LAR = 0xC5ACCE55;
	DWT->CTRL |= 1;
	
	SysTick->CTRL &= ~SysTick_CTRL_ENABLE_Msk;
	counter = 0;
	
	lastTemp = 24.0f;
	previousMeanTemp = -273.15f;	//absolute zero

	// LCD Init
	lcd_init();
	lcd_clear();
	
	// Global variables init
	for(int i = 0; i < 5; i++)
		remainingTimes[i] = 0;
    if(!queue_init(&jobs, 20))
        return 0;
	
	// LED Init
    gpio_set_mode(P_LED_R, Output);
	gpio_set_mode(P_LED_G, Output);
	gpio_set_mode(P_LED_B, Output);
    
	// Ultrasonic sensor init
    hcsr04_init(HCSR04_TRIGGER_PIN, HCSR04_ECHO_PIN);
	
	// DS18B20/1-Wire Init
    onewire_init(DS18B20_DATA_PIN);
	
	// Check DS18B20 for response
    if(!configSensor(DS18B20_DATA_PIN, 0x64, 0x00, 0x3f))
        return 0;
	
	// Request temperature to change the power-on reset value of the temperature register, which is +85 degrees
	lastTemp = readTemperature(DS18B20_DATA_PIN);
	
	// Timer config
    SysTick_Config(SystemCoreClock / 5);    // Every 200ms
	SysTick->CTRL = SysTick_CTRL_CLKSOURCE_Msk | SysTick_CTRL_TICKINT_Msk | SysTick_CTRL_ENABLE_Msk;
	counter = 0;
	lcd_clear();
	return 1;
}

void what_to_print() {
	static int currentJob = -1;
    static char value[9];
	static float lastknownTemp;
	if (remainingTimes[2] > 0) {   //upper temp
		if(!(currentJob == 2) || (lastknownTemp != lastTemp)) {
			lcd_clear();
			lastknownTemp = lastTemp;
			float2char(value, lastTemp, 2, 1);
			lcd_print("Temperature HIGH");
			lcd_print("Now: ");
			lcd_print(value);
			currentJob = 2;
		}
        remainingTimes[2]--;
		if(remainingTimes[2] == 0) {
			lcd_clear();
			currentJob = -1;
		}
    }
	else if (remainingTimes[3] > 0) {   //relay temp
		if(!(currentJob == 3) || (lastknownTemp != lastTemp)) {
			lcd_clear();
			lastknownTemp = lastTemp;
			float2char(value, lastTemp, 2, 1);
			lcd_print("Fan Enabled.    ");
			lcd_print("Now: ");
			lcd_print(value);
			currentJob = 3;
		}
        remainingTimes[3]--;
		if(remainingTimes[3] == 0) {
			lcd_clear();
			currentJob = -1;
		}
    }
	else if (remainingTimes[1] > 0) {   //lower temp
		if(!(currentJob == 1) || (lastknownTemp != lastTemp)) {
			lcd_clear();
			lastknownTemp = lastTemp;
			float2char(value, lastTemp, 2, 1);
			lcd_print("Temperature LOW ");
			lcd_print("Now: ");
			lcd_print(value);
			currentJob = 1;
		}
        remainingTimes[1]--;
		if(remainingTimes[1] == 0) {
			lcd_clear();
			currentJob = -1;
		}
    }
    else if (remainingTimes[0] > 0) {    //current and mean
		if(!(currentJob == 0) || (lastknownTemp != lastTemp)) {
			lcd_clear();
			lastknownTemp = lastTemp;
			float2char(value, lastTemp, 2, 1);
			lcd_print("Now:  ");
			lcd_print(value);
			if (!(previousMeanTemp == -273.15f)) {	//don't print mean at first time
				lcd_set_cursor(0,1);
				float2char(value, previousMeanTemp, 2, 1);
				lcd_print("Mean: ");
				lcd_print(value);
			}
			currentJob = 0;
		}
        remainingTimes[0]--;
		if(remainingTimes[0] == 0) {
			lcd_clear();
			currentJob = -1;
		}
    }
	else if (remainingTimes[4] > 0) {   //mean
		if(!(currentJob == 4)) {
			lcd_clear();
			float2char(value, previousMeanTemp, 2, 1);
			lcd_print("Mean: ");
			lcd_print(value);
			currentJob = 4;
		}
        remainingTimes[4]--;
		if(remainingTimes[4] == 0) {
			lcd_clear();
			currentJob = -1;
		}
    }
}

void float2char(char* str, float a, uint8_t decimalDigits, uint8_t celciusFlag) {
	uint8_t pos = 0, integerDigits = 0;
	int integerPart, temp;
	float decimalPart;
	integerPart = (int)a;
	a -= (float)integerPart;
	if(a < 0) {
		str[pos++] = (char)45;
		integerPart *= -1;
		a *= -1;
	}
	temp = integerPart;
	decimalPart = a;
	while(temp != 0) {
		integerDigits++;
		temp /= 10;
	}
	if(integerDigits == 0 && integerPart == 0)
        str[pos++] = '0';
	for(uint8_t i = 0; i < integerDigits; i++) {
		temp = integerPart;
		for(uint8_t j = 0; j < integerDigits-i-1; j++)
			temp /= 10;
		temp = temp % 10;
		str[pos++] = '0' + temp;
	}
	str[pos++] = (char)46;
	for(uint8_t i = 0; i < decimalDigits; i++) {
        decimalPart *= 10;
		temp = (int)decimalPart;
		str[pos++] = '0' + temp;
        decimalPart -= temp;
	}
	if(celciusFlag != 0) {
		str[pos++] = (char)223;
		str[pos++] = 'C';
	}
	str[pos] = '\0';
}
