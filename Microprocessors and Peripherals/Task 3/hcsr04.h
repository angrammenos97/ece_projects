#ifndef HCSR04_H
#define HCSR04_H

#include "platform.h"

/*! \brief Initialize pin for ultrasonic module.
 *  \param trigger		Pin for trigger.
 *  \param echo   		Pin for echo.
 */
void hcsr04_init(Pin trigger, Pin echo);

/*! \brief Finds the distance from an object
 *  in front of the sensor.
 *  \return The distance in centimeters.
 */
float hcsr04_get_distance(void);

#endif //HCSR04_H
