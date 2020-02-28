#Written by Junhyeok Jeong

#------------------------------------------------------------------------------

#  LIDARLite Arduino Library
#  v3HP/v3HP_I2C

#  This example shows methods for running the LIDAR-Lite v3 HP in various
#  modes of operation. To exercise the examples open a serial terminal
#  program (or the Serial Monitor in the Arduino IDE) and send ASCII
#  characters to trigger the commands. See "loop" function for details.

#  Connections:
#  LIDAR-Lite 5 Vdc (red) to Arduino 5v
#  LIDAR-Lite I2C SCL (green) to Arduino SCL
#  LIDAR-Lite I2C SDA (blue) to Arduino SDA
#  LIDAR-Lite Ground (black) to Arduino GND

#  (Capacitor recommended to mitigate inrush current when device is enabled)
#  680uF capacitor (+) to Arduino 5v
#  680uF capacitor (-) to Arduino GND

#  See the Operation Manual for wiring diagrams and more information:
#  http://static.garmin.com/pumac/LIDAR_Lite_v3HP_Operation_Manual_and_Technical_Specifications.pdf

#------------------------------------------------------------------------------

import serial
import syslog
import time

if __name__ == "__main__":
    arduino_data = serial.Serial ('/dev/ttyACM0',115200) #change comX, Serial.begin(value)
    time.sleep(3)
    arduino_data.flush()
    #arduino_data.write('s'.encode())     #'s', read range once
    arduino_data.write('c'.encode())     #'c', read range continuously
    #arduino_data.write('t'.encode())     #'t', timed measurement
    #arduino_data.write('.'.encode())     #'.', stop measurement
    #arduino_data.write('d'.encode())     #'d', dump corrleation record

    while (1):
        data = (arduino_data.readline().strip())
        print(data.decode('utf-8'))



    
