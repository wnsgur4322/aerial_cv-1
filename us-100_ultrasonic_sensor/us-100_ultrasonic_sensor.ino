// the US-100 module has jumper cap on the back.
unsigned int HighLen = 0;
unsigned int LowLen  = 0;
unsigned int Len_mm  = 0;

void setup() {   
  // connect RX (Pin 0 of Arduino digital IO) to Echo/Rx (US-100), TX (Pin 1 of Arduino digital IO) to Trig/Tx (US-100) 
    Serial.begin(9600);                          // set baudrate as 9600bps.
}

void loop() {
    Serial.flush();                               // clear receive buffer of serial port
    Serial.write(0X55);                           // trig US-100 begin to measure the distance
    if(Serial.available() >= 2)                   // when receive 2 bytes 
    {
        HighLen = Serial.read();                   // High byte of distance
        LowLen  = Serial.read();                   // Low byte of distance
        Len_mm  = HighLen*256 + LowLen;            // Calculate the distance
        if((Len_mm > 1) && (Len_mm < 10000))       // normal distance should between 1mm and 10000mm (1mm, 10m)
        {
            Serial.print(Len_mm/10.00, 1);             // output the result to serial monitor
            Serial.println("cm");                  // output the result to serial monitor
        }
    }
    delay(200);                                   // delay to wait result
}
