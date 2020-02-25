import serial

if __name__ == "__main__":
    arduino_data = serial.Serial ('com5',115200) #change comX, Serial.begin(value)

    while (1):
        data = (arduino_data.readline().strip())
        print(data.decode('utf-8'))



    