import lis3dh, time, math, utime
import machine
from machine import UART, Pin, I2C
import machine, sdcard, uos
import uasyncio

#machine.freq(240000000)
#Initializing SD Card
cs = machine.Pin(13, machine.Pin.OUT)
spi = machine.SPI(1,baudrate=1000000,polarity=0,phase=0,bits=8,
                  firstbit=machine.SPI.MSB,sck=machine.Pin(10),
                  mosi=machine.Pin(11),miso=machine.Pin(12))
sd = sdcard.SDCard(spi, cs)
vfs = uos.VfsFat(sd)
uos.mount(vfs, "/sd")

async def getImuData():
    #Initializing IMU
    i2c = I2C(0,sda=Pin(0), scl=Pin(1)) # Correct I2C pins for TinyPICO
    imu = lis3dh.LIS3DH_I2C(i2c)
    tic = utime.ticks_ms()
    while True:
        x, y, z = [value / lis3dh.STANDARD_GRAVITY for value in imu.acceleration]
        if utime.ticks_ms() - tic >= 100:
            with open("/sd/dataA/accel.txt", "a") as file:
                file.write("{};{};{};{}\r\n".format(tic,x,y,z))
                file.close()
#             print('time = {}, x = {}, y = {}, z = {}'.format(tic,x,y,z))
            while utime.ticks_ms() - tic < 200:
                await uasyncio.sleep_ms(1)
            tic=utime.ticks_ms()
        await uasyncio.sleep_ms(10)
        
def convertToDigree(RawDegrees):
    RawAsFloat = float(RawDegrees)
    firstdigits = int(RawAsFloat/100) #degrees
    nexttwodigits = RawAsFloat - float(firstdigits*100) #minutes
    Converted = float(firstdigits + nexttwodigits/60.0)
    Converted = '{0:.6f}'.format(Converted) # to 6 decimal places
    return float(Converted)

async def getPositionData():
    #Initializing GPS
    gps_module = UART(1, baudrate=9600, tx=Pin(4), rx=Pin(5))
    buff = bytearray(255) #Used to Store NMEA Sentences
    latitude = 999
    longitude = 999
    satellites = 999
    tic = utime.ticks_ms()
    while True:
        gps_module.readline() 
        buff = str(gps_module.readline())
        parts = buff.split(',')
        if (parts[0] == "b'$GPGGA" and len(parts) == 15):
            if(parts[1] and parts[2] and parts[3] and parts[4] and parts[5] and parts[6] and parts[7]):                
                latitude = convertToDigree(parts[2])
                if (parts[3] == 'S'): # parts[3] contain 'N' or 'S'
                    latitude = -1.0*float(latitude)
                longitude = convertToDigree(parts[4])
                if (parts[5] == 'W'): # parts[5] contain 'E' or 'W'
                    longitude = -1.0*float(longitude)
                satellites = parts[7]
                gpsTime = parts[1][0:2] + ":" + parts[1][2:4] + ":" + parts[1][4:6]    
        if utime.ticks_ms() - tic >= 900:
            with open("/sd/dataG/gps.txt", "a") as file2:
                file2.write("{};{};{};{}\r\n".format(tic,latitude,longitude,satellites))
                file2.close()
#                 print('time = {}, latitude = {}, longitude = {}, satellites = {}'.format(tic,latitude,longitude,satellites))
            while utime.ticks_ms() - tic < 1000:
                await uasyncio.sleep_ms(1)
            tic=utime.ticks_ms()            
        await uasyncio.sleep_ms(100)

async def main():
    uasyncio.create_task(getImuData())
    uasyncio.create_task(getPositionData())
    await uasyncio.sleep_ms(3600000)

uasyncio.run(main())