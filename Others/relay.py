from lzma import CHECK_SHA256
from socketserver import ThreadingUDPServer
import RPi.GPIO as GPIO
import time
GPIO.setmode(GPIO.BCM) # GPIO Numbers instead of board numbers
GPIO.setwarnings(False)

class Relay:
    def __init__(self):
        self.Channel1 = 17
        self.Channel2 = 27    
        self.Channel3 = 22
        GPIO.setup(self.Channel1, GPIO.OUT) # GPIO Assign mode
        GPIO.setup(self.Channel2, GPIO.OUT)
        GPIO.setup(self.Channel3, GPIO.OUT)

    def Ch1(self,Status):
        if Status == True:
            GPIO.output(self.Channel1, GPIO.HIGH)
        elif Status == False:
            GPIO.output(self.Channel1, GPIO.LOW)

    def Ch2(self,Status):
        if Status == True:
            GPIO.output(self.Channel2, GPIO.HIGH)
        elif Status == False:
            GPIO.output(self.Channel2, GPIO.LOW)

    def Ch3(self,Status):
        if Status == True:
            GPIO.output(self.Channel3, GPIO.HIGH)
        elif Status == False:
            GPIO.output(self.Channel3, GPIO.LOW)

    def blynk(LED,enter):
        while(enter == False):
            LED(True)
            time.sleep(1)
            LED(False)

R = Relay()

# R.Ch1(True)
# time.sleep(1)
# R.Ch1(False)

# R.Ch2(True)
# time.sleep(1)
# R.Ch2(False)

# R.Ch3(True)
# time.sleep(1)
# R.Ch3(False)
