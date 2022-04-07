import sys
import RPi.GPIO as GPIO
import time

BuzzerPin = 12
GPIO.setmode(GPIO.BCM)
GPIO.setup(BuzzerPin,GPIO.OUT)

buzzer = GPIO.PWM(BuzzerPin, 1000) # Set frequency to 1 Khz
while True:
    buzzer.start(10) # Set dutycycle to 10
    time.sleep(0.3)
    buzzer.stop(10)
    time.sleep(0.3)
 
