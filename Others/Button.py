from pickle import FALSE
import RPi.GPIO as GPIO
import time
#Set warnings off (optional)
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

#Set Button and LED pins
Button1 = 12

#Setup Button and LED
GPIO.setup(Button1,GPIO.IN,pull_up_down=GPIO.PUD_UP)


def Waitbtn():
    Chstate =[False]
    while True: # Run forever
        if GPIO.input(Button1) == GPIO.LOW and Chstate[0] == False:
            Chstate[0] = True
            print("Button1 was pushed!")
        if GPIO.input(Button1) == GPIO.HIGH and Chstate[0] == True:
            Chstate[0] = False
            print("Button1 was release!")
        time.sleep(0.25)
        return Chstate[0]


