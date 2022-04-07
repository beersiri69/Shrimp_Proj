# from Yolo_module.fdetect import Detect
#from asyncio.windows_events import NULL
from ast import In
from Others import Cam
from Others import InputNum
from Yolo_module import Newtestmodule as Y
from Others import relay as r
from Others import Button as Btn

import shutil
import time
import cv2
import RPi.GPIO as GPIO

CamObj = Cam.Loadcam()
R = r.Relay()
LED1 = R.Ch1(False)
LED2 = R.Ch2(False)
LED3 = R.Ch3(False)


def clear(x):
    x=0
    return x
# CamObj.ShowVideo()

print("Start Program")
R.Ch1(True)
try :
    while(True):
        Count=0
        x=0
        print("Wait for input")
        CountInput = InputNum.Keynumber()
        InputNum.ShowValue(CountInput,0)
        print("Input is {CountInput}")
        if CountInput == -1 : break
        R.Ch3(False) 
        while(Count < int(CountInput)):  # Snap shot loop
                print("Push button for taking picture")
                # x = input(f"z for snapshot ({x}): ")
                R.Ch2(True)
                status = Btn.Waitbtn()
                if(status):
                    R.Ch2(False)
                    #print("hi btn")
                # if x == 'z':
                #     x = 'a'
                    CamObj.Snap()
                    
                    # CamObj.Imgshow()
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    #R.blynk(LED3,False)
                    rcount = Y.mdetect()
                    #
                    # +   if(int(rcount) > 0):
                    #     R.blynk(LED3,False)
                    #     R.Ch3(True)
                    print(">>>>"+rcount[0])
                    Count = Count + int(rcount[0])
                    print(f"Count :{Count}")
                    print(f"CountInput :{CountInput}")
                    InputNum.ShowValue(CountInput,Count)

        R.Ch3(True)  
        time.sleep(3)
        InputNum.clear()    
except KeyboardInterrupt:    
    CamObj.Endcam()
    GPIO.cleanup()
finally:
    CamObj.Endcam()
    GPIO.cleanup()