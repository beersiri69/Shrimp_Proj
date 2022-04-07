import RPi.GPIO as GPIO
import time
LED0_PIN = 17
LED1_PIN = 27
LED2_PIN = 22
btn1 = True;
btn2 = True;
num = 0;
global count;

check_status = 0;

def warmup():
  # run starting mode
  print('complete', check_status); # if check_status == 0? "fail":"complete"
  return check_status;

def LED0_on():
    GPIO.setup(LED0_PIN, GPIO.OUT)
    GPIO.output(LED0_PIN, GPIO.HIGH)
    #time.sleep(1)
    
    #GPIO.output(LED_PIN, GPIO.LOW)
    #GPIO.cleanup()

def LED1_on():
    GPIO.setup(LED1_PIN, GPIO.OUT)
    GPIO.output(LED1_PIN, GPIO.HIGH)

def LED2_on():
    GPIO.setup(LED2_PIN, GPIO.OUT)
    GPIO.output(LED2_PIN, GPIO.HIGH)

def input_number():
    # keypad recieving
    # num = keypad recieving
    # wait for A pressed
    num = input("Enter number: ")
    return num;
    #camera taking ... to some path

def system():
  # assume
  count = 20
  return count;    
      
def cam():
  # taking picture      
  system();

if __name__ == "__main__":
  GPIO.setmode(GPIO.BCM)
  LED0_on();    
  sys = warmup();
  while(sys == 1):
    LED1_on();
    input_number();
    cam()
    btn2_state = GPIO.input(btn2)
    if btn2_state == 'True':
      LED2_on()
      count = system()
      if(count >= num):
        system()

else:
    print("Fail to start system")

