import sys
from Others import LCD_drivers
import time


display = LCD_drivers.Lcd()

# python ~/Desktop/Gold/Others/InputNum.py
class _Getch:
    """Gets a single character from standard input.  Does not echo to the
screen."""
    def __init__(self):
        self.impl = _GetchUnix()
    def __call__(self): return self.impl()


class _GetchUnix:
    def __init__(self):
        import tty, sys

    def __call__(self):
        import sys, tty, termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

def Array2string(q):
    pstr=''
    for i in q:
            pstr = pstr + str(i)
    return pstr

def PrintScreen(q,t1,clear=False):

    pstr = Array2string(q)

    if clear == True:
        display.lcd_clear()
 
    display.lcd_display_string('', 1)
    display.lcd_display_string('', 2)
    display.lcd_display_string(t1, 1)
    display.lcd_display_string(pstr, 2)  # Write line of text to first line of display

def ShowValue(CountInput,Count):
    display.lcd_display_string("Input: " + str(CountInput), 1)
    display.lcd_display_string("Counting: "+ str(Count), 2)

def clear():
    print("Cleaning up!")
    display.lcd_clear()

def Keynumber():
    
    queue = []
    pstr=''

    PrintScreen(queue,"Input: ")

    while True:
        
        getch = _Getch()
        ch = getch.impl()
        # print(ord(ch))

        if ord(ch) == 27: # ESC
            display.lcd_clear()
            return -1
        elif ord(ch) == 127 and len(queue) >0: # Del number
            queue.pop()
            PrintScreen(queue,"Input: ",True)

        elif ord(ch) == 13: # Confirm
            for i in range(3): # Blink 3 times
                display.lcd_clear()
                time.sleep(0.2)
                PrintScreen(queue,"Confirm: " + str(pstr))
                time.sleep(0.2)
                return Array2string(queue)

        elif ord(ch)>=48 and ord(ch) <=57:
            queue.append(ch)
            PrintScreen(queue,"Input: ")

        

        
        
    
    
    
 