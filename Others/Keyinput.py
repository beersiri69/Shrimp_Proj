from pynput import keyboard
import LCD_drivers
from time import sleep

global ch

def on_press(key):
    try:
        print('Alphanumeric key pressed: {0} '.format(
            key.char))
        ch = key.char
    except AttributeError:
        print('special key pressed: {0}'.format(
            key))
        ch = key
def on_release(key):
    # print('Key released: {0}'.format(
    #     key))
    if key == keyboard.Key.esc:
        # Stop listener
        return False

# Collect events until released
with keyboard.Listener(
        on_press=on_press,
        on_release=on_release) as listener:
    # listener.join()
    while True:
        on_press(key)




display = LCD_drivers.Lcd()

# # Main body of code
# try:
#     while True:
#         # Remember that your sentences can only be 16 characters long!
#         print("Writing simple string")
#         display.lcd_display_string("Simple string", 1)  # Write line of text to first line of display
#         display.lcd_display_extended_string("Ext. str:{0xEF}{0xF6}{0xA5}{0xDF}{0xA3}", 2)  # Write line of text to second line of display
#         sleep(2) # Give time for the message to be read
# except KeyboardInterrupt:
#     # If there is a KeyboardInterrupt (when you press ctrl+c), exit the program and cleanup
#     print("Cleaning up!")