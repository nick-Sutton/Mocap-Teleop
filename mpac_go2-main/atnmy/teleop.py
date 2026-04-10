from mpac_cmd import *

from pynput import keyboard

keys = set()

def on_press(key):
    keys.add(key)
    send_motion_primitive(keys)

def on_release(key):
    keys.remove(key)
    if key == keyboard.Key.esc:
        # Stop listener
        return False

def send_motion_primitive(keys):
  vx = 0
  vy = 0
  vrz = 0
  if keyboard.KeyCode.from_char('w') in keys:
    vx = 0.2
  if keyboard.KeyCode.from_char('2') in keys:
    vx = 0.3
  if keyboard.KeyCode.from_char('s') in keys:
    vx = -0.2
  if keyboard.KeyCode.from_char('a') in keys:
    vy = 0.15
  if keyboard.KeyCode.from_char('d') in keys:
    vy = -0.15
  if keyboard.KeyCode.from_char('q') in keys:
    vrz = 0.4
  if keyboard.KeyCode.from_char('e') in keys:
    vrz = -0.4
  if (keyboard.KeyCode.from_char('w') in keys or
      keyboard.KeyCode.from_char('2') in keys or  
      keyboard.KeyCode.from_char('s') in keys or  
      keyboard.KeyCode.from_char('a') in keys or  
      keyboard.KeyCode.from_char('d') in keys or  
      keyboard.KeyCode.from_char('q') in keys or  
      keyboard.KeyCode.from_char('e') in keys or
      keyboard.KeyCode.from_char('x') in keys):
    walk_idqp(vx=vx,vy=vy,vrz=vrz)
    return

  if keyboard.KeyCode.from_char('l') in keys:
    lie()
    return
  if keyboard.KeyCode.from_char('j') in keys:
    jump()
    return
  if keyboard.KeyCode.from_char('t') in keys:
    stand_idqp()
    return

# Collect events until released
with keyboard.Listener(
        on_press=on_press,
        on_release=on_release) as listener:
    listener.join()


