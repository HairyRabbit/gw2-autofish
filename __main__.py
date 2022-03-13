import win32con, win32gui, win32ui
import cv2 as cv
import numpy as np
import pyautogui
import pydirectinput as gui
import time

WINDOW_NAME = 'Guild Wars 2'

def get_window_rect(name):
  hwnd = win32gui.FindWindow(None, WINDOW_NAME)
  l,t,r,b = win32gui.GetWindowRect(hwnd) # (left,top,right,bottom)
  w = r - l
  h = b - t
  return hwnd, w, h

def cap(hwnd, w, h):
  hwnd_dc = win32gui.GetWindowDC(hwnd)
  mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
  mem_dc = mfc_dc.CreateCompatibleDC()
  bitmap = win32ui.CreateBitmap()
  bitmap.CreateCompatibleBitmap(mfc_dc, w, h)
  mem_dc.SelectObject(bitmap)
  mem_dc.BitBlt((0, 0), (w, h), mfc_dc, (0, 0), win32con.SRCCOPY)

  buf = bitmap.GetBitmapBits(True)
  img = np.frombuffer(buf, dtype="uint8")
  img.shape = (h, w, 4)

  mfc_dc.DeleteDC()
  mem_dc.DeleteDC()
  win32gui.ReleaseDC(hwnd, hwnd_dc)
  win32gui.DeleteObject(bitmap.GetHandle())

  img = cv.cvtColor(np.asarray(img), cv.COLOR_RGBA2RGB)
  # img = cv.cvtColor(np.asarray(img), cv.COLOR_RGBA2GRAY)
  return img

def draw_boom_scope_rect(img, w, h):
  dw = 100
  dh = 200
  tl = (w // 2 - dw // 2, h // 2 - dh // 2)
  br = (w // 2 + dw // 2, h // 2 + dh // 2)
  cv.rectangle(img, tl, br, 255, 1)

TEMPLATE_BOOM_SRC = "C:/Users/yfhj1/Desktop/Robot/fish/boom.png"
def match_boom(img): 
  temp = cv.imread(TEMPLATE_BOOM_SRC, cv.IMREAD_GRAYSCALE)
  res = cv.matchTemplate(img, temp, cv.TM_CCOEFF_NORMED)

  min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
  
  tw,th = temp.shape[::-1]
  tl = max_loc
  br = (tl[0] + tw,tl[1] + th)
  cv.rectangle(img,tl,br,255,2)
  # cv.imshow('img', img)
  # cv.waitKey(delay=1)
  # cv.waitKey(0)
  # print(min_val, max_val, tl, br)
  return max_val > 0.9

def proc_boom(img):
  # img = cv.GaussianBlur(img, 5)
  dw = 100
  dh = 200
  img = img[h // 2 - dh // 2:h // 2 + dh // 2,w // 2 - dw // 2: w // 2 + dw // 2]

  img[:,:,0] = 0
  img[:,:,1] = 0

  _,img = cv.threshold(img, 120,255,cv.THRESH_BINARY)
  img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
  _,img = cv.threshold(img, 0,255,cv.THRESH_BINARY)
  # cv.imshow('img', img)
  # cv.waitKey(delay=1)
  # cv.waitKey(0)
  return img


def find_cursor(img):
  img = img[518:531,571:791]
  img[:,:,0] = 0
  img[:,:,2] = 0
  _,img = cv.threshold(img, 100,255,cv.THRESH_BINARY)
  img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
  _,img = cv.threshold(img, 0,255,cv.THRESH_BINARY)
  return img
 
TEMPLATE_CURSOR_SRC = "C:/Users/yfhj1/Desktop/Robot/fish/cursor.png"
def match_cursor(img): 
  temp = cv.imread(TEMPLATE_CURSOR_SRC, cv.IMREAD_GRAYSCALE)
  res = cv.matchTemplate(img, temp, cv.TM_CCOEFF_NORMED)

  min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
  tw,th = temp.shape[::-1]
  tl = max_loc
  br = (tl[0] + tw,tl[1] + th)
  cv.rectangle(img,tl,br,255,2)
  center = (br[0] - tl[0]) // 2
  # print(min_val, max_val, tl, br, center)
  return center

def find_scope(img):
  img = img[516:517,571:791]
  img[:,:,0] = 0
  img[:,:,1] = 0
  # img = cv.medianBlur(img, 5)
  # k = np.ones((5,5), np.uint8)
  # img = cv.morphologyEx(img, cv.MORPH_OPEN, k)
  _,img = cv.threshold(img, 180, 255, cv.THRESH_BINARY)
  
  img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
  res,img = cv.threshold(img, 0,255,cv.THRESH_BINARY)
  counts,c = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
  # center = counts[0][1][0][0] - counts[0][0][0][0]

  if not counts:
    return None

  # print(counts)
  x,y,w,h = cv.boundingRect(counts[0])
  center = x + w // 2

  # cv.drawContours(img, counts, 0, (0,0,255), 1)
  # cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
  # img = cv.resize(img, (0,0), fx=5, fy=20)
  # cv.imshow('img', img)
  # cv.waitKey(0)
  # cv.waitKey(delay=1)

  # _,_,_,max_value = cv.minMaxLoc(img)

  # for i in img[0]:
  #   print(img[0,i])

  # print(counts, center)
  
  return center, x, x + w

def find_cursor2(img):
  img = img[518:531,571:791]
  img[:,:,0] = 0
  img[:,:,2] = 0
  _,img = cv.threshold(img, 100,255,cv.THRESH_BINARY)
  img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
  _,img = cv.threshold(img, 0,255,cv.THRESH_BINARY)

  counts,c = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
  x,y,w,h = cv.boundingRect(counts[0])
  # cv.drawContours(img, counts, -1, (0,0,255), 3)
  # cv.imshow('img', img)
  # cv.waitKey(0)
  # cv.waitKey(delay=1)

  center = x + w // 2

  # print(center)
  return center, x, x + w

hwnd, w, h = get_window_rect(WINDOW_NAME)
state = 0
dir_state = 0 # 1 left, 2 right 0 init
time.sleep(2)

while True:
  print('state:', state)
  beg = time.clock()
  if state == 1:
    img = cap(hwnd, w, h)
    img = proc_boom(img)
    ma = match_boom(img)
    print("state1", ma)
    if ma:
      gui.press("1")
      time.sleep(0.5)
      # break
      state = 2
    time.sleep(1)
  elif state == 2:
    img = cap(hwnd, w, h)
    scope = find_scope(img)
    if scope == None:
      time.sleep(5)
      state = 0
    else:
      center_scope, left_scope, right_scope = scope
      center_cursor, left_cursor, right_cursor = find_cursor2(img)
      print(center_scope,center_cursor,dir_state)

      # to left
      if left_scope > right_cursor:
        if dir_state == 0:
          gui.keyDown("2")
          dir_state = 1
        elif dir_state == 2:
          gui.keyUp("3")
          gui.keyDown("2")
          dir_state = 1
      # to right
      elif right_scope < left_cursor:
        if dir_state == 0:
          gui.keyDown("3")
          dir_state = 2
        elif dir_state == 1:
          gui.keyUp("2")
          gui.keyDown("3")
          dir_state = 2
      else:
        if center_scope >= center_cursor:
          if dir_state == 0:
            gui.keyDown("2")
            dir_state = 1
          elif dir_state == 2:
            gui.keyUp("3")
            gui.keyDown("2")
            dir_state = 1
        else:
          if dir_state == 0:
            gui.keyDown("3")
            dir_state = 2
          elif dir_state == 1:
            gui.keyUp("2")
            gui.keyDown("3")
            dir_state = 2

      time.sleep(0.033)
  elif state == 0:
    gui.press("1")
    time.sleep(1)
    state = 1
  end = time.clock()
  print('cost', end - beg)



# img = cv.imread(cv.samples.findFile("C:/Users/yfhj1/Desktop/Robot/fish/s2.png"), cv.IMREAD_COLOR)
# img = proc_boom(img)
# ma = match_boom(img)
# print("state1", ma)

# cv.rectangle(
#   img,
#   (571,517),
#   (791,532),
#   255,
#   2
# )

# center_scope = find_scope(img)
# img = find_cursor(img)
# center_cursor = find_cursor2(img)

# print(center_scope,center_cursor)

# img = cv.resize(img, (0,0), fx=5, fy=10)
# cv.imshow('img', img)
# cv.waitKey(0)
# cv.destroyAllWindows()

# cv.imwrite("C:/Users/yfhj1/Desktop/Robot/fish/s2c.png", img)