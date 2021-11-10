from os import stat
from easygraphics import *
from easygraphics.easygraphics import *
import numpy as np
import math
import copy
import matplotlib.pyplot as plt

class View:
  """docstring for View."""
  vec_head_length_fac = 0.02
  cross_offset        = 0.02

  # def __init__(self):
    # pass
  @staticmethod
  def init(width=800):
    init_graph(width, width)
    set_window(-2, -2, 4, 4) #set origin to center
    # self.width = width

  @staticmethod
  def run(main_func):
    easy_run(main_func)

  @staticmethod
  def wait_close():
    pause()
    close_graph()

  @staticmethod
  def pause():
    pause()

  @staticmethod
  def is_run():
    return is_run()

  # @staticmethod
  # def set_color(color):
  #   set_color(color)
  @staticmethod
  def getMouse():
    m_x, m_y = get_cursor_pos()
    # if m_x > 800 or m_y > 800:
    #   return None
    return Vector2(m_x * 4/800 - 2, m_y * 4/800 - 2 )

  @staticmethod
  def getMouseLeftClicked():
    if has_mouse_msg():
      msg = get_mouse_msg()
      if msg.type == MouseMessageType.PRESS_MESSAGE:
        print("press")
        return True
    return False

  @staticmethod
  def isOnPlane(v):
    if v.x > -2 and v.x < 2 and v.y > -2 and v.y < 2:
      return True
    return False


  @staticmethod
  def clear():
    fill_image(Color.WHITE)

  @staticmethod
  def draw_line(p1, p2, color = Color.BLACK):
    set_color(color)
    line(p1.x, p1.y, p2.x, p2.y)

  @staticmethod
  def draw_coord_cross(color = Color.BLACK):
    set_color(color)
    set_line_width(0.25)
    set_line_style(LineStyle.DASH_LINE)
    line(2, 0, -2, 0)
    line(0, 2, 0, -2)
    set_line_style(LineStyle.SOLID_LINE)
    set_line_width(0.5)

  @staticmethod
  def draw_vector(head, tail, color = Color.BLACK):
    set_color(color)
    set_line_width(0.5)
    #draw body
    line(tail.x, tail.y * -1, head.x, head.y * -1)
    #draw head(arrow style)
    tmp_vec = head - (head - tail) * View.vec_head_length_fac
    tmp_vec2 = head - tmp_vec;
    tmp_r = tmp_vec + Vector2(tmp_vec2.y, -tmp_vec2.x)
    tmp_l = tmp_vec + Vector2(-tmp_vec2.y, tmp_vec2.x)
    line(head.x, head.y * -1, tmp_r.x, tmp_r.y * -1)
    line(head.x, head.y * -1, tmp_l.x, tmp_l.y * -1)

  @staticmethod
  def draw_cross(p, color = Color.LIGHT_RED):
    set_color(color)
    # draw 2 lines
    set_line_width(2)
    tl = Vector2(p.x - View.cross_offset, -1 * (p.y + View.cross_offset))
    tr = Vector2(p.x + View.cross_offset, -1 * (p.y + View.cross_offset))
    bl = Vector2(p.x - View.cross_offset, -1 * (p.y - View.cross_offset))
    br = Vector2(p.x + View.cross_offset, -1 * (p.y - View.cross_offset))
    View.draw_line(tl, br, color)
    View.draw_line(tr, bl, color)
    set_line_width(0.5)






class Vector2:
  """docstring for Vector2."""
  def __init__(self, x = 0, y = 0):
    # super(Vector, self).__init__()
    self.v = np.array([x,y])
  
  @classmethod
  def from_vec2(self, orig):
    self = Vector2(orig.x, orig.y)

  @property
  def x(self):
    return self.v[0]
  
  @property
  def y(self):
    return self.v[1]

  @x.setter
  def x(self, value):
    self.v[0] = value

  @y.setter
  def y(self, value):
    self.v[1] = value
  # def get(self):
  #   return self.v
  # @classmethod
  # def from_vec

  def draw(self, color = Color.LIGHT_RED):
    View.draw_cross(self, color)
    # View.draw_circ(self, 0.1)

  def draw_vec(self, color = Color.BLACK):
    View().draw_vector(self, Vector2(0,0), color)

  def draw_vec_orig(self, orig, color = Color.BLACK):
    View().draw_vector(self, orig, color)

  def draw_vec_orig_reverse(self, orig, color = Color.BLACK):
    View().draw_vector(orig, self, color)
  
  def normalize(self):
    norm = np.linalg.norm(self.v)
    if norm == 0:
      self.v = np.array([0,0])
    self.v = self.v / norm

  def __add__(self, rhs):
    if isinstance(rhs, Vector2):
      return Vector2(self.v[0] + rhs.v[0], self.v[1] + rhs.v[1])
    else:
      return Vector2((self.v[0] + rhs, self.v[1] + rhs))
  
  def __iadd__(self, rhs):
    if isinstance(rhs, Vector2):
      self = Vector2(self.v[0] + rhs.v[0], self.v[1] + rhs.v[1])
    else:
      self = Vector2((self.v[0] + rhs, self.v[1] + rhs))

  def __sub__(self, rhs):
    # if isinstance(rhs, Vector2):
      return Vector2(self.v[0] - rhs.v[0], self.v[1] - rhs.v[1])

  def __mul__(self, rhs):
    # if isinstance(rhs, Vector2):
      return Vector2(self.v[0] * rhs, self.v[1] * rhs)
  
  # def 

  def __str__(self):
    return "({}, {})".format(self.x, self.y)