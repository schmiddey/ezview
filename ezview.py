from os import stat
from easygraphics import *
from easygraphics.easygraphics import *
import numpy as np
import math
import copy
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import size


def constrain(val, min_val, max_val):
  return min(max_val, max(min_val, val))

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


  #
  @staticmethod
  def __flip(v):
    ret = copy.deepcopy(v)
    ret.y *= -1
    return ret


  # @staticmethod
  # def set_color(color):
  #   set_color(color)
  @staticmethod
  def getMouse():
    m_x, m_y = get_cursor_pos()
    # if m_x > 800 or m_y > 800:
    m_x = constrain(m_x,0,800);
    m_y = constrain(m_y,0,800);
    #   return None
    return View.__flip(Vector2(m_x * 4/800 - 2, m_y * 4/800 - 2 ))

  @staticmethod
  def getMousePressed():
    if has_mouse_msg():
      msg = get_mouse_msg()
      # print(msg.button) #todo 
      if msg.type == MouseMessageType.PRESS_MESSAGE:
        MouseUtil.instance().setPressEvent()
      elif msg.type == MouseMessageType.RELEASE_MESSAGE:
        MouseUtil.instance().setReleaseEvent()
    return MouseUtil.instance().isPressed() 

  @staticmethod
  def isOnPlane(v):
    if v.x > -2 and v.x < 2 and v.y > -2 and v.y < 2:
      return True
    return False


  @staticmethod
  def clear():
    fill_image(Color.WHITE)

  @staticmethod
  def draw_line(_p1, _p2, color = Color.BLACK, width = 0.5):
    set_color(color)
    set_line_width(width)
    p1 = View.__flip(_p1);
    p2 = View.__flip(_p2)
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
  def draw_vector(_head, _tail, color = Color.BLACK):
    set_color(color)
    set_line_width(0.5)
    head = View.__flip(_head)
    tail = View.__flip(_tail)
    #draw body
    line(tail.x, tail.y , head.x, head.y)
    #draw head(arrow style)
    tmp_vec = head - (head - tail) * View.vec_head_length_fac
    tmp_vec2 = head - tmp_vec;
    tmp_r = tmp_vec + Vector2(tmp_vec2.y, -tmp_vec2.x)
    tmp_l = tmp_vec + Vector2(-tmp_vec2.y, tmp_vec2.x)
    line(head.x, head.y, tmp_r.x, tmp_r.y )
    line(head.x, head.y, tmp_l.x, tmp_l.y )

  @staticmethod
  def draw_cross(p, color = Color.LIGHT_RED):
    # p = View.__flip(_p)
    set_color(color)
    # draw 2 lines
    set_line_width(2)
    tl = Vector2(p.x - View.cross_offset, (p.y + View.cross_offset))
    tr = Vector2(p.x + View.cross_offset, (p.y + View.cross_offset))
    bl = Vector2(p.x - View.cross_offset, (p.y - View.cross_offset))
    br = Vector2(p.x + View.cross_offset, (p.y - View.cross_offset))
    View.draw_line(tl, br, color)
    View.draw_line(tr, bl, color)
    set_line_width(0.5)

  @staticmethod
  def draw_filled_circ(_p, diameter, color = Color.DARK_CYAN):
    # print("hans")
    p = View.__flip(_p)
    set_color(color)
    set_fill_color(color)
    #note: draw circle does not work with float values :( but draw_chort does
    draw_chord(p.x, p.y, -180, 180, diameter/2, diameter/2)

  @staticmethod
  def draw_circ(_p, diameter, color = Color.BLACK):
    p = View.__flip(_p)
    set_color(color)
    draw_arc(p.x, p.y, -180, 180, diameter/2, diameter/2)

  @staticmethod
  def saveWindowAsImage(filename):
    save_image(filename)

class MouseUtil(object):
  __instance = None

  @staticmethod
  def instance():
    if(MouseUtil.__instance == None):
      MouseUtil()
    return MouseUtil.__instance

  def __init__(self):
    if MouseUtil.__instance != None:
      raise Exception("Only one instance is allowed")
    else:
      self.is_pressed = False
      MouseUtil.__instance = self

  @staticmethod
  def setPressEvent():
    # print("setPressed")
    MouseUtil.__instance.is_pressed = True
  
  @staticmethod  
  def setReleaseEvent():
    # print("setReleased")
    MouseUtil.__instance.is_pressed = False

  @staticmethod
  def isPressed():
    return MouseUtil.__instance.is_pressed




    


class Vector2(object):
  """docstring for Vector2."""
  def __init__(self, x = 0, y = 0):
    # super(Vector, self).__init__()
    self.v = np.array([x,y])
  
  # @classmethod
  # def from_vec2(self, orig):
  #   self = Vector2(orig.x, orig.y)

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


  def get(self):
    return self.v
  # @classmethod
  # def from_vec
  
  # def copy(self):
  #   return Vector2(self.x, self.y)

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
    else:
      self.v = self.v / norm

  def normalized(self):
    ret = copy.deepcopy(self)
    ret.normalize()
    return ret

  def length(self):
    return np.linalg.norm(self.v)

  def rotate(self, angle):
    rot = Rotation2(angle)
    tmp = rot.dot(self)
    self.x = tmp.x
    self.y = tmp.y
    return self

  def angle(self):
    theta = math.atan2(self.y, self.x)
    return theta

  def angleTo(self, rhs):
    #return math.acos(self.dotScalar(rhs) / (self.length() * rhs.length()))
    #angle = math.atan2(self.y, self.x) - math.atan2(rhs.y, rhs.x)  #https://stackoverflow.com/a/21484228
    dot = self.dotScalar(rhs)
    det = self.x * rhs.y - self.y * rhs.x
    angle = math.atan2(det, dot)  #https://stackoverflow.com/a/16544330
    if angle > math.pi:
      angle -= math.pi * 2
    elif angle <= math.pi * -1:
      angle += math.pi * 2
    return angle
  

  def dotScalar(self, rhs):
    return self.v.dot(rhs.get())

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
    return self

  def __sub__(self, rhs):
    # if isinstance(rhs, Vector2):
      return Vector2(self.v[0] - rhs.v[0], self.v[1] - rhs.v[1])

  def __mul__(self, rhs):
    # if isinstance(rhs, Vector2):
      return Vector2(self.v[0] * rhs, self.v[1] * rhs)
  
  def __imul__(self, rhs):
    self = Vector2(self.v[0] * rhs, self.v[1] * rhs)
    return self

  def __str__(self):
    return "({}, {})".format(self.x, self.y)


class Rotation2(object):
  """docstring for Rotation2."""
  def __init__(self, theta):
    super(Rotation2, self).__init__()
    c, s = np.cos(theta), np.sin(theta)
    self.R = np.array(((c, -s), (s, c)))
  
  def dot(self, rhs):
    if isinstance(rhs, Vector2):
      # print("mult")
      # print(rhs.get)
      p = self.R.dot(rhs.get())
    return Vector2(p[0], p[1])
    



class Vehicle(object):
  """docstring for Vehicle."""
  def __init__(self, mode = "diff", pos = Vector2(0,0), orientation_theta = 0 , size = 0.2): #todo what orientation
    super(Vehicle, self).__init__()
    
    self.__set_geometry(pos, orientation_theta)

    self.mode = mode
    self.size = size
    self.max_vel_lin = 1.0
    self.max_vel_ang = 1.0

  def __set_geometry(self, pos = Vector2(), orientation_theta = 0):
    self.pos = pos
    self.orientation_theta = orientation_theta

    self.velocity_lin = Vector2(0,0)
    self.velocity_ang = 0
    
    self.orientation_vec = Vector2(1,0)
    self.orientation_vec = Rotation2(self.orientation_theta).dot(self.orientation_vec)

    self.orientation_vec.normalize()


  def draw(self):
    View.draw_filled_circ(self.pos, self.size)
    vec_ori = self.orientation_vec
    vec_ori *= self.size * 0.5 #scale to agent size
    View.draw_line(self.pos + vec_ori, self.pos, Color.BLACK, 1)
    
  def teleport(self, pos = Vector2(0), orientation_theta = 0):
    self.__set_geometry(pos, orientation_theta)
    

  def setVelocity(self, linear = Vector2(0,0), angular = 0):
    self.velocity_lin = linear
    self.velocity_lin.x = self.velocity_lin.x
    if self.mode == "diff":
      # print("mode diff")
      self.velocity_lin.y = 0
    else:
      # print("omni")
      self.velocity_lin.y = self.velocity_lin.y

    #constrain lin vel
    if self.velocity_lin.length() > self.max_vel_lin:
      self.velocity_lin.normalize()
      self.velocity_lin *= self.max_vel_lin
   
    self.velocity_ang = angular
    self.velocity_ang = constrain(self.velocity_ang, -1 * self.max_vel_ang, self.max_vel_ang)


  def tick(self):
    tmp_vel = copy.deepcopy(self.velocity_lin)

    tmp_vel = Rotation2(self.orientation_theta).dot(tmp_vel)

    self.pos += tmp_vel

    self.pos.x = constrain(self.pos.x, -2, 2)
    self.pos.y = constrain(self.pos.y, -2, 2)

    self.orientation_theta += self.velocity_ang
    self.orientation_vec = Vector2(1,0)
    self.orientation_vec = Rotation2(self.orientation_theta).dot(self.orientation_vec)

