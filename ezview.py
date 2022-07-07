from os import stat
from pathlib import Path
from pickletools import uint8
from typing import List
from easygraphics import *
from easygraphics.easygraphics import *
import numpy as np
import math
import copy
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import size
import time


def constrain(val, min_val, max_val):
  return min(max_val, max(min_val, val))

#ala arduino map
def rescale(val_in, in_min, in_max, out_min, out_max):
  return (val_in - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

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
        MouseUtil.instance().setPressEvent(msg.button)
      elif msg.type == MouseMessageType.RELEASE_MESSAGE:
        MouseUtil.instance().setReleaseEvent()
    return MouseUtil.instance().isPressed() 

  @staticmethod
  def waitForMouseRelease():
    old_state = False
    while True:
      if old_state and not View.getMousePressed():
        break
      old_state = View.getMousePressed()
      time.sleep(0.02)
    pass

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
  def draw_rect(_p, width , height, line_width = 0.5,  color = Color.BLACK):
    p = View.__flip(_p)
    set_color(color)
    set_line_width(line_width)
    rect(p.x, p.y, p.x + width, p.y -height)

  @staticmethod
  def draw_filled_rect(_p, width , height, color = Color.BLACK):
    p = View.__flip(_p)
    set_color(color)
    set_fill_color(color)
    fill_rect(p.x, p.y, p.x + width, p.y -height)

  @staticmethod
  def draw_filled_circ(_p, diameter, color = Color.DARK_CYAN):
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
  def plot(data, diameter = 0.1, color = Color.GREEN):
    for p in data:
      View.draw_filled_circ(p, diameter, color)
    return

  def drawText(text, _p):
    p = View.__flip(_p)
    # draw_text(p.x, p.y, text)
    draw_rect_text(p.y, p.y, 1, 1, text)
    return

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
      self.is_pressed = 0
      MouseUtil.__instance = self

  @staticmethod
  def setPressEvent(value = 1):
    # print("setPressed")
    MouseUtil.__instance.is_pressed = value
  
  @staticmethod  
  def setReleaseEvent():
    # print("setReleased")
    MouseUtil.__instance.is_pressed = 0

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

  def draw_vec_trans(self, orig, color = Color.BLACK):
    vec = self + orig
    View().draw_vector(vec, orig, color)
  
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

  def lengthTo(self, rhs):
    tmp_vec: Vector2 = rhs - self
    return tmp_vec.length()

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

  def __eq__(self, rhs: object) -> bool:
    if abs(self.x - rhs.x) < 0.08 and abs(self.y - rhs.y) < 0.08:
      return True
    return False


class Rotation2(object):
  """docstring for Rotation2."""
  def __init__(self, theta):
    super(Rotation2, self).__init__()
    c, s = np.cos(theta), np.sin(theta)
    self.R = np.array(((c, -s), (s, c)))
    self.theta = theta
  
  def dot(self, rhs):
    if isinstance(rhs, Vector2):
      # print("mult")
      # print(rhs.get)
      p = self.R.dot(rhs.get())
    return Vector2(p[0], p[1])
    

class Pose2D(object):
  """docstring for pose2d"""
  def __init__(self, pos: Vector2, rotation: Rotation2):
    self.pos = copy.deepcopy(pos)
    self.orientation = copy.deepcopy(rotation)
  
  def draw(self, color = Color.BLUE):
    #draw circle
    View.draw_filled_circ(self.pos, 0.1, color)
    #draw 
    #create vec to draw
    tmp_vec = copy.deepcopy(self.pos)
    tmp_vec += Vector2(0.2, 0)
    tmp_rot_vec = tmp_vec - self.pos
    tmp_rot_vec = tmp_rot_vec.rotate(self.orientation.theta)
    tmp_rot_vec += self.pos
    # View.draw_filled_circ(tmp_vec, 0.2, Color.RED)
    View.draw_vector(tmp_rot_vec, self.pos, color)



class GridPoint(object):
  def __init__(self, _x:int, _y:int):
    self.x:int = _x
    self.y:int = _y

  def __str__(self):
    return "(" + str(self.x) + ", " + str(self.y) + ")"

  def __repr__(self):
    return "(" + str(self.x) + ", " + str(self.y) + ")"

  def __eq__(self, other):
    if isinstance(other, GridPoint):
      return self.x == other.x and self.y == other.y
    return False

  def __add__(self, other):
    if isinstance(other, GridPoint):
      return GridPoint(self.x + other.x, self.y + other.y)
    return None

  def __sub__(self, other):
    if isinstance(other, GridPoint):
      return GridPoint(self.x - other.x, self.y - other.y)
    return None





class Grid(object):
  def __init__(self, num_cells_edge: int):
    self.num_cells_edge = num_cells_edge
    
    self.__WORLD_WIDTH = 4.0
    self.__WORLD_MIN_WIDTH = -2.0
    self.__WORLD_MAX_WIDTH = 2.0

    self.cell_size: float = self.__WORLD_WIDTH / self.num_cells_edge

    self.origin = Vector2(self.__WORLD_MIN_WIDTH + (self.cell_size * 0.5), self.__WORLD_MIN_WIDTH + (self.cell_size * 0.5)) #origin is pos of cell 0,0 (bottom left)


    self.cells = [0] * (self.num_cells_edge * self.num_cells_edge) #cell container

    pass
  
  def clear(self):
    self.cells = [0] * (self.num_cells_edge * self.num_cells_edge)

  def setValue(self, p, value):
    if type(p) == Vector2:
      self.cells[self.worldTodIdx(p)] = value
    elif type(p) == GridPoint:
      self.cells[self.gridPointToIdx(p)] = value
    else:
      print("error")

  def getValue(self, p):
    if type(p) == Vector2:
      return self.cells[self.worldTodIdx(p)]
    elif type(p) == GridPoint:
      return self.cells[self.gridPointToIdx(p)]
    else:
      print("error")
      return None

  def worldToGridPoint(self, v: Vector2) -> GridPoint:
    #todo check bounds?

    x = (v.x + (self.cell_size * 0.5) - self.origin.x) / self.cell_size
    y = (v.y + (self.cell_size * 0.5) - self.origin.y) / self.cell_size
    # print(f"x: {x}, y: {y}")
    grid_p = GridPoint(int(constrain(x, 0, self.num_cells_edge - 1)),int(constrain(y, 0, self.num_cells_edge - 1)))
    return grid_p

  def gridPointToIdx(self, p: GridPoint) -> int:
    idx: int = p.x + p.y * self.num_cells_edge
    idx = constrain(idx, 0, len(self.cells))
    return idx

  def worldTodIdx(self, v: Vector2) -> int:
    # grid_p = self.worldToGridPoint(v)
    # # print(grid_p)
    # grid_idx = self.gridPointToIdx(grid_p)
    # # print(grid_idx)
    # return grid_idx
    return self.gridPointToIdx(self.worldToGridPoint(v))

  def idxToWorld(self, idx: int) -> Vector2:
    return self.gridPointToWorld(self.idxToGridPos(idx))

  def idxToGridPos(self, idx: int) -> GridPoint:
    x = idx % self.num_cells_edge
    y = idx // self.num_cells_edge
    return GridPoint(x, y)

  def gridPointToWorld(self, p) -> Vector2:
    world_p = Vector2(0.0,0.0)
    world_p.x = self.origin.x + (p.x * self.cell_size)# + (self.cell_size * 0.5)
    world_p.y = self.origin.y + (p.y * self.cell_size)# + (self.cell_size * 0.5)
    return world_p


  def draw_cell(self, cell: GridPoint, color = Color.BLACK):
    vec = self.gridPointToWorld(cell)
    vec.x -= self.cell_size * 0.5
    vec.y -= self.cell_size * 0.5

    View.draw_filled_rect(vec, self.cell_size, self.cell_size, color)


  def draw_grid(self):

    #draw cells first
    for i in range(self.num_cells_edge * self.num_cells_edge):
      if self.cells[i] != 0:
        # col:Color = 0xB8B8B8
        tmp: uint8 = self.cells[i]
        tmp_val : uint8 = round(rescale(tmp, 0, 100, 255, 0))
        # print(type(tmp_val))
        col = tmp_val | (tmp_val << 8) | (tmp_val << 16)
        self.draw_cell(self.idxToGridPos(i), col)

    #draw vertical lines
    for i in range(self.num_cells_edge):
      x = i * self.cell_size + self.__WORLD_MIN_WIDTH
      st =Vector2(  x, self.__WORLD_MIN_WIDTH)
      end = Vector2(x, self.__WORLD_MAX_WIDTH)
      View.draw_line(st, end, Color.BLACK)

    #draw horizontal lines
    for i in range(self.num_cells_edge):
      y = i * self.cell_size + self.__WORLD_MIN_WIDTH
      st = Vector2(self.__WORLD_MIN_WIDTH, y)
      end = Vector2(self.__WORLD_MAX_WIDTH, y)
      View.draw_line(st, end, Color.BLACK)

    pass
    

  # def draw_pos_text(self):
  #   for i in range(self.num_cells_edge * self.num_cells_edge):
  #     p = self.idxToWorld(i)
  #     View.drawText(str(i), p)
  #     # View.draw_text(p, str(i), Color.BLACK)

  def inflate(self, radius):
    old_cells = copy.deepcopy(self.cells)

    # print('create lut')
    #create lut circle around midpoint
    r_pixel: int = round(radius / self.cell_size)
    # print(f"r_pixel: {r_pixel}")
    width_pixel: int = 2 * r_pixel + 1
    # print(f"width_pixel: {width_pixel}")
    lut = [0] * (width_pixel * width_pixel) 
    for i in range(len(lut)):
      x = i % width_pixel
      y = i // width_pixel
      if pow(x - r_pixel, 2) + pow(y - r_pixel, 2) <= pow(r_pixel, 2):
        # print(f"{x}, {y}")
        lut[i] = 100
    
    #apply lut do cells
    for i in range(len(self.cells)):
      if(old_cells[i] != 100): #find obstacles
        continue
      curr_gp = self.idxToGridPos(i)
      # print(f"curr_gp: {curr_gp}")
      #draw circle around current cell based on lut
      idx_lut = 0
      for x in range(curr_gp.x - r_pixel, curr_gp.x + r_pixel + 1):
        for y in range(curr_gp.y - r_pixel, curr_gp.y + r_pixel + 1):
          tmp_x = constrain(x, 0, self.num_cells_edge - 1)
          tmp_y = constrain(y, 0, self.num_cells_edge - 1)
          new_val = lut[idx_lut]
          # print(f"x: {tmp_x}, y: {tmp_y}, val: {new_val}")
          idx_lut += 1
          idx_cells = self.gridPointToIdx(GridPoint(tmp_x, tmp_y))
          # print(f"idx_cells: {idx_cells}")
          if self.cells[idx_cells] < new_val:
            self.cells[idx_cells] = new_val

  def distanceTransform(self, radius):
    old_cells = copy.deepcopy(self.cells)

    # print('create lut')
    #create lut circle around midpoint
    r_pixel: int = round(radius / self.cell_size)
    # print(f"r_pixel: {r_pixel}")
    width_pixel: int = 2 * r_pixel + 1
    # print(f"width_pixel: {width_pixel}")
    lut = [0] * (width_pixel * width_pixel) 
    for i in range(len(lut)):
      x = i % width_pixel
      y = i // width_pixel
      dt_r = math.sqrt(pow(x - r_pixel, 2) + pow(y - r_pixel, 2))
      if round(dt_r) <= r_pixel:
        # print(f"{x}, {y}")
        lut[i] = rescale(dt_r, 0, r_pixel, 100, 0)
    
    #apply lut to cells
    for i in range(len(self.cells)):
      if(old_cells[i] != 100): #find obstacles
        continue
      curr_gp = self.idxToGridPos(i)
      # print(f"curr_gp: {curr_gp}")
      #draw circle around current cell based on lut
      idx_lut = 0
      for x in range(curr_gp.x - r_pixel, curr_gp.x + r_pixel + 1):
        for y in range(curr_gp.y - r_pixel, curr_gp.y + r_pixel + 1):
          tmp_x = constrain(x, 0, self.num_cells_edge - 1)
          tmp_y = constrain(y, 0, self.num_cells_edge - 1)
          new_val = lut[idx_lut]
          # print(f"x: {tmp_x}, y: {tmp_y}, val: {new_val}")
          idx_lut += 1
          idx_cells = self.gridPointToIdx(GridPoint(tmp_x, tmp_y))
          # print(f"idx_cells: {idx_cells}")
          if self.cells[idx_cells] < new_val:
            self.cells[idx_cells] = new_val












class Path2D(object):
  """docstring"""
  def __init__(self, poses: List[Pose2D] = []):
    self.poses = copy.deepcopy(poses)
  def draw(self, color = Color.MAGENTA):
    for i in range(0,len(self.poses)):
      #draw point
      View.draw_filled_circ(self.poses[i].pos, 0.05, color)
      if i >= 1:
        #draw line to last
        View.draw_line(self.poses[i-1].pos, self.poses[i].pos, color, 0.9)

  def drawDetail(self, color = Color.MAGENTA):
    for i in range(0,len(self.poses)):
      #draw point
      self.poses[i].draw(color)
      if i >= 1:
        #draw line to last
        View.draw_line(self.poses[i-1].pos, self.poses[i].pos, color, 0.9)

  def pushBack(self, pose: Pose2D):
    self.poses.append(pose)

  def at(self, idx: int) -> Pose2D:
    return self.poses[idx]
    # pass

  def clear(self):
    self.poses.clear

class PathClicker(object):
  """docstring"""
  def __init__(self) -> None:
      pass
  
  @staticmethod
  def create(color = Color.DARK_CYAN) -> Path2D:
    rdy = False
    path = Path2D()
    while not rdy:
      View.waitForMouseRelease()
      tmp_vec = View.getMouse()
      if len(path.poses) > 1:
        if tmp_vec == path.poses[len(path.poses) -1].pos:
          break
      path.pushBack(Pose2D(tmp_vec, Rotation2(0)))
      path.draw(color)
    return path

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

