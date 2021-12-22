from bullet import *
from geometry.transform import quat2rot


def plot_frame_pb(pos, orn=np.array([0., 0., 0., 1.]), w_first=False):
    rot = quat2rot(orn, w_first)
    endPos = pos + 0.1 * rot[:, 0]
    p.addUserDebugLine(pos, endPos, lineColorRGB=[1, 0, 0], lineWidth=5)
    endPos = pos + 0.1 * rot[:, 1]
    p.addUserDebugLine(pos, endPos, lineColorRGB=[0, 1, 0], lineWidth=5)
    endPos = pos + 0.1 * rot[:, 2]
    p.addUserDebugLine(pos, endPos, lineColorRGB=[0, 0, 1], lineWidth=5)


def plot_line_pb(p1, p2, lineColorRGB=[1, 0, 0], lineWidth=5):
    p.addUserDebugLine(p1, p2, lineColorRGB=lineColorRGB, lineWidth=lineWidth)
