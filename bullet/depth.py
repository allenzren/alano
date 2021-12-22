from bullet import *
import math


def pixelToWorld(depthBuffer, center, dim, params=None):
    point_cloud = np.zeros((0, 3))
    far = params['far']  # 998.6
    near = params['near']  # 0.01

    pixel2xy = np.zeros((dim, dim, 2))
    stepX = 1
    stepY = 1
    for w in range(center - dim // 2, center + dim // 2, stepX):
        for h in range(center - dim // 2, center + dim // 2, stepY):
            rayFrom, rayTo, alpha = getRayFromTo(w, h, params)
            rf = np.array(rayFrom)
            rt = np.array(rayTo)
            vec = rt - rf
            l = np.sqrt(np.dot(vec, vec))
            depthImg = float(depthBuffer[h, w])
            depth = far * near / (far - (far - near) * depthImg)

            depth /= math.cos(alpha)
            newTo = (depth / l) * vec + rf

            pixel2xy[w - center - dim // 2,
                     h - center - dim // 2] = newTo[:2]  #* had to flip
            if newTo[2] > 0.0:
                point_cloud = np.concatenate(
                    (point_cloud, newTo.reshape(1, 3)), axis=0)
    np.savez('pixel2xy20.npz', pixel2xy=pixel2xy)
    # print(pixel2xy)
    return pixel2xy


def getRayFromTo(mouseX, mouseY, params):

    width = params['imgW']
    height = params['imgH']
    # cameraUp = params['cameraUp']
    camForward = params['camForward']
    horizon = params['horizon']
    vertical = params['vertical']
    dist = params['dist']
    camTarget = params['camTarget']

    camPos = [
        camTarget[0] - dist * camForward[0],
        camTarget[1] - dist * camForward[1],
        camTarget[2] - dist * camForward[2]
    ]
    farPlane = 10000
    rayForward = [(camTarget[0] - camPos[0]), (camTarget[1] - camPos[1]),
                  (camTarget[2] - camPos[2])]
    lenFwd = math.sqrt(rayForward[0] * rayForward[0] +
                       rayForward[1] * rayForward[1] +
                       rayForward[2] * rayForward[2])
    invLen = farPlane * 1. / lenFwd
    rayForward = [
        invLen * rayForward[0], invLen * rayForward[1], invLen * rayForward[2]
    ]
    rayFrom = camPos
    oneOverWidth = float(1) / float(width)
    oneOverHeight = float(1) / float(height)

    dHor = [
        horizon[0] * oneOverWidth, horizon[1] * oneOverWidth,
        horizon[2] * oneOverWidth
    ]
    dVer = [
        vertical[0] * oneOverHeight, vertical[1] * oneOverHeight,
        vertical[2] * oneOverHeight
    ]
    # rayToCenter = [
    # rayFrom[0] + rayForward[0], rayFrom[1] + rayForward[1], rayFrom[2] + rayForward[2]
    # ]
    ortho = [
        -0.5 * horizon[0] + 0.5 * vertical[0] + float(mouseX) * dHor[0] -
        float(mouseY) * dVer[0], -0.5 * horizon[1] + 0.5 * vertical[1] +
        float(mouseX) * dHor[1] - float(mouseY) * dVer[1], -0.5 * horizon[2] +
        0.5 * vertical[2] + float(mouseX) * dHor[2] - float(mouseY) * dVer[2]
    ]

    rayTo = [
        rayFrom[0] + rayForward[0] + ortho[0],
        rayFrom[1] + rayForward[1] + ortho[1],
        rayFrom[2] + rayForward[2] + ortho[2]
    ]
    lenOrtho = math.sqrt(ortho[0] * ortho[0] + ortho[1] * ortho[1] +
                         ortho[2] * ortho[2])
    alpha = math.atan(lenOrtho / farPlane)
    return rayFrom, rayTo, alpha
