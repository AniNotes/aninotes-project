import sys
import requests
from bs4 import BeautifulSoup
import urllib.request
from PIL import Image
import os
import cv2
import numpy as np
import math
from sentence_transformers import SentenceTransformer, util
from anytree import Node, RenderTree
from random import sample

#                                                     .  .  .
# Neighborhood of range 1 around a point (*) --->     .  *  .
#                                                     .  .  .

# Returns a list of all continuous groups of points in img with the same characteristic as denoted by color.
#
# For color: 0 = black ; 1 = white ; 2 = nonwhite ; 3 = pseudowhite ; 4 = transparent
def getSetsOfColoredPoints(img, color, pilImg = -1):
    h, w = img.shape[0:2]
    objPtLsts = []
    for y in range(h):
        for x in range(w):
            isTrspnt = -1
            if color not in (0, 1):
                isWhite = True
                for val in img[y, x]:
                    if val < 248:
                        isWhite = False
                if color == 4:
                    isTrspnt = pilImg.getpixel((x, y))[3] == 0
            if (color == 0 and img[y, x] == 0) or (color == 1 and img[y, x] == 255) or (color == 2 and not(isWhite)) or (color == 3 and isWhite) or isTrspnt == True:
                pt = (y, x)
                inNewObj = True
                for lst in objPtLsts:
                    if pt in lst:
                        inNewObj = False
                        break
                if inNewObj:
                    ptSet = getSetOfPointsFromPoint(img, pt, color, pilImg = pilImg)
                    objPtLsts.append(ptSet)
    return objPtLsts

# Returns a black and white image where pixels are colored in if they are distinctly different from backgroundColor
# in img.
def backgroundThreshold(img, bkgrdColor, pts, edgePt):
    h, w = img.shape[0:2]
    thresh = np.zeros((h, w, 3), np.uint8)
    thresh = ~thresh
    gradPts = getSetOfPointsFromPoint(img, edgePt, 5)
    if len(pts) - 5 <= len(gradPts) <= len(pts):
        for y, x in gradPts:
            thresh[y, x] = 0
    else:
        for y, x in pts:
            isBkgrd = True
            for val in range(0, 3):
                if not(img[y, x][val] >= bkgrdColor[val] - 8 and img[y, x][val] <= bkgrdColor[val] + 8):
                    isBkgrd = False
            if not isBkgrd:
                thresh[y, x] = 0
    return thresh

# PRECONDITION: img is thresholded and contains one object.
#
# Returns whether or not the img is "hollow" (i.e. the object within it has an inside.)
def isHollow(img):
    h, w = img.shape[0:2]
    edges = np.zeros((h, w, 3), np.uint8)
    edges = ~edges
    drawnPts = []
    for y in range(h):
        for x in range(w):
            if img[y, x] == 255:
                isEdge = False

                # Checks if this (white) point is on the edge. i.e. at least one point that neighbors it in a range
                # of 1 is black.
                for iterY in range(y - 1, y + 2):
                    for iterX in range(x - 1, x + 2):
                        if not(iterY == y and iterX == x) and iterY >= 0 and iterY < h and iterX >= 0 and iterX < w:
                            if img[iterY, iterX] == 0:
                                isEdge = True

                # If this point is on the edge, draw it to my edges img and record it in drawnPoints.
                if isEdge:
                    drawnPts.append((y, x))
                    edges[y, x] = (0, 0, 0)
            else:

                # The only time a h point is considered on the edge is when it touches the edge of the img.
                if y == 0 or y == h - 1 or x == 0 or x == w - 1:
                    drawnPts.append((y, x))
                    edges[y, x] = (0, 0, 0)

    # Find all of the groups of points in edges that are black.
    ptSets = getSetsOfColoredPoints(edges, 2)

    # Returns whether or not img is hollow, which occurs when the edges img has more than one group of black points,
    # and the points that were drawn to edges (drawnPts).
    return len(ptSets) != 1, drawnPts

# PRECONDITION: img is black and white.
#
# Returns a list of points of the edge of the object given by objPts on img.
def getEdgePoints(img, objPts, removedPts, bndryLyrs):
    h, w = img.shape[0:2]

    # The points on the very edge of objPts.
    edgePts = []

    # The points not on the very edge of objPts.
    newObjPts = []
    for y, x in objPts:
        pt = (y, x)

        # Points on the edge of the bounds of img are automatically considered edge points.
        isEdge = y == 0 or y == h - 1 or x == 0 or x == w - 1
        if isEdge:
            edgePts.append(pt)
        else:

            # Marks the point as an edge point if a white point is a neighbor to it in a range of 1.
            for iterY in range(y - 1, y + 2):
                for iterX in range(x - 1, x + 2):
                    if img[iterY, iterX] == 255:
                        isEdge = True
            if isEdge:
                edgePts.append(pt)
            else:
                newObjPts.append(pt)

    # If I am done removing the outer layers of the object, return relevant information.
    if bndryLyrs == 0:
        return edgePts, removedPts

    # Otherwise, remove the edge points, add them to the removed points, and recursively call this function with
    # boundary layers minus one.
    else:
        for y, x in edgePts:
            removedPts.append((y, x))
            img[y, x] = 255
        return getEdgePoints(img, newObjPts, removedPts, bndryLyrs - 1)

# PRECONDITION: img is black and white.
#
# Fills in all small holes (black or white) with sizes less than maxSize or a smart size if maxSize is equal to -1.
def removeSmallObjects(img, maxSize):
    h, w = img.shape[:2]

    # List of white point groups in img.
    whiteObjPtLsts = []
    for y in range(h):
        for x in range(w):
            if img[y][x] >= 250:
                pt = (y, x)
                inNewObj = True
                for lst in whiteObjPtLsts:
                    if pt in lst:
                        inNewObj = False
                        break
                if inNewObj:
                    ptSet = getSetOfPointsFromPoint(img, pt, 1)
                    whiteObjPtLsts.append(ptSet)

    # List of black point groups in img.
    blackObjPtLsts = []
    blackPtCt = 0
    for y in range(h):
        for x in range(w):
            if img[y][x] <= 5:
                blackPtCt += 1
                pt = (y, x)
                inNewObj = True
                for lst in blackObjPtLsts:
                    if pt in lst:
                        inNewObj = False
                        break
                if inNewObj:
                    ptSet = getSetOfPointsFromPoint(img, pt, 0)
                    blackObjPtLsts.append(ptSet)

    # If maxSize is -1, make it a "smart" size equal to roughly log base 2 of the number of black points in img. The
    # idea behind this choice is that the black points are the main points I care about for this function.
    if maxSize == -1:
        maxSize = blackPtCt.bit_length() - 1

    # Fill in all the small white groups.
    for lst in whiteObjPtLsts:
        if len(lst) < maxSize:
            for pt in lst:
                y, x = pt
                img[y][x] = 0

    # Fill in all the small black groups.
    for lst in blackObjPtLsts:
        if len(lst) < maxSize:
            for pt in lst:
                y, x = pt
                img[y][x] = 255
    return img

# PRECONDITION: img has no transparent points.
#
# Returns a black and white image containing the objects in img.
def getThresholdedImg(img, crntPts, edgePts, thresh):

    # Don't continue if the current points are small in number (extraneous).
    if len(crntPts) <= 5:
        return thresh
    else:
        h, w = img.shape[:2]

        edgeColors = []
        edgeColorsSet = set()
        for y, x in edgePts:
            v1, v2, v3 = img[y, x]
            edgeColors.append((v1, v2, v3))
            edgeColorsSet.add((v1, v2, v3))
        colorMode = 1
        for color in edgeColorsSet:
            if edgeColors.count(color) > colorMode:
                colorMode = edgeColors.count(color)
        bkgrdColor = (255, 255, 255)
        for color in edgeColorsSet:
            if edgeColors.count(color) == colorMode:
                bkgrdColor = color
                break
        edgePt = edgePts[0]

        # currentThresh is an image with points in currentPoints colored black if they have significantly
        # different colors in img than the background color.
        crntThresh = backgroundThreshold(img, bkgrdColor, crntPts, edgePt)
        crntThresh = cv2.cvtColor(crntThresh, cv2.COLOR_BGR2GRAY)

        # Do a smart removal of small objects from currentThresh.
        crntThresh = removeSmallObjects(crntThresh, -1)

        # All of the black groups of points in currentThresh.
        crntObjs = getSetsOfColoredPoints(crntThresh, 0)
        for obj in crntObjs:
            crntObj = np.zeros((h, w, 3), np.uint8)
            crntObj = ~crntObj
            crntObj = cv2.cvtColor(crntObj, cv2.COLOR_BGR2GRAY)

            # Draw object to currentObject.
            for y, x in obj:
                crntObj[y, x] = 0
            hollow, drawnPts = isHollow(crntObj)

            # If object is hollow, draw it exactly to thresh.
            if hollow:
                for y, x in obj:
                    thresh[y, x] = (0, 0, 0)

            # Otherwise, draw its edge to thresh.
            else:
                for y, x in drawnPts:
                    thresh[y, x] = (0, 0, 0)

            # removedPoints are the first five outer layers of object, and currentEdgePoints is the sixth.
            crntEdgePts, removedPts = getEdgePoints(crntObj, obj, [], 6)

            # Remove the first five layers from object.
            for pt in removedPts:
                obj.remove(pt)

            # Draw onto thresh the threshold of all objects present within img. What this does is it gets objects
            # inside of other objects that have distinctly different colors. For an example, look at "set" on
            # wikipedia.
            thresh = getThresholdedImg(img, obj, crntEdgePts, thresh)
        return thresh

# PRECONDITION: img is black and white.
#
# Thins img to specific requirements.
def thinImg(img):
    h, w = img.shape[:2]

    # The coordinates of a point's range 1 neighbors relative to it.
    nbrCoords = ((1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1))
    for y in range(h):
        for x in range(w):
            if img[y][x] <= 5:

                # List of booleans for each neighbor in the range 1 neighborhood around the current point
                # where True means that the neghbor is in img and is black, and False means otherwise.
                mkdNbrs = []
                for coord in nbrCoords:
                    newY = y + coord[0]
                    newX = x + coord[1]
                    if newY >= 0 and newY < h and newX >= 0 and newX < w:
                        mkdNbrs.append(img[newY][newX] <= 5)
                    else:
                        mkdNbrs.append(False)

                # List of consecutive segments of points in the range 1                           o . .
                # neighborhood around the current point. For example,         ----->              o * o
                # the point to the right has two neighbor segments                                . o o
                # (a 'o' means a black point).
                nbrSegments = [[]]
                firstIdxMkd = False
                for idx in range(8):
                    frontSegment = -1
                    frontSegment = nbrSegments[len(nbrSegments) - 1]
                    if mkdNbrs[idx] == False:
                        if frontSegment != []:
                            nbrSegments.append([])
                    else:
                        coord = nbrCoords[idx]
                        newY = y + coord[0]
                        newX = x + coord[1]
                        frontSegment.append((newY, newX))
                        if idx == 0:
                            firstIdxMkd = True
                        if idx == 7 and firstIdxMkd and len(nbrSegments) > 1:
                            firstSegment = nbrSegments.pop(0)
                            for pt in firstSegment:
                                frontSegment.append(pt)
                if [] in nbrSegments:
                    nbrSegments.remove([])

                # If there are no neigbor segments around the current point, then make the current point white. If
                # there is only one segment, and if it contains more than one point, them make the current point
                # white.
                if len(nbrSegments) <= 1:
                    if len(nbrSegments) == 0:
                        img[y][x] = 255
                    else:
                        if len(nbrSegments[0]) > 1:
                            img[y][x] = 255
    return img

# PRECONDITION: img is black and white.
#
# Thickens img by an amount designatied by thickFactor.
def thickenImg(img, thickFactor):
    h, w = img.shape[:2]

    # The result image.
    thick = np.zeros((h, w, 3), np.uint8)
    thick = ~thick

    # Go thorugh all the points and only care about the black points.
    for y in range(h):
        for x in range(w):
            if img[y][x] <= 5:

                # Mark the result image black at the current point.
                thick[y][x] = 0

                # For each value i in the range of [1, rangeFactor], in the result image, mark each point adjacent
                # to the current point i pixels away.
                for i in range(1, thickFactor + 1):
                    if y - i >= 0 and y - i < h and x >= 0 and x < w:
                        thick[y - i][x] = 0
                    if y >= 0 and y < h and x - i >= 0 and x - i < w:
                        thick[y][x - i] = 0
                    if y + i >= 0 and y + i < h and x >= 0 and x < w:
                        thick[y + i][x] = 0
                    if y >= 0 and y < h and x + i >= 0 and x + i < w:
                        thick[y][x + i] = 0
    return thick

# One-stop-shop for image preparation according to everything required for the analysis to work.
def prepForImgAnalysis(img):
    h, w = img.shape[:2]
    thresh = np.zeros((h, w, 3),np.uint8)
    thresh = ~thresh
    pts = []

    # The initial edge points (on the borders of img.)
    edgePts = []
    for y in range(h):
        for x in range(w):
            pts.append((y, x))
            if y == 0 or y == h - 1 or x == 0 or x == w - 1:
                edgePts.append((y, x))
    thresh = getThresholdedImg(img, pts, edgePts, thresh)
    thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
    thick = thickenImg(thresh, 1)
    thick = cv2.cvtColor(thick, cv2.COLOR_BGR2GRAY)
    thick = ~thick
    thin = cv2.ximgproc.thinning(thick)
    thin = ~thin
    final = thinImg(thin)
    return final

def numberOfObjects(img):
    h, w = img.shape[:2]
    objs = []
    objPtLsts = []
    for y in range(h):
        for x in range(w):
            if img[y][x] <= 5:
                pt = (y, x)
                inNewObj = True
                for lst in objPtLsts:
                    if pt in lst:
                        inNewObj = False
                        break
                if inNewObj:
                    ptSet = getSetOfPointsFromPoint(img, pt, 0)
                    if len(ptSet) == 0:
                        emptyPts.append(pt)
                    objPtLsts.append(ptSet)
    return len(objPtLsts)

def resizeImage(imgPath, imgName, outputPath, maxDim):
    img = Image.open(imgPath + imgName)
    w, h = img.size
    bigDim = max(h, w)
    fctr = bigDim/maxDim
    resizedImg = img.resize((math.floor(w/fctr), math.floor(h/fctr)), Image.ANTIALIAS)
    resizedImg.save(outputPath + "R" + str(maxDim) + "-" + imgName, optimize = True, quality = 95)
    cvImg = cv2.imread(outputPath + "R" + str(maxDim) + "-" + imgName)
    return cvImg

def getTransparentBoundaryPoints(cvImg, pilImg):
    h, w = cvImg.shape[:2]
    bndryPots = set()
    ptNbrs = {}
    for y in range(h):
        for x in range(w):
            if pilImg.getpixel((x, y))[3] == 0:
                nbrs = set()
                isBndry = False
                nbrsToChk = []
                for iterY in range(y - 1, y + 2):
                    for iterX in range(x - 1, x + 2):
                        if pilImg.getpixel((iterX, iterY))[3] == 0:
                            nbrsToChk.append((iterY, iterX))
                            nbrs.add((iterY, iterX))
                        else:
                            isBndry = True
                if isBndry:
                    for i in range(3):
                        nextNbrsToChk = []
                        if nbrsToChk:
                            for nbrY, nbrX in nbrsToChk:
                                psblNextNbrs = []
                                iterIsBndry = False
                                for iterY in range(nbrY - 1, nbrY + 2):
                                    for iterX in range(nbrX - 1, nbrX + 2):
                                        if pilImg.getpixel((iterX, iterY))[3] == 0:
                                            psblNextNbrs.append((iterY, iterX))
                                        else:
                                            iterIsBndry = True
                                if iterIsBndry:
                                    nextNbrsToChk += psblNextNbrs
                            nbrsToChk = []
                            for nbr in nextNbrsToChk:
                                nbrs.add(nbr)
                                nbrsToChk.append(nbr)
                if isBndry:
                    bndryPts.add((y, x))
                    ptNbrs[(y, x)] = nbrs
    return bndryPts, ptNbrs

def removeTransparentBorder(cvImg, pilImg):
    h, w = cvImg.shape[0:2]
    trspntPtSets = getSetsOfColoredPoints(cvImg, 4, pilImg = pilImg)
    trspntPts = []
    bkgrdTrspntPts = []
    for ptSet in trspntPtSets:
        trspntPts += ptSet
        isBkgrdSet = False
        for pt in ptSet:
            y, x = pt
            if y == 0 or y == h - 1 or x == 0 or x == w - 1:
                isBkgrdSet = True
                break
        if isBkgrdSet:
            bkgrdTrspntPts += ptSet
    if len(bkgrdTrspntPts) == len(trspntPts):
        for y in range(h):
            for x in range(w):
                if y >= 0 and y < h and x >= 0 and x < w:
                    pxl = pilImg.getpixel((x, y))
                    if pxl[3] != 0:
                        isTrspntBdr = False
                        rngLngth = 4
                        for iterY in range(y - rngLngth, y + rngLngth + 1):
                            for iterX in range(x - rngLngth, x + rngLngth + 1):
                                if not(iterY == y and iterY == x) and iterY >= 0 and iterY < h and iterX >= 0 and iterX < w:
                                    if pilImg.getpixel((iterX, iterY))[3] == 0:
                                        isTrspntBdr = True
                            if isTrspntBdr:
                                break
                        if isTrspntBdr:
                            cvImg[y, x] = (255, 255, 255)
    return cvImg

def opaqueImage(imgPath, imgName, outputPath):
    cvImg = cv2.imread(imgPath + imgName)
    pilImg = Image.open(imgPath + imgName)
    pilImg = pilImg.convert("RGBA")
    h, w = cvImg.shape[0:2]
    trspntPtSets = getSetsOfColoredPoints(cvImg, 4, pilImg = pilImg)
    if len(trspntPtSets) > 0:
        whitePtSets = getSetsOfColoredPoints(cvImg, 3)
        cvImg = removeTransparentBorder(cvImg, pilImg)
        for ptSet in trspntPtSets:
            for y, x in ptSet:
                cvImg[y, x] = (255, 255, 255)
        newWhitePtSets = getSetsOfColoredPoints(cvImg, 3)
        bkgrdWhiteSets = []
        for ptSet in newWhitePtSets:
            ptSet = {*ptSet}
            isBkgrdSet = False
            for pt in ptSet:
                y, x = pt
                if y == 0 or y == h - 1 or x == 0 or x == w - 1:
                    isBackgroundSet = True
                    break
            if isBkgrdSet:
                bkgrdWhiteSets.append(ptSet)
        for ptSet in whitePtSets:
            ptSet = {*ptSet}
            isBkgrdSet = False
            for otherSet in bkgrdWhiteSets:
                otherSet = {*otherSet}
                if ptSet <= otherSet:
                    isBkgrdSet = True
                    break
            if not isBkgrdSet:
                for y, x in ptSet:
                    cvImg[y, x] = (0, 0, 0)
    cv2.imwrite(outputPath + "Opaque" + imgName, cvImg)
    return cvImg

def calculateSlope(line, approx):
    return ["UND", 10000][approx] if line[2] - line[0] == 0 else (line[3] - line[1])/(line[2] - line[0])

def floorPoint(pt):
    return (math.floor(pt[0]), math.floor(pt[1]))

def isNearSubset(A, B, closeBnd): # A is near subset of B
    if len(A) > len(B):
        return False
    else:
        isNearSubset = True
        for APt in A:
            isNearInB = False
            for BPt in B:
                AY, AX = APt
                BY, BX = BPt
                if APt == BPt or (abs(AY - BY) <= closeBnd and abs(AX - BX) <= closeBnd):
                    isNearInB = True
                    break
            if not(isNearInB):
                isNearSubset = False
                break
        return isNearSubset

def calculateYValOnCircle(circle, x, sign):
    a, b, r = circle
    return "none" if (r ** 2) - ((x - a) ** 2) < 0 else sign * math.sqrt((r ** 2) - ((x - a) ** 2)) + b

def removeNearSubsetElements(lst, closeBnd):
    newLst = []
    for itm in lst:
        if itm not in newLst:
            newLst.append(itm)
    lst = newLst
    finalLst = []
    lstCopy = lst.copy()
    for elem in lst:
        otherElems = []
        for otherElem in lstCopy:
            if otherElem != elem:
                for pt in otherElem[1]:
                    if pt not in otherElems:
                        otherElems.append(pt)
        if not isNearSubset(elem[1], otherElems, closeBnd):
            finalLst.append(elem)
        else: # MAYBE REMOVE? IDK... IF ISSUES --> CONSIDER
            lstCopy.remove(elem)
    return finalLst

def hcAccumArray(img, rVals):
    h, w = img.shape[:2]
    accumArr = np.zeros((len(rVals), h, w))
    for i in range(h):
        for j in range(w):
            if img[i][j] != 0:
                for r in range(len(rVals)):
                    rr = rVals[r]
                    hdown = max(0, i - rr)
                    for a in range(hdown, i):
                        b = round(j+math.sqrt(rr*rr - (a - i) * (a - i)))
                        if b>=0 and b<=w-1:
                            accumArr[r][a][b] += 1
                            if 2 * i - a >= 0 and 2 * i - a <= h - 1:
                                accumArr[r][2 * i - a][b] += 1
                        if 2 * j - b >= 0 and 2 * j - b <= w - 1:
                            accumArr[r][a][2 * j - b] += 1
                        if 2 * i - a >= 0 and 2 * i - a <= h - 1 and 2 * j - b >= 0 and 2 * j - b <= w - 1:
                            accumArr[r][2 * i - a][2 * j - b] += 1
    return accumArr

def findCircles(img, accumArr, rVals, houghThresh):
    resLst = []
    hLst = []
    wLst = []
    rLst = []
    resImg = img.copy()
    for r in range(accumArr.shape[0]):
        for h in range(accumArr.shape[1]):
            for w in range(accumArr.shape[2]):
                if accumArr[r][h][w] > houghThresh:
                    tmp = 0
                    for i in range(len(hLst)):
                        if abs(w - wLst[i]) < 10 and abs(h - hLst[i]) < 10:
                            tmp = 1
                            break
                    if tmp == 0:
                        rr = rVals[r]
                        resLst.append((w, h, rr))
                        hLst.append(h)
                        wLst.append(w)
                        rLst.append(rr)
    return resLst

def houghCircles(img):
    rVals = []
    for r in range(30):
        rVals.append(r)
    invImg = ~img
    accumArr = hcAccumArray(invImg, rVals)
    houghThresh = 30
    resLst = findCircles(img, accumArr, rVals, houghThresh)
    return resLst

def findCurves(img):
    h, w = img.shape[:2]
    thick = thickenImg(img, 2)
    thick = cv2.cvtColor(thick, cv2.COLOR_BGR2GRAY)
    crvLst = []
    circles = houghCircles(img)
    if circles is not None:
        for circle in circles:
            upperCrvMks = {}
            lowerCrvMks = {}
            aInt, bInt, rInt = circle[0], circle[1], circle[2]
            for x in range(aInt - rInt, aInt + rInt):
                if x >= 0 and x < width:
                    upperCrvMks[x] = []
                    y1 = math.floor(calculateYValOnCircle(circle, x, 1))
                    y2 = calculateYValOnCircle(circle, x + 1, 1)
                    yTop = math.floor(y2)
                    if y2.is_integer():
                        yTop -= 1
                    if yTop < y1:
                        tmp = yTop
                        yTop = y1
                        y1 = tmp
                    for y in range(y1, yTop + 1):
                        if y >= 0 and y < h:
                            if thick[y][x] <= 5:
                                upperCrvMks[x].append(y)
                    lowerCrvMks[x] = []
                    scndY1 = math.floor(calculateYValOnCircle(circle, x, -1))
                    scndY2 = calculateYValOnCircle(circle, x + 1, -1)
                    yBtm = math.floor(scndY2)
                    if scndY2.is_integer():
                        yBtm -= 1
                    if yBtm < scndY1:
                        tmp = yBtm
                        yBtm = scndY1
                        scndY1 = tmp
                    for y in range(scndY1, yBtm + 1):
                        if y >= 0 and y < h:
                            if thick[y][x] <= 5:
                                lowerCrvMks[x].append(y)
            yBnd = 1
            lastY = -1
            crvBtwnHalves = []
            frontCrv = []
            x = aInt - rInt
            while x in range(aInt - rInt, aInt + rInt) and x >= 0 and x < width:
                if len(upperCrvMks[x]) != 0:
                    crntCrvMks = upperCrvMks[x]
                    lastY = crntCrvMks[0]
                    thisCrv = []
                    nearY = True
                    while x in upperCrvMks and len(crntCrvMks) > 0 and nearY:
                        crntCrvMks.sort()
                        if x >= aInt:
                            crntCrvMks.reverse()
                        for y in crntCrvMks:
                            if abs(y - lastY) > yBnd:
                                nearY = False
                                break
                            thisCrv.append((y, x))
                            lastY = y
                        x += 1
                        if x in upperCrvMks:
                            crntCrvMks = upperCrvMks[x]
                    if frontCrv == []:
                        frontCrv = thisCrv
                    elif x in upperCrvMks:
                        if len(thisCrv) >= 3:
                            isNewCrv = True
                            newCrvLst = []
                            for crv in crvLst:
                                if isNearSubset(crv[1], thisCrv, 4):
                                    continue
                                elif isNearSubset(thisCrv, crv[1], 4):
                                    isNewCrv = False
                                newCrvLst.append(crv)
                            if isNewCrv:
                                newCrvLst.append(((aInt, bInt, rInt), thisCrv))
                            crvLst = newCrvLst
                    if x >= aInt + rInt - 1:
                        crvBtwnHalves = thisCrv
                x += 1
                if frontCrv == []:
                    frontCrv = -1
            x = aInt + rInt - 1
            while x >= aInt - rInt and x >= 0 and x < w:
                if len(lowerCrvMks[x]) != 0:
                    crntCrvMks = lowerCrvMks[x]
                    lastY = crntCrvMks[0]
                    if x == aInt + rInt - 1 and crvBtwnHalves != []:
                        lastY = crvBtwnHalves[len(crvBtwnHalves) - 1][0]
                    thisCurve = crvBtwnHalves
                    crvBtwnHalves = []
                    nearY = True
                    while x in lowerCrvMks and len(crntCrvMks) > 0 and nearY:
                        crntCrvMks.sort()
                        if x >= aInt:
                            crntCrvMks.reverse()
                        for y in crntCrvMks:
                            if abs(y - lastY) > yBnd:
                                nearY = False
                                break
                            thisCrv.append((y, x))
                            lastY = y
                        x -= 1
                        if x in lowerCrvMks:
                            crntCrvMks = lowerCrvMks[x]
                    if x not in lowerCrvMks and frontCrv != -1:
                        thisCrv += frontCrv
                    if len(thisCrv) >= 3:
                        isNewCrv = True
                        newCrvLst = []
                        for crv in crvLst:
                            if isNearSubset(crv[1], thisCrv, 4):
                                continue
                            elif isNearSubset(thisCrv, crv[1], 4):
                                isNewCrv = False
                            newCrvLst.append(crv)
                        if isNewCrv:
                            newCrvLst.append(((aInt, bInt, rInt), thisCrv))
                        crvLst = newCrvLst
                x -= 1
    resLst = removeNearSubsetElements(crvLst, 4)
    return resLst

def houghLines(edgeImg, numRhos = 180, numThetas = 180, tCt = 3):
    edgeH, edgeW = edgeImg.shape[:2]
    d = np.sqrt(np.square(edgeH) + np.square(edgeW))
    dTheta = 180 / numThetas
    dRho = (2 * d) / numRhos
    thetas = np.arange(0, 180, step = dTheta)
    rhos = np.arange(-d, d, step = dRho)
    cosThetas = np.cos(np.deg2rad(thetas))
    sinThetas = np.sin(np.deg2rad(thetas))
    accum = np.zeros((len(rhos), len(rhos)))
    for y in range(edgeH):
        for x in range(edgeW):
            if edgeImg[y][x] != 0:
                edgePt = [y - edgeH / 2, x - edgeW / 2]
                ys, xs = [], []
                for thetaIdx in range(len(thetas)):
                    rho = (edgePt[1] * cosThetas[thetaIdx]) + (edgePt[0] * sinThetas[thetaIdx])
                    theta = thetas[thetaIdx]
                    rhoIdx = np.argmin(np.abs(rhos - rho))
                    accum[rhoIdx][thetaIdx] += 1
                    ys.append(rho)
                    xs.append(theta)
    lines = set()
    for y in range(accum.shape[0]):
        for x in range(accum.shape[1]):
            if accum[y][x] > tCt:
                rho = rhos[y]
                theta = thetas[x]
                a = np.cos(np.deg2rad(theta))
                b = np.sin(np.deg2rad(theta))
                x0 = (a * rho) + edgeW / 2
                y0 = (b * rho) + edgeH / 2
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                line = (x1, y1, x2, y2)
                lines.add(line)
    return lines

def findEdges(img, intrsctMode, imgObjs = -1):
    h, w = img.shape[:2]
    thickFctr = 2 if intrsctMode else 4
    thick = thickenImg(img, thickFctr)
    thick = cv2.cvtColor(thick, cv2.COLOR_BGR2GRAY)
    if imgObjs != -1:
        for obj in imgObjs:
            newObj = set()
            for y, x in obj:
                for iterY in range(y - 1, y + 2):
                    for iterX in range(x - 1, x + 2):
                        if 0 <= iterY < h and 0 <= iterX < w:
                            newObj.add((iterY, iterX))
            imgObjs.remove(obj)
            imgObjs.append(newObj)
    lines = houghLines(~thick)
    edgeLst = []
    for line in lines:
        m = calculateSlope(line, False)
        if m != "UND":
            x1 = line[0]
            y1 = line[1] - calculateYVal(line, 0)
            edgeMks = {}
            for x in range(w):
                edgeMks[x] = []
                yFlr = math.floor(calculateYVal(line, x))
                yNext = calculateYVal(line, x + 1)
                yTop = math.floor(yNext)
                if yNext.is_integer():
                    yTop -= 1
                if yTop < yFlr:
                    tmp = yTop
                    yTop = yFlr
                    yFlr = tmp
                for y in range(yFlr, yTop + 1):
                    if y >= 0 and y < h:
                        if thick[y][x] <= 5:
                            edgeMks[x].append(y)
            lastY = -1
            x = 0
            while x in range(w):
                if len(edgeMks[x]) != 0:
                    crntEdgeMks = edgeMks[x]
                    if m < 0:
                        crntEdgeMks.reverse()
                    lastY = crntEdgeMks[0]
                    yBnd = 1
                    thisEdge = []
                    nearY = True
                    while x in edgeMks and len(crntEdgeMks) > 0 and nearY:
                        crntEdgeMks = edgeMks[x]
                        if m < 0:
                            crntEdgeMks.reverse()
                        for y in crntEdgeMks:
                            if abs(y - lastY) > yBnd:
                                nearY = False
                                break
                            thisEdge.append((y, x))
                            lastY = y
                        x += 1
                    if len(thisEdge) >= 3:
                        isNewEdge = True
                        newEdgeLst = []
                        for edge in edgeLst:
                            if isNearSubset(edge[1], thisEdge, 2 * thickFctr):
                                continue
                            elif isNearSubset(thisEdge, edge[1], 2 * thickFctr):
                                isNewEdge = False
                            newEdgeLst.append(edge)
                        if isNewEdge:
                            if imgObjs != -1:
                                smpl = sample(thisEdge, math.ceil(len(thisEdge)/10))
                                objPtCts = {}
                                for obj in imgObjs:
                                    obj = tuple(obj)
                                    objPtCts[obj] = 0
                                for pt in smpl:
                                    for obj in objPtCts:
                                        obj = tuple(obj)
                                        if pt in obj:
                                            objPtCts[obj] += 1
                                maxCt = 0
                                for obj in objPtCts:
                                    obj = tuple(obj)
                                    ct = objPtCts[obj]
                                    if ct > maxCt:
                                        maxCt = ct
                                edgeObj = -1
                                for obj in objPtCts:
                                    obj = tuple(obj)
                                    if objPtCts[obj] == maxCt:
                                        edgeObj = obj
                                newEdgeLst.append((line, thisEdge, edgeObj))
                            else:
                                newEdgeLst.append((line, thisEdge))
                        edgeLst = newEdgeLst
                x += 1
        else:
            edgeMks = {}
            lineXVal = math.floor(line[0])
            if lineXVal >= 0 and lineXVal < w:
                for y in range(h):
                    if thick[y][lineXVal] <= 5:
                        edgeMks[y] = True
                    else:
                        edgeMks[y] = False
                y = 0
                while y in range(h):
                    if edgeMks[y] == True:
                        thisEdge = []
                        crntEdgeMk = edgeMks[y]
                        while y in edgeMks and crntEdgeMk == True:
                            crntEdgeMk = edgeMks[y]
                            thisEdge.append((y, lineXVal))
                            y += 1
                        if len(thisEdge) >= 3:
                            isNewEdge = True
                            newEdgeLst = []
                            for edge in edgeLst:
                                if isNearSubset(edge[1], thisEdge, 2 * thickFctr):
                                    continue
                                elif isNearSubset(thisEdge, edge[1], 2 * thickFctr):
                                    isNewEdge = False
                                newEdgeLst.append(edge)
                            if isNewEdge:
                                if imgObjs != -1:
                                    smpl = sample(thisEdge, math.ceil(len(thisEdge) / 10))
                                    objPtCts = {}
                                    for obj in imgObjs:
                                        obj = tuple(obj)
                                        objPtCts[obj] = 0
                                    for pt in smpl:
                                        for obj in objPtCts:
                                            obj = tuple(obj)
                                            if pt in obj:
                                                objPtCts[obj] += 1
                                    maxCt = 0
                                    for obj in objPtCts:
                                        obj = tuple(obj)
                                        ct = objPtCts[obj]
                                        if ct > maxCt:
                                            maxCt = ct
                                    edgeObj = -1
                                    for obj in objPtCts:
                                        obj = tuple(obj)
                                        if objPtCts[obj] == maxCt:
                                            edgeObj = obj
                                    newEdgeLst.append((line, thisEdge, edgeObj))
                                else:
                                    newEdgeLst.append((line, thisEdge))
                            edgeLst = newEdgeLst
                    y += 1
    removeFctr = 1 if intrsctMode else 2
    if len(edgeLst) > 60:
        edgeLst.sort(key = lambda edge: len(edge[1]))
        edgeLst = edgeLst[int(len(edgeLst) * (3/4)):]
    resLst = removeNearSubsetElements(edgeLst, thickFctr * removeFctr)
    return resLst

def calculateYVal(line, xVal):
    x1, y1, x2, y2 = line
    return "none" if x2 - x1 == 0 else ((y2 - y1)/(x2 - x1)) * (xVal - x1) + y1

def calculateXVal(line, yVal):
    x1, y1, x2, y2 = line
    return "none" if x2 - x1 == 0 else (yVal - y1)/((y2 - y1)/(x2 - x1)) + x1

def solveQuadratic(a, b, c):
    return "none" if  a == 0 or (b ** 2) - (4 * a * c) < 0 else ( (-b + math.sqrt((b ** 2) - (4 * a * c)))/(2 * a) , (-b - math.sqrt((b ** 2) - (4 * a * c)))/(2 * a) )

def calculateIntersection(struct1, struct2):
    lngth1 = len(struct1)
    lngth2 = len(struct2)
    if lngth1 == lngth2 == 3:
        circle1 = struct1
        circle2 = struct2
        if circle1 == circle2:
            return "none"
        else:
            h1, k1, r1 = circle1
            h2, k2, r2 = circle2
            line = x1 = y1 = x2 = y2 = -1
            if k1 == k2:
                frontCircle = circle1
                behindCircle = circle2
                if h1 < h2:
                    frontCircle = circle2
                    behindCircle = circle1
                firstX = behindCircle[0] + behindCircle[2]
                scndX = frontCircle[0] - frontCircle[2]
                x1 = x2 = (firstX + scndX)/2
                y1 = 0
                y2 = 1
            else:
                m = (-2 * h2 + 2 * h1)/(-2 * k1 + 2 * k2)
                b = (h2 ** 2 - (h1 ** 2) + r1 ** 2 - (r2 ** 2) - (k1 ** 2) + k2 ** 2)/(-2 * k1 + 2 * k2)
                x1 = 0
                y1 = b
                x2 = 1
                y2 = m + b
            line = (x1, y1, x2, y2)
            return calculateIntersection(line, circle1)
    elif lngth1 == lngth2 == 4:
        line1 = struct1
        line2 = struct2
        l1x1, l1y1, l1x2, l1y2 = line1
        l2x1, l2y1, l2x2, l2y2 = line2
        m1 = m2 = 0
        if l1x2 - l1x1 == 0:
            m1 = "UND"
        else:
            m1 = (l1y2 - l1y1)/(l1x2 - l1x1)
        if l2x2 - l2x1 == 0:
            m2 = "UND"
        else:
            m2 = (l2y2 - l2y1)/(l2x2 - l2x1)
        if not(m2 == "UND" and m1 == "UND"):
            if m2 == "UND" or m1 == "UND":
                x = y = 0
                if m1 == "UND":
                    x = l1x1
                    y = m2 * (x - l2x1) + l2y1
                elif m2 == "UND":
                    x = l2x1
                    y = m1 * (x - l1x1) + l1y1
                else:
                    x = (m2 * l2x1 - (m1 * l1x1) - l2y1 + l1y1)/(m2 - m1)
                    y = m2 * (x - l2x1) + l2y1
                return [(y, x)]
            if m2 - m1 != 0:
                x = y = 0
                if m1 == "UND":
                    x = l1x1
                    y = m2 * (x - l2x1) + l2y1
                elif m2 == "UND":
                    x = l2x1
                    y = m1 * (x - l1x1) + l1y1
                else:
                    x = (m2 * l2x1 - (m1 * l1x1) - l2y1 + l1y1)/(m2 - m1)
                    y = m2 * (x - l2x1) + l2y1
                return [(y, x)]
            else:
                return "none"
        else:
            return "none"
    elif lngth1 == 3 and lngth2 == 4 or lngth1 == 4 and lngth2 == 3:
        circle = line = -1
        if lngth1 < lngth2:
            circle = struct1
            line = struct2
        else:
            circle = struct2
            line = struct1
        h, k, r = circle
        x1, y1, x2, y2 = line
        m = calculateSlope(line, False)
        if m != "UND":
            q = calculateIntersection(line, (0, 0, 0, 1))[0][0]
            a = -(m ** 2) - 1
            b = 2 * h + 2 * (k - q) * m
            c = -(-(r ** 2) + h ** 2 + (k - q) ** 2)
            xVals = solveQuadratic(a, b, c)
            if xVals == "none":
                return "none"
            firstX, scndX = xVals
            firstY = calculateYVal(line, firstX)
            scndY = calculateYVal(line, secondX)
            firstPt = (firstY, firstX)
            scndPt = (scndY, scndX)
            return [firstPt, scndPt]
        else:
            x = x1
            firstY = calculateYValOnCircle(circle, x, 1)
            scndY = calculateYValOnCircle(circle, x, -1)
            if firstY == "none":
                return "none"
            firstPt = (firstY, x)
            scndPt = (scndY, x)
            return [firstPt, scndPt]

def findIntersections(img):
    h, w = img.shape[:2]
    crvs = findCurves(img)
    edges = findEdges(img, True)
    structLst = crvs + edges
    finalStructs = removeNearSubsetElements(structLst, 2)
    mdls = set()
    for struct in finalStructs:
        mdls.add(struct[0])
    intrsctngPairs = []
    for mdl in mdls:
        for otherMdl in mdls:
            if mdl != otherMdl:
                intrsctns = calculateIntersection(mdl, otherMdl)
                if intrsctns != "none":
                    intrsctngPair = (mdl, otherMdl)
                    revIntrsctngPair = (otherMdl, mdl)
                    if intrsctngPair not in intrsctngPairs and revIntrsctngPair not in intrsctngPairs:
                        intrsctngPairs.append(intrsctngPair)
    resIntrsctns = set()
    for intrsctngPair in intrsctngPairs:
        mdl1, mdl2 = intrsctngPair
        intrsctns = calculateIntersection(mdl1, mdl2)
        for intrsctn in intrsctns:
            rndIntrsctn = floorPoint(intrsctn)
            rndY, rndX = rndIntrsctn
            potIntrsctns = {}
            rngLngth = 5
            for y in range(rndY - rngLngth, rndY + rngLngth + 1):
                for x in range(rndX - 5, rndX + 6):
                    if y >= 0 and y < h and x >= 0 and x < w:
                        if img[y][x] <= 5:
                            isNewIntrsctn = True
                            for pt in resIntrsctns:
                                chkY, chkX = pt
                                closeBnd = 1
                                if abs(y - chkY) <= closeBnd and abs(x - chkX) <= closeBnd:
                                    isNewIntrsctn = False
                                    break
                            if isNewIntrsctn:
                                potIntrsctns[(y, x)] = abs(y - rndY) + abs(x - rndX)
            minDist = 2 * rngLngth
            for pt in potIntrsctns:
                dist = potIntrsctns[pt]
                if dist < minDist:
                    minDist = dist
            for pt in potIntrsctns:
                dist = potIntrsctns[pt]
                if dist == minDist:
                    resIntrsctns.add(pt)
                    break
    resIntrsctns = [*resIntrsctns]
    return resIntrsctns

def findOpenPoints(img):
    h, w = img.shape[:2]
    openPts = []
    for x in range(w):
        for y in range(h):
            if img[y][x] <= 5:
                nearPts = []
                for xAdd in range(-1, 2):
                    for yAdd in range(-1, 2):
                        newX = x + xAdd
                        newY = y + yAdd
                        if newX < 0 or newX >= w or newY < 0 or newY >= h or (xAdd == yAdd == 0):
                            continue
                        if img[newY][newX] <= 5:
                            nearPts.append((newY, newX))
                if len(nearPts) <= 1:
                    openPts.append((y, x))
    return openPts

def propagationHelper(img, startPt):
    h, w = img.shape[:2]
    ptLst = [startPt]
    startTree = Node(startPt)
    treeStk = [startTree]
    while len(treeStk) > 0:
        crntTree = treeStk.pop(len(treeStk) - 1)
        crntPt = crntTree.name
        y, x = crntPt
        nextSegments = [[]]
        nbrCoords = ((1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1))
        mkdNbrs = []
        for coord in nbrCoords:
            newY = y + coord[0]
            newX = x + coord[1]
            if newY >= 0 and newY < h and newX >= 0 and newX < w:
                mkdNbrs.append(img[newY][newX] <= 5)
            else:
                mkdNbrs.append(False)
        firstIdxMkd = False
        for idx in range(8):
            frontSegment = -1
            frontSegment = nextSegments[len(nextSegments) - 1]
            if mkdNbrs[idx] == False:
                if frontSegment != []:
                    nextSegments.append([])
            else:
                coord = nbrCoords[idx]
                newY = y + coord[0]
                newX = x + coord[1]
                if idx % 2 == 1:
                    frontSegment.insert(0, (newY, newX))
                else:
                    frontSegment.append((newY, newX))
                if idx == 0:
                    firstIdxMkd = True
                if idx == 7 and firstIdxMkd and len(nextSegments) > 1:
                    firstSegment = nextSegments.pop(0)
                    for pt in firstSegment:
                        ptY, ptX = pt
                        relCoords = (ptY - y, ptX - x)
                        ptIdx = nbrCoords.index(relCoords)
                        if ptIdx % 2 == 1:
                            frontSegment.insert(0, pt)
                        else:
                            frontSegment.append(pt)
        nextPts = []
        for segment in nextSegments:
            nextPt = -1
            for pt in segment:
                if pt not in ptLst:
                    nextPt = pt
                    break
            if nextPt != -1:
                ptLst.append(nextPt)
                nextPts.append(nextPt)
        for pt in nextPts:
            nextTree = Node(pt, parent = crntTree)
            treeStk.append(nextTree)
    return (startTree, ptLst)

def getOrderedPointTree(img, intrsctns, openPts): # assume given img has 1 obj
    h, w = img.shape[:2]
    startPt = -1
    if len(intrsctns) == 0:
        if len(openPts) == 0:
            for x in range(w):
                for y in range(h):
                    if img[y][x] <= 5:
                        startPt = (y, x)
                        break
                if startPt != -1:
                    break
        else:
            startPt = openPts[0]
    else:
        startPt = intrsctns[0]
    return propagationHelper(img, startPt)

def getSetOfPointsFromPoint(img, startPt, color, pilImg = -1): # 0 = black ; 1 = white ; 2 = nonwhite ; 3 = pseudowhite ; 4 = transparent # 5 = gradient
    h, w = img.shape[:2]
    resPtSet = {startPt}
    ptQ = [startPt]
    for crntPt in ptQ:#while len(ptQ) > 0:
        #ptQ.pop(0)
        srndPts = []
        up = (crntPt[0] + 1, crntPt[1])
        right = (crntPt[0], crntPt[1] + 1)
        down = (crntPt[0] - 1, crntPt[1])
        left = (crntPt[0], crntPt[1] - 1)
        adjPts  = (up, right, down, left)
        for pt in adjPts:
            y, x = pt
            if (y >= 0 and y < h) and (x >= 0 and x < w):
                if color == 0:
                    if img[y][x] <= 5 and (y, x) not in resPtSet:
                        srndPts.append(pt)
                elif color == 1:
                    if img[y][x] >= 250 and (y, x) not in resPtSet:
                        srndPts.append(pt)
                elif color in (2, 3):
                    isWhite = True
                    for val in img[y, x]:
                        if val < 248:
                            isWhite = False
                    if color == 2 and not(isWhite) and (y, x) not in resPtSet:
                        srndPts.append(pt)
                    elif color == 3 and isWhite and (y, x) not in resPtSet:
                        srndPts.append(pt)
                elif color == 4:
                    if pilImg.getpixel((x, y))[3] == 0 and (y, x) not in resPtSet:
                        srndPts.append(pt)
                elif color == 5:# RGB
                    crntY, crntX = crntPt
                    crntColor = img[crntY, crntX]
                    iterColor = img[y, x]
                    isNearColor = True
                    for val in range(3):
                        if abs(crntColor[val] - iterColor[val]) > 8:
                            isNearColor = False
                    if isNearColor and (y, x) not in resPtSet:
                         srndPts.append(pt)
        topRight = (crntPt[0] + 1, crntPt[1] + 1)
        btmRight = (crntPt[0] - 1, crntPt[1] + 1)
        btmLeft = (crntPt[0] - 1, crntPt[1] - 1)
        topLeft = (crntPt[0] + 1, crntPt[1] - 1)
        diagPts = (topRight, btmRight, btmLeft, topLeft)
        for pt in diagPts:
            y, x = pt
            if (y >= 0 and y < h) and (x >= 0 and x < w):
                if color == 0:
                    if img[y][x] <= 5 and (y, x) not in resPtSet:
                        srndPts.append(pt)
                elif color == 1:
                    if img[y][x] >= 250 and (y, x) not in resPtSet:
                        srndPts.append(pt)
                elif color in (2, 3):
                    isWhite = True
                    for val in img[y, x]:
                        if val < 248:
                            isWhite = False
                    if color == 2 and not(isWhite) and (y, x) not in resPtSet:
                        srndPts.append(pt)
                    elif color == 3 and isWhite and (y, x) not in resPtSet:
                        srndPts.append(pt)
                elif color == 4:
                    if pilImg.getpixel((x, y))[3] == 0 and (y, x) not in resPtSet:
                        srndPts.append(pt)
                elif color == 5:# RGB
                    crntY, crntX = crntPt
                    crntColor = img[crntY, crntX]
                    iterColor = img[y, x]
                    isNearColor = True
                    for val in range(3):
                        if abs(crntColor[val] - iterColor[val]) > 8:
                            isNearColor = False
                    if isNearColor and (y, x) not in resPtSet:
                        srndPts.append(pt)
        for pt in srndPts:
            resPtSet.add(pt)
            ptQ.append(pt)
    return resPtSet

def getSeparateObjects(img):
    h, w = img.shape[:2]
    objs = []
    objPtLsts = []
    for y in range(h):
        for x in range(w):
            if img[y][x] <= 5:
                pt = (y, x)
                inNewObj = True
                for lst in objPtLsts:
                    if pt in lst:
                        inNewObj = False
                        break
                if inNewObj:
                    ptSet = getSetOfPointsFromPoint(img, pt, 0)
                    objPtLsts.append(ptSet)
    for objPtLst in objPtLsts:
        intrsctns = ordPtLst = closed = False
        objImg = np.zeros((h, w, 3), np.uint8)
        objImg = ~objImg
        for pt in objPtLst:
            y, x = pt
            objImg[y][x] = 0
        objImg = cv2.cvtColor(objImg, cv2.COLOR_BGR2GRAY)
        intrsctns = findIntersections(objImg)
        openPts = findOpenPoints(objImg)
        (ordPtTree, ptLst) = getOrderedPointTree(objImg, intrsctns, openPts)
        if len(openPts) == 0:
            clsd = True
        obj = (intrsctns, ordPtTree, ptLst)
        objs.append(obj)
    return objs

def getOrderedListsOfPointsFromTree(tree):
    ordPtLsts = []
    ordPtLst = []
    treePrsr = tree
    while len(treePrsr.children) == 1:
        ordPtLst.append(treePrsr.name)
        treePrsr = treePrsr.children[0]
    ordPtLst.append(treePrsr.name)
    ordPtLsts.append(ordPtLst)
    for subtree in treePrsr.children:
        ordPtLsts += getOrderedListsOfPointsFromTree(subtree)
    return ordPtLsts

def linearApproximation(img):
    h, w = img.shape[:2]
    objs = getSeparateObjects(img)
    linAppImg = np.zeros((h, w, 3), np.uint8)
    linAppImg = ~linAppImg
    drawnObjs = []
    for obj in objs:
        intrsctns, ordPtTree, ptLst = obj
        ordPtLsts = getOrderedListsOfPointsFromTree(ordPtTree)
        newIntrsctns = []
        for intrsctn in intrsctns:
            srndPts = []
            y, x = intrsctn
            for iterY in range(-1, 2):
                for iterX in range(-1, 2):
                    if iterY != 0 or iterX != 0:
                        srndPts.append((y + iterY, x + iterX))
            crntNewIntrsctns = [intrsctn]
            for pt in srndPts:
                isNewIntrsctn = False
                for ptLst in ordPtLsts:
                    if pt in ptLst:
                        isAlone = True
                        for crntNewIntrsctn in crntNewIntrsctns:
                            if crntNewIntrsctn in ptLst:
                                isAlone = False
                                break
                        if isAlone:
                            isNewIntrsctn = True
                        break
                if isNewIntrsctn:
                    crntNewIntrsctns.append(pt)
            newIntrsctns += crntNewIntrsctns
        allPts = []
        for lst in ordPtLsts:
            allPts += lst
        segmentLngth = 10
        psblLngth = math.floor(len(allPts)/8)
        if psblLngth > segmentLngth:
            segmentLngth = psblLngth
        linearApproximationImg, drawnObj = linearApproximationHelper(linAppImg, newIntrsctns, ordPtTree, segmentLngth, [])
        drawnObjs.append(set(drawnObj))
    return linAppImg, drawnObjs

def linearApproximationHelper(linAppImg, intrsctns, ordPtTree, segmentLngth, drawnObj):
    ordPtLst = []
    treePrsr = ordPtTree
    while len(treePrsr.children) == 1:
        ordPtLst.append(treePrsr.name)
        treePrsr = treePrsr.children[0]
    ordPtLst.append(treePrsr.name)
    focPtIdcs = []
    focPtIdcs.append(len(ordPtLst) - 1)
    for intrsctn in intrsctns:
        for idx in range(len(ordPtLst)):
            if ordPtLst[idx] == intrsctn:
                focPtIdcs.append(len(ordPtLst) - 1 - idx)
                break
    ordPtLst.reverse()
    focPtIdcs.sort()
    stopPtIdcs = [0]
    firstPtIdx = scndPtIdx = 0
    focPtEnctd = 0
    if 0 in focPtIdcs:
        focPtEnctd = 1
    while scndPtIdx != len(ordPtLst) - 1:
        nextFocPtIdx = focPtIdcs[focPtEnctd]
        firstPtIdx = scndPtIdx
        scndPtIdx += segmentLngth
        if nextFocPtIdx - scndPtIdx < segmentLngth:
            scndPtIdx = nextFocPtIdx
        if scndPtIdx in focPtIdcs:
            focPtEnctd += 1
        stopPtIdcs.append(scndPtIdx)
    for idx in range(len(stopPtIdcs) - 1):
        firstPt = ordPtLst[stopPtIdcs[idx]]
        scndPt = ordPtLst[stopPtIdcs[idx + 1]]
        linAppImg[firstPt[0]][firstPt[1]] = (0, 0, 0)
        drawnObj.append((firstPt[0], firstPt[1]))
        if scndPt == ordPtLst[len(ordPtLst) - 1]:
            linAppImg[scndPt[0]][scndPt[1]] = (0, 0, 0)
            drawnObj.append((scndPt[0], scndPt[1]))
        line = (firstPt[1], firstPt[0], scndPt[1], scndPt[0])
        if calculateSlope(line, False) == "UND":
            yVal = firstPt[0]
            nextYVal = scndPt[0]
            if nextYVal < yVal:
                yVal = scndPt[0]
                nextYVal = firstPt[0]
            for y in range(yVal, nextYVal):
                linAppImg[y][firstPt[1]] = (0, 0, 0)
                drawnObj.append((y, firstPt[1]))
        else:
            xVal = firstPt[1]
            nextPtXVal = scndPt[1]
            if nextPtXVal < xVal:
                xVal = scndPt[1]
                nextPtXVal = firstPt[1]
            for x in range(xVal, nextPtXVal):
                yVal = calculateYVal(line, x)
                nextYVal = calculateYVal(line, x + 1)
                slopeSign = 1
                if nextYVal < yVal:
                    yVal = calculateYVal(line, x + 1)
                    nextYVal = calculateYVal(line, x)
                    slopeSign = -1
                for y in range(math.floor(yVal), math.ceil(nextYVal)):
                    startX = calculateXVal(line, y)
                    endX = calculateXVal(line, y + 1)
                    if y == math.ceil(nextYVal) - 1 * slopeSign:
                        endX = x + 1 * slopeSign
                    midX = (startX + endX)/2
                    midY = calculateYVal(line, midX)
                    linAppImg[math.floor(midY)][x] = (0, 0, 0)
                    drawnObj.append((math.floor(midY), x))
                linAppImg[math.floor(yVal)][x] = (0, 0, 0)
                drawnObj.append((math.floor(yVal), x))
    for child in treePrsr.children:
        linAppImg, drawnObj = linearApproximationHelper(linAppImg, intrsctns, child, segmentLngth, drawnObj)
    return linAppImg, drawnObj

def main():
    args = sys.argv[1:]
    term = args[0]

if __name__ == "__main__":
    main()
