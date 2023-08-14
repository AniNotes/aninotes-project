import sys; args = sys.argv[1:]
import os
import requests
from bs4 import BeautifulSoup
import urllib.request
from sentence_transformers import SentenceTransformer, util
import cv2
import math
import json

sys.path.append(os.path.abspath("AninotesLibs/FlatCVLib/src/FlatCV_aninotes"))
import FlatCV as fcv

def formatText(text):
    fText = ""
    skip = False
    for i, ch in enumerate(text):
        if skip:
            skip = False
            continue
        if ch == " ":
            skip = True
            fText += line[i + 1].upper()
            continue
        fText += ch
    fText = fText[0].upper() + fText[1:]
    return fText

# Takes in a term and finds the (best) wikipedia page on that term.
def getImgUrls(text, semCheck):
    wiki = "https://en.wikipedia.org"
    fText = text.replace(" ", "_")
    initHref = "/wiki/" + fText
    initUrl = wiki + initHref
    page = requests.get(initUrl)
    initSoup = BeautifulSoup(page.content, 'html.parser')
    initPTags = initSoup.find_all('p')
    isRedirectionPage = False

    # Finds if the initial page is a redirection page, in which case I need another step to get to the correct page.
    for tag in initPTags:
        tag = str(tag)
        if "may refer to:" in tag:
            isRedirectionPage = True
            break
    mainLink = initUrl

    # The next step.
    if isRedirectionPage:
        titleTags = initSoup.find_all('span', class_ = "mw-headline")

        # Titles of the possible next pages.
        titles = []
        for tag in titleTags:
            titles.append(tag.get("id"))
        if "See_also" in titles:
            titles.remove("See_also")
        liTags = initSoup.find_all('li')
        tagCt = 0
        for tag in liTags:
            badTitle = False
            ulTags = tag.find_all("ul")
            if ulTags != None:
                for ulTag in ulTags:
                    ulLiTags = ulTag.find_all("li")
                    if ulLiTags != False:
                        for ulLiTag in ulLiTags:
                            ulLiTagsClasses = ulLiTag.get("class")
                            if ulLiTagsClasses != None:

                                # 'toclevel' is only in the tags of titles that don't directly link to new pages
                                # (I think 'Science, technology, and mathematics' in https://en.wikipedia.org/wiki/Set
                                # should be an example).
                                if "toclevel" in ulLiTagsClasses[0]:
                                    badTitle = True
                                    break
                    if badTitle:
                        break
            if badTitle:
                titles.pop(tagCt)
                tagCt -= 1
            tagCt += 1

        # The title whose links I use is determined by semantic similarity to the word 'mathematics'.
        model = SentenceTransformer('all-MiniLM-L6-v2')
        scrs = {}
        for title in titles:
            wrds = [title]
            if "_" in title:
                wrds = title.split("_")
            semWrds = []
            for _ in wrds:
                semWrds.append(semCheck)
            embed1 = model.encode(semWrds, convertToTensor = True)
            embed2 = model.encode(wrds, convertToTensor = True)
            cosScrs = util.cos_sim(embed1, embed2)
            totalScr = 0
            for i in range(len(semWrds)):
                totalScr += cosScrs[i][i].item()

            # If there are multiple words in the title, I take the average of their similarity score.
            avgScr = totalScr/len(semWrds)
            scrs[title] = avgScr

        # The final title whose links I will use.
        bestTitle = titles[0]
        for title in titles:
            if scrs[title] > scrs[bestTitle]:
                bestTitle = title
        ulTags = initSoup.find_all("ul")
        nextPageHrefs = []

        # The position of the selected title among all of the possible titles on the page.
        linkIdx = titles.index(bestTitle)
        for tag in ulTags:
            nextPageLinkTag = False
            for subtag in tag:
                subtag = str(subtag)
                subtagChnks = subtag.split(" ")
                for chnk in subtagChnks:
                    if "href=" in chnk:
                        href = chnk[6:len(chnk) - 1]

                        # A check to see if the current tag is representative of a title that links to other pages.
                        if "/wiki/" in href:

                            # If the current tag is representative of the selected title...
                            if linkIdx == 0:
                                nextPageHrefs.append(href)
                            nextPageLinkTag = True

            # Decrease linkIdx if the tag is of a linking title to ultimately arrive at the selected title.
            if nextPageLinkTag:
                linkIdx -= 1
        nextPageHref = nextPageHrefs[0]
        mainLink = wiki + nextPageHref

    # Where the image links of the page will be stored
    fileName = formatText(text) + "_IMAGELINKS.txt"
    path = "ImageLinks/" + fileName
    if not os.path.isfile(path):
        with open(path, "a") as f:
            page = requests.get(mainLink).text
            imgSoup = BeautifulSoup(page, 'html.parser')

            # Finding the images.
            for rawImg in imgSoup.find_all('img'):
                link = rawImg.get('src')
                if link:

                    # Some wikipedia images are used across different pages, and they are for the site instead of the
                    # topic(s) described on the page.
                    ubiqLinks = [
                        "/static/images/icons/wikipedia.png",
                        "/static/images/mobile/copyright/wikipedia-wordmark-en.svg",
                        "/static/images/mobile/copyright/wikipedia-tagline-en.svg",
                        "//upload.wikimedia.org/wikipedia/en/thumb/1/1b/Semi-protection-shackle.svg/20px-Semi-protection-shackle.svg.png"
                    ]
                    if link not in ubiqLinks:
                        if "https:" not in link:
                            link = "https:" + link
                        f.write(link + "\n")
            f.close()

    # Return the path of the image link file.
    return fileName

def capitalizeFirstLetters(text):
    return " ".join(word[0].upper() + word[1:] for word in text.split(" "))

# Takes in the image link file and downloads all of the images linked to within it to a folder.
def downloadImages(fileName):
    term = fileName[0:fileName.index("IMAGELINKS") - 1]

    # Where the images will be downloaded.
    folderName = term + "Images"
    dir = os.path.join("Images/OriginalImages/", folderName)
    if not os.path.exists(dir):
        os.mkdir(dir)
        with open("ImageLinks/" + fileName, "r") as f:
            count = 1
            for line in f:
                try:
                    line = line.strip()
                    if line[0] != "/":
                        content = requests.get(line).content

                        # Determine the ending of the image (ext).
                        ext = -1
                        if content.startswith(b'<svg'):
                            ext = '.svg'
                        elif line.endswith('.png'):
                            ext = '.png'
                        elif line.endswith('.jpg'):
                            ext = '.jpg'
                        if ext == -1 or ext == ".svg":
                            continue

                        # Write the image ending to a certain file.
                        endingFileName = term + "_IMAGEENDINGS.txt"
                        path = "LinkEndings/" + endingFileName
                        with open(path, "a") as f2:
                            f2.write(ext + "\n")
                        f2.close()
                        urllib.request.urlretrieve(line, "Images/OriginalImages/" + folderName + "/" + term + str(count) + ext)
                        count += 1
                except:
                    continue
            f.close()

    # Create two new folders for image preparation in the future.
    dir2 = os.path.join("Images/OpaqueImages/", "Opaque" + folderName)
    if not os.path.exists(dir2):
        os.mkdir(dir2)
    dir3 = os.path.join("Images/ResizedImages/", "Resized" + folderName)
    if not os.path.exists(dir3):
        os.mkdir(dir3)

# Checks if the text is the name of a symbol that Manim can display through LaTex.
def isSymbol(text):
    symbolDct = {"plus" : "+"} # WILL FILL LATER
    return symbolDct[text] if text in symbolDct else False

# Creates a tuple with the text representing the object and the object's type.
def createObject(text):

    # Type 1 = text ; Type 2 = image.
    objText = objType = -1
    symbol = isSymbol(text)
    if symbol != False:
        objText = symbol
        objType = 1
    else:
        objText = text
        fileName = getImgUrls(text, "mathematics")
        downloadImages(fileName)
        objType = 2
    return objText, objType

def buildWireframe(text):
    fText = formatText(text)
    imgEnding = ""
    with open("LinkEndings/" + fText + "_IMAGEENDINGS.txt") as f:
        ending = ".png"
        for line in f:
            ending = line
            break
        if ending[len(ending) - 1] == "\n":
            ending = ending[:len(ending) - 1]
        imgEnding = ending
    imgPath = "Images/OriginalImages/" + fText + "Images/"
    imgName = fText + "1" + imgEnding
    opaqueOutputPath = "Images/OpaqueImages/Opaque" + fText + "Images/"
    opaquedImg = fcv.opaqueImage(imgPath, imgName, opaqueOutputPath)
    resizeOutputPath = "Images/ResizedImages/Resized" + fText + "Images/"
    resizedImg = fcv.resizeImage(opaqueOutputPath, "Opaque" + imgName, resizeOutputPath, 300)
    preppedImg = fcv.prepForImgAnalysis(resizedImg)
    linAppImg, objs = fcv.linearApproximation(preppedImg)
    linAppImg = cv2.cvtColor(linAppImg, cv2.COLOR_BGR2GRAY)
    preFindEdges = fcv.removeSmallObjects(linAppImg, 60)
    edges = fcv.findEdges(linAppImg, False, imgObjs = objs)
    edgeLngths = [*{len(edge[1]) for edge in edges}]
    edgeLngths.sort(reverse = True)
    edgeLngthDct = [[lngth, []] for lngth in edgeLngths]
    for edge in edges:
        lngth = len(edge[1])
        for itm in edgeLngthDct:
            if itm[0] == lngth:
                itm[1].append(edge)
                break
    srtdEdges = [edge for itm in edgeLngthDct for edge in itm[1]]
    pts = []
    segments = []
    for edge in srtdEdges:
        closeBnd = 5
        psblBnd = math.floor(len(edge[1])/8)
        if psblBnd > closeBnd:
            closeBnd = psblBnd
        edgePts, ptsInEdge, obj = edge
        endpt1 = ptsInEdge[0]
        endpt2 = ptsInEdge[len(ptsInEdge) - 1]
        for pt, iterObj in pts:
            endptY, endptX = endpt1
            ptY, ptX = pt
            if abs(endptY - ptY) <= closeBnd and abs(endptX - ptX) <= closeBnd and obj == iterObj:
                endpt1 = pt
                break
        for pt, iterObj in pts:
            endptY, endptX = endpt2
            ptY, ptX = pt
            if abs(endptY - ptY) <= closeBnd and abs(endptX - ptX) <= closeBnd and obj == iterObj:
                endpt2 = pt
                break
        if (endpt1, obj) not in pts:
            pts.append((endpt1, obj))
        if (endpt2, obj) not in pts:
            pts.append((endpt2, obj))
        segment = [pts.index((endpt1, obj)), pts.index((endpt2, obj))]
        if segment not in segments and reversed(segment) not in segments:
            segments.append(segment)
    initPt, initObj = pts[0]
    for pt, obj in pts:
        if pt[1] < initPt[1]:
            initPt = pt
            initObj = obj
        elif pt[1] == initPt[1]:
            if pt[0] < initPt[0]:
                initPt = pt
                initObj = obj
    initPtIdx = pts.index((initPt, initObj))
    pts.remove((initPt, initObj))
    for segment in segments:
        for idx in segment:
            addIdx = segment.pop(0)
            if addIdx < initPtIdx:
                addIdx += 1
            elif addIdx == initPtIdx:
                addIdx = 0
            segment.append(addIdx)
    initY, initX = initPt
    shiftPts = [(pt[0] - initY, pt[1] - initX) for pt, obj in pts]
    maxVal = max([max(abs(pt[0]), abs(pt[1])) for pt in shiftPts])
    sclConst = maxVal/3
    resPts = [(-1 * pt[0]/sclConst, pt[1]/sclConst) for pt in shiftPts]
    return resPts, segments

def visualizeObject(text):
    obj = createObject(text)
    objText, objType = obj
    term = capitalizeFirstLetters(text)
    with open("TermData.json", "a") as f:
        data = {"term": term}
        if objType == 1:
            data["type"] = "T_OBJ"
            data["text"] = objText
        elif objType == 2:
            data["type"] = "G_OBJ"
            fileName = getImgUrls(text, "mathematics")
            wf = buildWireframe(text)
            pts, segments = wf
            data["points"] = [[*reversed(pt)] for pt in pts]
            data["segments"] = segments
        jsonObj = json.dumps(data)
        f.write(jsonObj)
        f.close()

def main():
    term = args[0]
    visualizeObject(term)

if __name__ == "__main__":
    main()
