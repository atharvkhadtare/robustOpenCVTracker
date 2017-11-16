import cv2
import sys
import math
import time as t

import numpy as np
import scipy
# import math

timeLimit = 30
video_source = "car3.mp4"
video_source = 0

cascadePath = "car4-1.xml"
# cascadePath = "haarcascade_frontalface_alt.xml"
# cascadePath = "haarcascade_eye.xml"

# Set up tracker.
# Instead of MIL, you can also use
# BOOSTING, KCF, TLD, MEDIANFLOW or GOTURN
TrackerType = "MEDIANFLOW"

def cent_dist(a,b):
    temp = math.sqrt( pow((a[1]-b[1]),2)+pow((a[0]-b[0]),2))
    return temp

def checkOverlap(a,b):
    (x1, y1, w1, h1) = a
    (x2, y2, w2, h2) = b

    if(x1 < (x2 +w2/2)):
        if(x1+w1 > (x2 +w2/2)):
            if(y1 < (y2 +h2/2)):
                if(y1+h1 > (y2 +h2/2)):
                    return True
    if(x2 < (x1 +w1/2)):
        if(x2+w2 > (x1 +w1/2)):
            if(y2 < (y1 +h1/2)):
                if(y2+h2 > (y1 +h1/2)):
                    return True
    return False

def removeOverlaps(objectsFoundLocal):
    objectsFoundTemp = []
    for i in range(0, len(objectsFoundLocal)):
        matchBool = False
        for j in range(i+1, len(objectsFoundLocal)):
            if (checkOverlap(objectsFoundLocal[i],objectsFoundLocal[j])):
                matchBool = True
        if not matchBool:
            objectsFoundTemp.append(objectsFoundLocal[i])
    return objectsFoundTemp

tracker={}
status={}
trackerLifeTime={}
bbox={}
ok={}
centroid_object={}
centroid_tracker={}
activeTrackers = np.array(0)
no_trackers=15

def deactivateTracker(indexOfTracker):
    global activeTrackers
    index = np.argwhere(activeTrackers==indexOfTracker)
    print ("index = ", index)
    activeTrackers = np.delete(activeTrackers, index)

for i in range(0,no_trackers):
    trackerLifeTime[i] = 0
    tracker[i]=cv2.Tracker_create(TrackerType)
    status[i] = "OFF"
    ok[i]=False

#cascade init
object_cascade = cv2.CascadeClassifier(cascadePath)

# Video Initialize
video = cv2.VideoCapture(video_source)

# Exit if video not opened.
if not video.isOpened():
    #1print ("Could not open video")
    sys.exit()

# Read first frame.
ok1, frame = video.read()
ok1, frame = video.read()
ok1, frame = video.read()
ok1, frame = video.read()
ok1, frame = video.read()
ok1, frame = video.read()
ok1, frame = video.read()
ok1, frame = video.read()
ok1, frame = video.read()
ok1, frame = video.read()
# frame = frame[:,(frame.shape[1]/2):frame.shape[1]].copy()
frameTrackersPrev = frame.copy()
frameHaarPrev     = frame.copy()
if not ok1:
    #1print ('Cannot read video file')
    sys.exit()

#convt gray
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#detect objects
objects = object_cascade.detectMultiScale(gray, 1.3, 5)


#Remove overlaping objects
objects = removeOverlaps(objects)
# objects = esvm_nms(objects, 30)

#initialize all tracker related arrays
for i in range(0, no_trackers):
    bbox[i]=(0, 0, 0, 0)
    tracker[i]=cv2.Tracker_create(TrackerType)
    ok[i] = tracker[i].init(frame, bbox[i])
    tracker[i]=cv2.Tracker_create(TrackerType)
    status[i] = "OFF"
    ok[i] = False

trackersOn     = 0  #counter for No of active trackers
objectCount       = 0     # counter
objectCountIn     = 0     # counter
objectCountOut    = 0     # counter
stTime         = t.time()
pause          = False
absoluteStTime = t.time()
totalFrames    = 0
lastUsedTracker = 0
while True:
    activeTrackers = np.unique(activeTrackers)
    # Read a new frame
    #1print("________________NEW LOOP_____________________________________________________________________________________________________________________________________")
    #1print("________________NEW LOOP_____________________________________________________________________________________________________________________________________")
    #1print("________________NEW LOOP_____________________________________________________________________________________________________________________________________")
    #1print("________________NEW LOOP_____________________________________________________________________________________________________________________________________")
    #1print("________________NEW LOOP_____________________________________________________________________________________________________________________________________")
    #1print("")
    #1print("")
    ok1, frame = video.read()
    # frame = frame[:,(frame.shape[1]/2):frame.shape[1]].copy()
    frameHaar = frame.copy()
    frameTrackers = frame.copy()
    if not ok1:
        break

    #convt gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #detect objects
    objects = object_cascade.detectMultiScale(gray, 1.3, 5)
    #Remove overlaping objects
    objects = removeOverlaps(objects)

    #1print("len(objects): ", len(objects))

    #prepare centroid of Haar objects
    i=0
    for (x,y,w,h) in objects:
        centroid_object[i]=(x+(w/2),y+(h/2))
        i+=1

    #prepare centroid of Tracker objects   (Not used for Computation. Used only for #1printing and debugging)
    for i in range(0,no_trackers):
        if status[i] != "OFF":
            temp = (bbox[i][0]+(bbox[i][2]/2),bbox[i][1]+(bbox[i][3]/2))
            # centroid_tracker[i]= temp
        else:
            centroid_tracker[i]=(0,0)
    for i in activeTrackers:
        temp = (bbox[i][0]+(bbox[i][2]/2),bbox[i][1]+(bbox[i][3]/2))

    #1print (": ",     centroid_object)
    #1print ("tracker: ", bbox,"\n\n")
    #1print ("tracker: ", centroid_tracker,"\n\n")
    for i in range(0,len(objects)):
        matchFound = False
        # for j in range (0,no_trackers):     #checck if any trackers are already tracking the haar 
        for j in activeTrackers:     #checck if any trackers are already tracking the haar 
            if status[j] == "DUP":
                continue
            if status[j] == "OFF":
                continue
            # if status[j] == "LOST":
            #     continue
            if ((checkOverlap(objects[i],bbox[j])) | (checkOverlap(bbox[j],objects[i]))):
                    #1print ("Overlap >> : ", objects[i], "TRACKER: ", bbox[j], end = '  ')
                    #1print ("      >> : ", i, "TRACKER: ", j)
                    if((objects[i][2] < bbox[j][2]) & (matchFound == False)):
                    # if((matchFound == False)):
                        p1 = (int(bbox[j][0]), int(bbox[j][1]))
                        p2 = (int(bbox[j][0] + bbox[j][2]), int(bbox[j][1] + bbox[j][3]))
                        # cv2.putText(frameTrackers,(str)(j),(int(bbox[j][0] + 5) ,int(bbox[j][1] + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0, 0, 255),2)
                        # cv2.putText(frameTrackers,(str)(status[j][0]),(int(bbox[j][0] + 5) ,int(bbox[j][1] - 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0, 0, 255),2)
                        # #1print ( "FTrack Text: : ", i,"tr: ", j, " Col: Red")
                        # cv2.rectangle(frameTrackers, p1, p2, (0,0,255), 1)
                        if status[j] == "NEw":
                            continue
                        #     status[j] = "UD"
                        tracker[j]=cv2.Tracker_create(TrackerType)
                        temp = (objects[i][0], objects[i][1], objects[i][2], objects[i][3])
                        bbox[j] = temp
                        ok[j] = tracker[j].init(frame, temp)
                        status[j] = "HUD"
                        #1print ("HUD")
                        #1print ("HUD")
                        #1print ("HUD")
                        #1print ("HUD")
                        #1print ("Tracker Updated TrNo: ", j, "No: ", i)
                        #1print ("HUD")
                        #1print ("HUD")
                        #1print ("HUD")
                        #1print ("HUD")
                        #1print ("HUD")
                        #1print ("HUD")
                        #1print ("HUD")
                        #1print ("HUD")
                        matchFound = True
                        p1 = (int(bbox[j][0]), int(bbox[j][1]))
                        p2 = (int(bbox[j][0] + bbox[j][2]), int(bbox[j][1] + bbox[j][3]))
                        # cv2.putText(frameTrackers,(str)(j),(int(bbox[j][0] + 5) ,int(bbox[j][1] + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(255, 0, 0),2)
                        # cv2.putText(frameTrackers,(str)(status[j][0]),(int(bbox[j][0] + 5) ,int(bbox[j][1] - 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(255, 0, 0),2)
                        # #1print ( "FTrack Text: : ", i,"tr: ", j, " Col: Blue")
                        # cv2.rectangle(frameTrackers, p1, p2, (255,0,0), 1)
                        # cv2.imshow('Haar', frameHaar)
                        # cv2.imshow('Trackers', frameTrackers)
                        # pause = True
                        # continue
                    cv2.putText  (frameHaar, (str)(i),(objects[i][0] + 5 , objects[i][1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    #1print ( "FHaar Text: : ", i,"tr: ", j, " Col: Green")
                    cv2.rectangle(frameHaar, (objects[i][0], objects[i][1]), (objects[i][0]+objects[i][2], objects[i][1]+objects[i][3]), (0, 255, 0), 2)
                    matchFound = True
        if matchFound == False:             #if not already tracked, create new tracker
            for k in range(0,no_trackers):
                if status[(lastUsedTracker+k)%no_trackers] == "OFF":               #check if tracker is available
                    lastUsedTracker = (lastUsedTracker+k)%no_trackers
                    k = lastUsedTracker
                    #1print ("Init New  >> : ", objects[i], end = '  ')
                    #1print ("NEW >> : ", i, "TRACKER: ", k)
                    # for j in range (0,no_trackers):
                    #     #1print(" ",objects[i],": trck ",bbox[j]," = ",checkOverlap(objects[i],bbox[j]))
                    cv2.putText(frameHaar,(str)(i),(objects[i][0] + 5 ,objects[i][1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 255),2)
                    #1print ( "FHaar Text: : ", i,"tr: ", j, " Col: Green")
                    cv2.rectangle(frameHaar, (objects[i][0], objects[i][1]), (objects[i][0]+objects[i][2], objects[i][1]+objects[i][3]), (0, 0, 255), 2)
                    trackerLifeTime[k] = 0
                    tracker[k]=cv2.Tracker_create(TrackerType)
                    temp = (objects[i][0], objects[i][1], objects[i][2], objects[i][3])
                    bbox[k] = temp
                    ok[k] = tracker[k].init(frame, temp)
                    temp = (objects[i][0], objects[i][1], objects[i][2], objects[i][3])
                    bbox[k] = temp
                    trackersOn += 1
                    status[k] = "NEW"
                    activeTrackers = np.append(activeTrackers, k)
                    break
    #1print("\n\n")

    # Update tracker
    #1print ("st = ", status)
    # for i in range(0,no_trackers):
    for i in activeTrackers:
        # if   status[i] == "OFF":      #If tracker is OFF do not update
        #     # #1print (i, ": OFF ", end='. ')
        #     pass
        if status[i] == "UD":      #If tracker is OFF do not update
            # #1print (i, ": UPD", end='. ')
            if((bbox[i][0] + bbox[i][2]/2) < (frame.shape[1]/2)):
                Dir = "OUT"
            else:
                Dir = "IN"
            ok[i], bbox[i] = tracker[i].update(frame)
            trackerLifeTime[i] += 1
            if not ok[i]:
                status[i] = "LOST"
                #1print ("trackerOn -= 1", i, "st = ", status[i])
                trackersOn -= 1
                pause = True
                objectCount += 1
                # if(Dir == "IN"):
                #     objectCountIn  += 1
                # elif(Dir == "OUT"):
                #     objectCountOut += 1
            elif bbox[i][2]<20:     #Out Of Sight (Too Small)
                status[i] = "LOST"
                #1print ("trackerOn -= 1", i, "st = ", status[i])
                trackersOn -= 1
                objectCount += 1
                # if(Dir == "IN"):
                #     objectCountIn  += 1
                # elif(Dir == "OUT"):
                #     objectCountOut += 1
                pause = True
                ok[i]=False
        elif status[i] == "NEW":      #If tracker is NEw do not update
            # #1print (i, ": NEW", end='. ')
            status[i] = "UD"
        elif status[i] == "HUD":      #If tracker is recently updated from Haar do not update
            # #1print (i, ": NEW", end='. ')
            status[i] = "UD"
        elif status[i] == "DUP":      #If tracker is duplicate do not update
            # #1print (i, ": DUP ", end='. ')
            status[i] = "OFF"
            deactivateTracker(i)
            bbox[i] == (0,0,0,0)
            ok[i]=False
        elif status[i] == "LOST":      #If tracker is LOST do not update
            # #1print (i, ": LOST ", end='. ')
            status[i] = "OFF"
            deactivateTracker(i)
            bbox[i] == (0,0,0,0)
            ok[i]=False

    #1print ("")
    #1print ("ok = ",ok)
                
    # Remove Duplicate Trackers
    # trackersTemp = {}
    for i in range(0, len(bbox)):
        if(bbox[i] == (0,0,0,0)):
            continue
        if(status[i] == "OFF"):
            continue
        matchBool = False
        for j in range(i+1, len(bbox)):
            if(bbox[j] == (0,0,0,0)):
                continue
            if(status[j] == "OFF"):
                continue
            if (checkOverlap(bbox[i],bbox[j])):
                #1print ("Dupl Tracker >> Track1: ", bbox[i], "Track2: ", bbox[j], end = '  ')
                #1print ("      >> Tr1: ", i, "Tr2: ", j)
                #1print ("initial Status: (i)", i, ": ", status[i], " (j)", j, ": ", status[j])
                status[i] = "DUP"
                trackersOn -= 1
                bbox[i] == (0,0,0,1)
                matchBool = True
    # for i in range(0,no_trackers):
    for i in activeTrackers:
        if status[i] == "OFF":
            continue
        if ok[i]:
            p1 = (int(bbox[i][0]), int(bbox[i][1]))
            p2 = (int(bbox[i][0] + bbox[i][2]), int(bbox[i][1] + bbox[i][3]))
            if((status[i] == "UD")):
                cv2.putText(frameTrackers,(str)(i),(int(bbox[i][0] + 5) ,int(bbox[i][1] + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0, 255, 0),2)
                cv2.putText(frameTrackers,(str)(trackerLifeTime[i]),(int(bbox[i][0] + 5) ,int(bbox[i][1] + 50)), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0, 255, 0),2)
                # cv2.putText(frameTrackers,(str)(status[i][0]),(int(bbox[i][0] + 5) ,int(bbox[i][1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0, 255, 0),2)
                #1print ( "FTrack Text: : ", i,"tr: ", j, " Col: Green")
                cv2.rectangle(frameTrackers, p1, p2, (0,255,0), 1)
            # elif((status[i] == "NEW")):
            #     cv2.putText(frameTrackers,(str)(i),(int(bbox[i][0] + 5) ,int(bbox[i][1] + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(255, 255, 0),2)
            #     cv2.putText(frameTrackers,(str)(status[i][0]),(int(bbox[i][0] + 5) ,int(bbox[i][1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(255, 255, 0),2)
            #     #1print ( "FTrack Text: : ", i,"tr: ", j, " Col: Blue-Green")
            #     cv2.rectangle(frameTrackers, p1, p2, (255,0,0), 1)
            #     #1print ("NEW")
            #     #1print ("NEW")
            #     #1print ("NEW")
            #     #1print ("NEW")
            #     #1print ("NEW")
            #     #1print ("NEW")
            #     #1print ("NEW")
            #     #1print ("NEW")
            #     #1print ("NEW")
            #     #1print ("NEW")
            #     #1print ("NEW")
            #     #1print ("NEW")
            #     pause = True

    # Display results
    # cv2.putText(frameTrackers,(str)(objectCountOut),(int(30) ,int(110)), cv2.FONT_HERSHEY_SIMPLEX, 2.5,(0, 0, 255),3)
    # cv2.putText(frameTrackers,(str)(objectCount   ),(int(300) ,int(110)), cv2.FONT_HERSHEY_SIMPLEX, 2.5,(0, 0, 255),3)
    # cv2.putText(frameTrackers,(str)(objectCountIn ),(int(530) ,int(110)), cv2.FONT_HERSHEY_SIMPLEX, 2.5,(0, 0, 255),3)
    #1print ("st = ", status)
    # cv2.imshow('Haar', frameHaar)
    cv2.imshow('Trackers', frameTrackers)
    # cv2.imshow('HaarPrev', frameHaarPrev)
    # cv2.imshow('TrackersPrev', frameTrackersPrev)
    # frameTrackersPrev = frameTrackers.copy()
    # frameHaarPrev     = frameHaar.copy()
    #1print ("TrackLife:", trackerLifeTime)
    #1print("trackersOn: ", trackersOn)
    #1print("objectCount: ", objectCount)
    # Exit if ESC pressed
    if pause:
        key = cv2.waitKey(1) & 0xFF
        pause = False
        if ((key == ord('q')) | (key == 27)):
            break
    key = cv2.waitKey(1) & 0xFF
    if ((key == ord('q')) | (key == 27)):
        break
    endTime = t.time()
    totalFrames += 1
    # print ("FPS = ", int(1/(endTime-stTime)), "      FPS = ", int(totalFrames/(endTime-absoluteStTime)), "Active = ", activeTrackers)
    print ("FPS = ", int(1/(endTime-stTime)), "      FPS = ", int(totalFrames/(endTime-absoluteStTime)))
    # if(totalFrames == 1000):
    #     totalFrames = 0
    #     absoluteStTime = t.time()
    if(t.time()-absoluteStTime > timeLimit):
        break
    stTime = endTime
video.release()
cv2.destroyAllWindows()
print ("Total Frames in ", t.time()-absoluteStTime, " seconds = ", totalFrames)
