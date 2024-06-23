import cv2

def run_tracker(tracker, video_path = 'data/test.mp4'):
    video = cv2.VideoCapture(video_path)
    if_tracker_inited = False

    while True:
        ret, frame = video.read()
        if not ret:
            print('End of video.')
            break

        if not if_tracker_inited:
            tracker.init(frame, (631, 332,  54,  54))
            if_tracker_inited = True
        else:
            ok, bbox = tracker.update(frame)
            cv2.rectangle(frame,(bbox[0], bbox[1]),(bbox[0]+bbox[2],bbox[1]+bbox[3]),(0,255,0),2)
            cv2.imshow('video', frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


run_tracker(cv2.TrackerKCF_create())
run_tracker(cv2.TrackerCSRT_create())

# Compare the results:
# * Do you see any differences? If so, what are they?
# * Does one tracker perform better than the other? In what way?

# Answers
#