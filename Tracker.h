#ifndef TRACKER_H
#define TRACKER_H

#include "FaceDetector.h"
#include "TrackItem.h"
#include <map>

namespace cvip
{
    /**
     * Tracker class written according to a "kind of" decorator pattern:
     * Take a detector and wrap it with this Tracker
     *
     * @todo convert FaceDetector* to Detector* after Detectors are abstracted
     * @todo include "Detector.h" rather than "FaceDetector.h" after abstraction
     * @todo update totalTime()
     * @author evangelos sariyanidi / sariyanidi[at]gmail[dot]com
     * @date april 2011
     */
    class Tracker
    {
    public:

        // construct tracker using a detector
        Tracker( cvip::FaceDetector* _detector ) : detector(_detector), tStart(cv::getTickCount()), numFrames(0) {}

        // in destructor delete detector and all track items
        ~Tracker();

        // Track items on video
        void onVideo();

        // add/drop trackItems
        void add(cvip::TrackItem* ti) { trackItems.insert(std::pair<uint, TrackItem*>(ti->id, ti)); }
        void drop(uint id) { delete trackItems[id]; trackItems.erase(id); }

        // update trackItems with fresh detections
        void updateWith(std::vector<DetectionRect>& freshDetects);

        // record regarding tracker
        uint numItems() const { return trackItems.size(); }
        double totalTime() const { return (double)(cv::getTickCount()-tStart);/*/CLOCKS_PER_SEC*/; }

        //! @property allowed num of inactive frames, drop tracking if this number exceeded
        static const unsigned short NUM_MAX_INACTIVE_FRAMES = 20;
		
        //! @property minimum number of detections before start to track an item
        static const unsigned short NUM_MIN_DETECTIONS = 3;

    private:
        //! @property detector to detect objects
        cvip::FaceDetector* detector;

        //! @property items being tracked -> associate each item with its id
        std::map<uint, cvip::TrackItem*> trackItems;

        //! @property tick count of Tracker initialization time
        unsigned long tStart;

        //! @property total number of frames run
        unsigned long numFrames;

        // see definition of Tracker::updateItems() for comments of these:
        std::map<uint, bool>
             updateActiveItems(std::vector<DetectionRect>& freshDetects);
        void updateInactiveItems(const std::map<uint, bool>& flagUpdated);
        void addNewItems(std::vector<DetectionRect>& freshDetects);
    };
}

#endif // TRACKER_H
