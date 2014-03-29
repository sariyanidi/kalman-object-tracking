#include "Tracker.h"
#include "FaceDetector.h"
#include "opencv2/highgui/highgui.hpp"

using namespace cvip;

/**
 * Destructor
 * Release memory. Delete detector and all trackItems
 */
Tracker::~Tracker()
{
    delete detector;

    typedef std::map<uint, TrackItem*>::iterator TiIter;

    for (TiIter it = trackItems.begin(); it != trackItems.end(); ++it)
        drop(it->first);
}

/**
 * Sample function, run tracker on video
 *
 * @todo erase this function !
 * @return void
 */
void Tracker::onVideo()
{
    using namespace cv;

    Mat frame;
    VideoCapture cap(0);

    if (!cap.isOpened())
        return;

    cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

    while (1)
    {
        double t = (double) cv::getTickCount();

        cap >> frame;

        cv::Mat frame2 = frame.clone();

		std::vector<Image*> images = Image::create_scale_space(frame,detector);
        std::vector<cvip::DetectionRect> detections;

        detections = detector->detect(images, true);

		for (uint i=0; i<images.size(); i++)
			delete images[i];

        for (uint i=0; i<detections.size(); ++i)
        {
            cvip::DetectionRect& d = detections[i];
            cv::Rect r(d.x1, d.y1, d.width, d.height);
            cv::rectangle(frame2, r, CV_RGB(255,0,0),3);
        }

        updateWith(detections);

        typedef std::map<uint, TrackItem*>::const_iterator TiIter;

        for (TiIter it=trackItems.begin();
                it != trackItems.end(); ++it)
        {
            cvip::DetectionRect& d = it->second->dRect;
            cv::Rect r(d.x1, d.y1, d.width, d.height);

            // draw bold if item is being tracked
            if (it->second->isActive())
            {
                cv::rectangle(frame, r, cv::Scalar::all(255),3);

                std::stringstream ss1, ss2, ss3;
                std::string faceTracked("Face Tracked!");

                ss1 << "id: " << it->first;
                ss2 << "Uptime: " << it->second->uptime();
                ss3 << "# of dead frames: " << it->second->numInactiveFrames;

                cv::putText(frame, ss1.str().c_str(), Point(d.x1,d.y2), FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255,0,0), 2);
                cv::putText(frame, ss2.str().c_str(), Point(d.x1,d.y2+20), FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255,0,0), 2);

                if (it->second->numInactiveFrames >0)
                    cv::putText(frame, faceTracked.c_str(), Point(d.x1,d.y2+40), FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255,0,0), 2);
            }
            else // draw thin otherwise
            {
                //cv::rectangle(frame, r, cv::Scalar::all(255),1);
            }
        }

        //std::cout << numItems() << std::endl;

        cv::imshow("DETECTION", frame2);
        cv::imshow("TRACKING", frame);

        if(waitKey(10) >= 0)
            break;
    }
}

/**
 * Take new detections and update the whole trackItems list.
 * Processes are distributed to some internal methods.
 *
 * @param  vector<DetectionRect>& freshDetects - incoming detections
 * @return void
 */
void Tracker::updateWith(std::vector<DetectionRect>& freshDetects)
{
    // a flag map keeping the state of each item: updated or not
    std::map<uint, bool> flagActive;

    // 1) update whatever you matchs
    flagActive = updateActiveItems(freshDetects);

    // 2) add remaining rectangles ass new items
    this->addNewItems(freshDetects);

    // 3) update unmatched items, drop them if necessary
    this->updateInactiveItems(flagActive);
}

/**
 * Take new detections and update the ones matched with the
 * existing items.
 * WARNING! the detections which are not matched are removed
 * from the vector.
 *
 * @param  vector<DetectionRect>& freshDetects - incoming detections
 * @return std::map<uint,bool> - a flag for each item stating whether item is updated or not
 */
std::map<uint,bool> Tracker::updateActiveItems(std::vector<DetectionRect>& freshDetects)
{
    std::map<uint, bool> flagActive;

    typedef std::map<uint, TrackItem*>::iterator TiIter;

    for (TiIter it=trackItems.begin(); it != trackItems.end(); ++it)
        flagActive[it->first] = false;

    // associate rects to items
    for (int i=freshDetects.size()-1; i>=0; --i)
    {
        int maxIdx = -1;
        double maxArea = 0.;

        for (TiIter it=trackItems.begin(); it != trackItems.end(); ++it)
        {
            if (flagActive[it->first]) // update an item only once
                continue;

            uint area = Rect::intersect(freshDetects[i], it->second->dRect);

            // check if they intersect
            if (area > 0)
            {
                double ratio1 = (double)area/(freshDetects[i].width*freshDetects[i].height);
                double ratio2 = (double)area/(it->second->dRect.width*it->second->dRect.height);

                // try to find the best/largest intersection between rects
                if (max<double>(ratio1,ratio2) > maxArea)
                {
                    maxArea = max<double>(ratio1,ratio2);
                    maxIdx = it->first;
                }
            }
        }

        // is the best good enough?
        if (maxArea > 0.20)
        {
            // if it is, freshDetects[i] is assumed to stand for the
            // TrackedItem with id = maxIdx
            flagActive[maxIdx] = true;
            trackItems[maxIdx]->update(freshDetects[i]);
            freshDetects.erase(freshDetects.begin()+i);
        }
    }

    // this will be piped to updateInactiveItems()
    return flagActive;
}

/**
 * Take a list including a flag for each TrackItem: updated or not.
 * Update each inactive item
 *
 * @param  map<uint,bool>& flagActive
 * @return void
 */
void Tracker::updateInactiveItems(const std::map<uint,bool>& flagActive)
{
    typedef std::map<uint, bool>::const_iterator FlagIter;

    for (FlagIter it = flagActive.begin(); it != flagActive.end(); ++it )
    {
        // skip if item is active at this frame
        if (it->second)
            continue;

        // drop item if it's inactive for long
        if (!trackItems[it->first]->update())
            drop(it->first);
    }
}

/**
 * Add remaining unmatched freshRects as new TrackItems
 *
 * @param  vector<DetectionRect>
 * @return void
 */
void Tracker::addNewItems(std::vector<DetectionRect>& freshDetects)
{
    for (uint i=0; i<freshDetects.size(); ++i)
        add(new TrackItem(freshDetects[i]));
}
