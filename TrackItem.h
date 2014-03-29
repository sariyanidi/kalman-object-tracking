#ifndef TRACKITEM_H
#define TRACKITEM_H

#include "FaceDetector.h"
#include "opencv2/video/tracking.hpp"

namespace cvip
{
    /**
     * Class to keep a track item.
     * Duty of this class:
     * - Update tracking rectangle
     * - Keep and give tracking statistics
     *
     * @author evangelos sariyanidi / sariyanidi[at]gmail[dot]com
     * @todo update uptime() !
     * @date april 2011
     */
    class TrackItem
    {
    public:
        // count instances + assign new
        TrackItem(const cvip::DetectionRect& d) : id(maxId++),
            numInactiveFrames(0), numActiveFrames(1), tStart(cv::getTickCount()), kalman(d),
            dRect(d.x1, d.y1, d.width, d.height, d.angle, d.scale) {}

        // decrease num of instances on destruct
        ~TrackItem() { --counter; if (!isActive()) --maxId; }

        // update active item with rect
        void update(const cvip::DetectionRect& dRect);

        // update inactive item
        bool update();

        // time passed since the tracking this (in secs.)
        double uptime() const { return (double)(cv::getTickCount()-tStart);/*/CLOCKS_PER_SEC;*/ }
		
        // dont begin tracking immediately, begin when ...
        bool isActive() const;

        /**
         * Wrap class cv::KalmanFilter like this to handle it prettier.
         */
        class Kalman
        {
        public:
            friend class TrackItem;

            Kalman(const cvip::DetectionRect& initRect);
            ~Kalman() { delete filter; }

        private:
            uint N; //! dimension of transition matrix: NxN
            uint M; //! length of measurement vector

            cv::KalmanFilter* filter;
        };

        //! @property unique id of track item
        const uint id;

        //! @property number of inactive frames - use to drop track if needed
        unsigned short numInactiveFrames;

        //! @property number of active frames - just record data
        uint numActiveFrames;

    private:
        // set detection item
        void setRectFrom(const cv::Mat& state);

        //! @property count TrackItem instances
        static uint counter;

        //! @property use to assign a new id
        static uint maxId;

        //! @property tick count at Track init. time of this item
        unsigned long tStart;

        //! @property anything kalman is inside here
        Kalman kalman;

    public:
        //! @property rect to draw, most up-to-date item position
        cvip::DetectionRect dRect;
    };

}

#endif // TRACKITEM_H
