#include "Tracker.h"
#include "TrackItem.h"

using namespace cvip;

uint TrackItem::counter = 0; //! keep number of TrackItem instances
uint TrackItem::maxId = 1; //! use this each time an id is assigned to a new TrackItem

/**
 * A TrackItem becomes active only if it is tracked for
 * at least Tracker::NUM_MIN_DETECTIONS times
 *
 * @return bool
 */
bool TrackItem::isActive() const
{
    return numActiveFrames > cvip::Tracker::NUM_MIN_DETECTIONS;
}

/**
 * Update an active item using new rectangle
 * Assuming that this TrackItem is active in this frame
 *
 * @param  DetectionRect&
 * @return void
 */
void TrackItem::update(const DetectionRect& d)
{
    ++numActiveFrames;
    numInactiveFrames = 0;

    // Tracking 4 points
    cv::Mat measurement(kalman.M, 1, CV_32F);
    measurement.at<float>(0,0) = d.x1;
    measurement.at<float>(1,0) = d.y1;
    measurement.at<float>(2,0) = d.x2;
    measurement.at<float>(3,0) = d.y2;

    kalman.filter->predict();
    const cv::Mat& statePost = kalman.filter->correct(measurement);

    // update rectangle
    setRectFrom(statePost);
}

/**
 * Update an non-active item, an item which is not
 * detected in this frame.
 * Return false if item is inactive for long time, true otherwise
 *
 * @return bool
 */
bool TrackItem::update()
{
//    std::cout << "Tracking ... " << numInactiveFrames << std::endl;

    if (++numInactiveFrames >= Tracker::NUM_MAX_INACTIVE_FRAMES || !isActive())
        return false;

    const cv::Mat& statePre = kalman.filter->predict();

    // update rectangle
    setRectFrom(statePre);

    return true;
}

/**
 * Update rectangle from the most recent state.
 *
 * @param  Mat& state
 * @return void
 */
void TrackItem::setRectFrom(const cv::Mat &state)
{
    using cvip::round;

    dRect.x1 = state.at<float>(0,0);
    dRect.y1 = state.at<float>(1,0);
    dRect.x2 = state.at<float>(2,0);
    dRect.y2 = state.at<float>(3,0);

    dRect.width = dRect.x2-dRect.x1;//2*halfWin;
    dRect.height = dRect.y2-dRect.y1;//2*halfWin;
}

/**
 * Kalman constructor: Parameters of the filter are set in here.
 * These parameters have a direct effect on the behaviour pf the filter.
 */
TrackItem::Kalman::Kalman(const DetectionRect& initRect)
{
    N = 8; // dimension of trans. matrix
    M = 4; // length of measurement

    // setup kalman filter with a Model Matrix, a Measurement Matrix and no control vars
    filter = new cv::KalmanFilter(N, M, 0);

    // transitionMatrix is eye(n,n) by default
    filter->transitionMatrix.at<float>(0,4) = 0.067f; // dt=0.04, stands for the time
    filter->transitionMatrix.at<float>(1,5) = 0.067f; // betweeen two video frames in secs.
    filter->transitionMatrix.at<float>(2,6) = 0.067f;
    filter->transitionMatrix.at<float>(3,7) = 0.067f;

    // measurementMatrix is zeros(n,p) by default
    filter->measurementMatrix.at<float>(0,0) = 1.0f;
    filter->measurementMatrix.at<float>(1,1) = 1.0f;
    filter->measurementMatrix.at<float>(2,2) = 1.0f;
    filter->measurementMatrix.at<float>(3,3) = 1.0f;

    using cv::Scalar;

    // assign a small value to diagonal coeffs of processNoiseCov
    cv::setIdentity(filter->processNoiseCov, Scalar::all(1e-2)); // 1e-2

    // Measurement noise is important, it defines how much can we trust to the
    // measurement and has direct effect on the smoothness of tracking window
    // - increase this tracking gets smoother
    // - decrease this and tracking window becomes almost same with detection window
    cv::setIdentity(filter->measurementNoiseCov, Scalar::all(1e-1)); // 1e-1
    cv::setIdentity(filter->errorCovPost, Scalar::all(1));

    // we are tracking 4 points, thus having 4 states: corners of rectangle
    filter->statePost.at<float>(0,0) = initRect.x1;
    filter->statePost.at<float>(1,0) = initRect.y1;
    filter->statePost.at<float>(2,0) = initRect.x2;
    filter->statePost.at<float>(3,0) = initRect.y2;
}
