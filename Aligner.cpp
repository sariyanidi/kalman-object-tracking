#include "aligner.h"
#include "opencv2/highgui/highgui.hpp"

using namespace cvip;

/**
 * Constructor helper, compute eye centers, they will be used later.
 *
 * @return void
 */
void Aligner::computeEyeCenters(const std::vector<DetectionRect>& eyes)
{
    unsigned short lIdx, rIdx;

    // which is the left eye?
    if (eyes[0].x1 < eyes[1].x1) {
        lIdx = 0;
        rIdx = 1;
    } else {
        lIdx = 1;
        rIdx = 0;
    }

    // get eye centers to find rotation angle
    lEye.x = round((eyes[lIdx].x1+eyes[lIdx].x2)/2.);
    lEye.y = round((eyes[lIdx].y1+eyes[lIdx].y2)/2.);

    rEye.x = round((eyes[rIdx].x1+eyes[rIdx].x2)/2.);
    rEye.y = round((eyes[rIdx].y1+eyes[rIdx].y2)/2.);
}

/**
 * Return aligned face image
 *
 * @return cv::Mat
 */
cv::Mat Aligner::getAligned()
{
    int mp = maxPadding();

    // limit padding to a half face
    if (mp > r->width*0.5)
        mp = std::floor(r->width*0.5);

    // the rect given with the relative coordinates within the new box
    cv::Rect rRel(mp, mp, r->width, r->height);
    cv::Mat cropped(im->I, cv::Rect(r->x1-mp, r->y1-mp, r->width+2*mp, r->height+2*mp));

    // convert eye coords to new relative coords after face is cropped
    updateEyes(-mp+r->x1, -mp+r->y1);

    // rotate image and eye coords
    double theta = rotAngle();
    cv::Mat rotated = cvip::imrotate(cropped, theta*180/cvip::PI);
    rotateEyes(theta, rotated.cols/2, rotated.rows/2);

    // find strict rect and update eyes coords again
    cv::Rect strict = strictRect();
    updateEyes(strict.x, strict.y);

    cv::Mat strictFace(rotated, strict);
    /*
    cv::circle(strictFace, lEye, 4,  CV_RGB(255,0,0),2);
    cv::circle(strictFace, rEye, 4, cv::Scalar::all(255),2);
    cv::imshow("minili", strictFace);
    cv::waitKey(100000);
    */
    return strictFace;
}

/**
 * Guess strict face rectangle from eye centers data
 *
 * @return cv::Rect
 */
cv::Rect Aligner::strictRect() const
{
    size_t eyeDist = fabs(lEye.x-rEye.x);
    double eyeSize = eyeDist/2.;

    size_t lxStartGuess = (double)lEye.x - 1.5*eyeSize;
    size_t lxEndGuess = lxStartGuess + 5*eyeSize;
    size_t rxEndGuess = (double)rEye.x + 1.5*eyeSize;
    size_t rxStartGuess = rxEndGuess - 5*eyeSize;

    size_t xStartGuess = round((lxStartGuess+rxStartGuess)/2.);
    size_t xEndGuess = round((lxEndGuess+rxEndGuess)/2.);

    size_t yStartGuess = (double)lEye.y - 1.5*eyeSize;

    size_t faceSize = xEndGuess-xStartGuess;

    return cv::Rect(xStartGuess, yStartGuess, faceSize, faceSize);
}

/**
 * Update eye coords, convert them to relative after
 * face is cropped
 *
 * @param dx - Crop window offset, x and y
 * @param dy
 * @return void
 */
void Aligner::updateEyes(unsigned int dx, unsigned int dy)
{
    lEye.x -= dx;
    lEye.y -= dy;
    rEye.x -= dx;
    rEye.y -= dy;
}

/**
 * Rotate eyes n radians
 *
 * @param  angle
 * @return voide
 */
void Aligner::rotateEyes(double theta, double oX, double oY)
{
    theta = -theta;

    lEye.x -= oX;
    lEye.y -= oY;

    rEye.x -= oX;
    rEye.y -= oY;

    lEye = Aligner::ptrotate(lEye, theta);
    rEye = Aligner::ptrotate(rEye, theta);

    lEye.x += oX;
    lEye.y += oY;

    rEye.x += oX;
    rEye.y += oY;
}

/**
 * Rotate a single point
 *
 * @param  cv::Point2i - input point
 * @param  double      - rot angle in radians
 * @return cv::Point2i
 */
cv::Point2i Aligner::ptrotate(const cv::Point2i& src, double theta)
{
    cv::Point2i dst;
    dst.x = cos(theta)*src.x - sin(theta)*src.y;
    dst.y = sin(theta)*src.x + cos(theta)*src.y;

    return dst;
}

/**
 * Get de-rotation angle in RADIANS, the angle needed to rotate
 * the image so the face gets aligned.
 *
 * @return double
 */
double Aligner::rotAngle() const
{
    int xDist = lEye.x - rEye.x;
    int yDist = lEye.y - rEye.y;

    double theta = atan2(yDist, xDist);

    // avoid upside-down rotations
    if (theta > cvip::PI/2) theta = cvip::PI - theta;
    else if (theta < -cvip::PI/2) theta = cvip::PI + theta;
    if (yDist > 0)
        theta = -theta;

    return theta;
}

/**
 * Check rectangle boundaries and return the largest
 * rectangle you can.
 *
 * @return double
 */
int Aligner::maxPadding() const
{
    int maxL = r->x1;
    int maxR = im->width-r->x2-1;
    int maxU = r->y1;
    int maxB = im->height-r->y2-1;

    using std::min;

    // calculate the minimum rate that this box can extand
    return min(maxL, min(maxR, min(maxU, maxB)));
}
