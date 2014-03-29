#ifndef ALIGNER_H
#define ALIGNER_H

#include "Image.h"

namespace cvip
{
    /**
     * Class used to align detected faces:
     * 1) Using two eyes for frontal view
     * 2) ... another method for profile view
     *
     * @author evangelos sariyanidi / sariyanidi at gmail dot com
     * @date may 2011
     */
    class Aligner
    {
    public:
        // constructor, use pointers only
        Aligner(const Image* _im, const DetectionRect* _r, const std::vector<DetectionRect>& eyes)
            : im(_im), r(_r) { computeEyeCenters(eyes); }

        // return registered and cropped face
        cv::Mat getAligned();

    private:
        // coordinates of eye centers
        cv::Point2i lEye, rEye;

        // face will be cropped, update eye coords
        void updateEyes(unsigned int dx, unsigned int dy);

        // get de-rotation angle in radians
        double rotAngle() const;

        // rotate eye coords
        void rotateEyes(double angle, double oX, double oY);

        // how much can this box expand
        int maxPadding() const;

        // compute lEye, rEye
        void computeEyeCenters(const std::vector<DetectionRect>& eyes);

        // rotate point
        static cv::Point2i ptrotate(const cv::Point2i& src, double theta);

        // compute strict rect from eye coords
        cv::Rect strictRect() const;

        // the whole input im
        const Image* im;

        // face rect within im
        const DetectionRect* r;
    };
}
#endif // ALIGNER_H
