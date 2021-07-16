#include <math.h>

#include <algorithm>

#include "openpifpaf/decoder/utils/nms_keypoints.hpp"


namespace openpifpaf {
namespace decoder {
namespace utils {


double NMSKeypoints::suppression = 0.0;
double NMSKeypoints::instance_threshold = 0.15;
double NMSKeypoints::keypoint_threshold = 0.15;


void NMSKeypoints::call(Occupancy* occupancy, std::vector<std::vector<Joint> >* annotations) {
    occupancy->clear();

    std::sort(
        annotations->begin(),
        annotations->end(),
        [&](const std::vector<Joint> & a, const std::vector<Joint> & b) {
            return (score->value(a) > score->value(b));
        }
    );

    int64_t n_occupancy = occupancy->occupancy.size(0);
    for (auto&& ann : *annotations) {
        TORCH_CHECK(n_occupancy <= int64_t(ann.size()),
                    "NMS occupancy map must be of same size or smaller as annotation");

        int64_t f = -1;
        for (Joint& joint : ann) {
            f++;
            if (f >= n_occupancy) break;
            if (joint.v == 0.0) continue;
            if (occupancy->get(f, joint.x, joint.y)) {
                joint.v *= suppression;
            } else {
                occupancy->set(f, joint.x, joint.y, joint.s);  // joint.s = 2 * sigma
            }
        }
    }

    // suppress below keypoint threshold
    for (auto&& ann : *annotations) {
        for (Joint& joint : ann) {
            if (joint.v > keypoint_threshold) continue;
            joint.v = 0.0;
        }
    }

    // remove annotations below instance threshold
    annotations->erase(
        std::remove_if(annotations->begin(), annotations->end(), [&](const std::vector<Joint>& ann) {
            return (score->value(ann) < instance_threshold);
        }),
        annotations->end()
    );

    std::sort(
        annotations->begin(),
        annotations->end(),
        [&](const std::vector<Joint> & a, const std::vector<Joint> & b) {
            return (score->value(a) > score->value(b));
        }
    );
}


}  // namespace utils
}  // namespace decoder
}  // namespace openpifpaf
