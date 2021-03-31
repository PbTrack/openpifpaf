#include <algorithm>
#include <math.h>

#include "openpifpaf/decoder/utils/cif_hr.hpp"


namespace openpifpaf {
namespace decoder {
namespace utils {


int64_t CifHr::neighbors = 16;
double CifHr::v_threshold = 0.1;


void CifHr::accumulate(const torch::Tensor& cif_field, int64_t stride, double min_scale, double factor) {
    auto cif_field_a = cif_field.accessor<double, 4>();
    double min_scale_f = min_scale / stride;

    double v, x, y, scale, sigma;
    for (int64_t f=0; f < cif_field_a.size(0); f++) {
        for (int64_t j=0; j < cif_field_a.size(2); j++) {
            for (int64_t i=0; i < cif_field_a.size(3); i++) {
                v = cif_field_a[f][0][j][i];
                if (v < v_threshold) continue;

                scale = cif_field_a[f][4][j][i];
                if (scale < min_scale_f) continue;

                x = cif_field_a[f][1][j][i] * stride;
                y = cif_field_a[f][2][j][i] * stride;
                sigma = fmax(1.0, 0.5 * scale * stride);

                // Occupancy covers 2sigma.
                // Restrict this accumulation to 1sigma so that seeds for the same joint
                // are properly suppressed.
                add_gauss(f, v / neighbors * factor, x, y, sigma);
            }
        }
    }
}


void CifHr::add_gauss(int64_t f, double v, double x, double y, double sigma, double truncate) {
    auto accumulated_a = accumulated.accessor<double, 3>();

    auto minx = std::clamp(int64_t(x - truncate * sigma), int64_t(0), accumulated_a.size(2) - 1);
    auto miny = std::clamp(int64_t(y - truncate * sigma), int64_t(0), accumulated_a.size(1) - 1);
    auto maxx = std::clamp(int64_t(x + truncate * sigma + 1), minx + 1, accumulated_a.size(2));
    auto maxy = std::clamp(int64_t(y + truncate * sigma + 1), miny + 1, accumulated_a.size(1));

    double sigma2 = sigma * sigma;
    double truncate2_sigma2 = truncate * truncate * sigma2;
    double deltax2, deltay2;
    double vv;
    for (int64_t xx=minx; xx < maxx; xx++) {
        deltax2 = (xx - x) * (xx - x);
        for (int64_t yy=miny; yy < maxy; yy++) {
            deltay2 = (yy - y) * (yy - y);

            if (deltax2 + deltay2 > truncate2_sigma2) continue;

            if (deltax2 < 0.25 && deltay2 < 0.25) {
                // this is the closest pixel
                vv = v;
            } else {
                vv = v * exp(-0.5 * (deltax2 + deltay2) / sigma2);
            }

            accumulated_a[f][yy][xx] += vv;
            accumulated_a[f][yy][xx] = fmin(accumulated_a[f][yy][xx], 1.0);
        }
    }
}


} // namespace utils
} // namespace decoder
} // namespace openpifpaf
