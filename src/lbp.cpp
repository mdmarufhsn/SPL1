#include "spl/lbp.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace spl {
namespace {

std::vector<double> lbpHistogram(const GrayImage& image,
                                 std::size_t x0,
                                 std::size_t y0,
                                 std::size_t width,
                                 std::size_t height) {
    std::vector<double> histogram(kLbpFeatureCount, 0.0);
    if (width < 3 || height < 3) {
        return histogram;
    }

    std::size_t sampleCount = 0;
    for (std::size_t y = y0 + 1; y + 1 < y0 + height; ++y) {
        for (std::size_t x = x0 + 1; x + 1 < x0 + width; ++x) {
            const std::uint8_t center = image.at(x, y);
            std::uint8_t code = 0;

            // Clockwise neighbor order, starting at the top-left.
            if (image.at(x - 1, y - 1) >= center) code |= 1U << 7;
            if (image.at(x,     y - 1) >= center) code |= 1U << 6;
            if (image.at(x + 1, y - 1) >= center) code |= 1U << 5;
            if (image.at(x + 1, y)     >= center) code |= 1U << 4;
            if (image.at(x + 1, y + 1) >= center) code |= 1U << 3;
            if (image.at(x,     y + 1) >= center) code |= 1U << 2;
            if (image.at(x - 1, y + 1) >= center) code |= 1U << 1;
            if (image.at(x - 1, y)     >= center) code |= 1U;

            histogram[code] += 1.0;
            ++sampleCount;
        }
    }

    if (sampleCount != 0) {
        const double denominator = static_cast<double>(sampleCount);
        for (double& value : histogram) {
            value /= denominator;
        }
    }
    return histogram;
}

}  // namespace

ImageFeatures extractLbpAndMaxPool(const GrayImage& image, std::size_t tileSize) {
    if (tileSize < 3) {
        throw std::invalid_argument("Tile size must be at least 3 pixels.");
    }
    if (image.width == 0 || image.height == 0 || image.pixels.size() != image.width * image.height) {
        throw std::invalid_argument("Cannot extract features from an empty or invalid image.");
    }

    ImageFeatures result;
    result.maxPooled.assign(kLbpFeatureCount, 0.0);

    std::size_t tileId = 0;
    for (std::size_t y = 0; y < image.height; y += tileSize) {
        const std::size_t tileHeight = std::min(tileSize, image.height - y);
        for (std::size_t x = 0; x < image.width; x += tileSize) {
            const std::size_t tileWidth = std::min(tileSize, image.width - x);
            TileFeature tile;
            tile.id = tileId++;
            tile.x = x;
            tile.y = y;
            tile.values = lbpHistogram(image, x, y, tileWidth, tileHeight);
            for (std::size_t feature = 0; feature < kLbpFeatureCount; ++feature) {
                result.maxPooled[feature] = std::max(result.maxPooled[feature], tile.values[feature]);
            }
            result.tiles.push_back(std::move(tile));
        }
    }
    return result;
}

}  // namespace spl
