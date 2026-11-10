#include "spl/image.hpp"

#include <windows.h>
#include <gdiplus.h>

#include <algorithm>
#include <cctype>
#include <limits>
#include <stdexcept>
#include <string>

namespace spl {
namespace {

std::string lowerExtension(const std::filesystem::path& path) {
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return extension;
}

}  // namespace

GrayImage readJpeg(const std::filesystem::path& path) {
    const std::string extension = lowerExtension(path);
    if (extension != ".jpg" && extension != ".jpeg") {
        throw std::runtime_error("Expected a .jpg or .jpeg image: " + path.string());
    }

    Gdiplus::Bitmap bitmap(path.c_str(), FALSE);
    const Gdiplus::Status imageStatus = bitmap.GetLastStatus();
    if (imageStatus != Gdiplus::Ok) {
        throw std::runtime_error("GDI+ could not read image (status " +
                                 std::to_string(static_cast<int>(imageStatus)) +"): " + path.string());
    }

    const UINT width = bitmap.GetWidth();
    const UINT height = bitmap.GetHeight();
    if (width == 0 || height == 0 ||
        static_cast<std::uint64_t>(width) * static_cast<std::uint64_t>(height) >
            static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error("Image dimensions are invalid or too large: " + path.string());
    }

    Gdiplus::Rect rectangle(0, 0, static_cast<INT>(width), static_cast<INT>(height));
    Gdiplus::BitmapData bitmapData{};
    const Gdiplus::Status lockStatus = bitmap.LockBits(
        &rectangle, Gdiplus::ImageLockModeRead, PixelFormat32bppARGB, &bitmapData);
    if (lockStatus != Gdiplus::Ok) {
        throw std::runtime_error("GDI+ could not access image pixels (status " +
                                 std::to_string(static_cast<int>(lockStatus)) + "): " + path.string());
    }

    GrayImage result;
    result.width = width;
    result.height = height;
    result.pixels.resize(static_cast<std::size_t>(width) * static_cast<std::size_t>(height));

    const auto* firstRow = static_cast<const std::uint8_t*>(bitmapData.Scan0);
    for (std::size_t y = 0; y < result.height; ++y) {
        const auto* row = firstRow + static_cast<std::ptrdiff_t>(y) * bitmapData.Stride;
        for (std::size_t x = 0; x < result.width; ++x) {
            // A 32bppARGB pixel is stored as blue, green, red, alpha on Windows.
            const std::uint8_t* pixel = row + x * 4;
            const unsigned blue = pixel[0];
            const unsigned green = pixel[1];
            const unsigned red = pixel[2];
            result.pixels[y * result.width + x] = static_cast<std::uint8_t>(
                (299U * red + 587U * green + 114U * blue + 500U) / 1000U);
        }
    }

    const Gdiplus::Status unlockStatus = bitmap.UnlockBits(&bitmapData);
    if (unlockStatus != Gdiplus::Ok) {
        throw std::runtime_error("GDI+ could not release image pixels (status " +
                                 std::to_string(static_cast<int>(unlockStatus)) + "): " + path.string());
    }
    return result;
}

}  // namespace spl
