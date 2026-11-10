#include "spl/image.hpp"
#include "spl/lbp.hpp"
#include "spl/random_forest.hpp"

#include <windows.h>
#include <gdiplus.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

class GdiPlusSession {
public:
    GdiPlusSession() {
        Gdiplus::GdiplusStartupInput input;
        const Gdiplus::Status status = Gdiplus::GdiplusStartup(&token_, &input, nullptr);
        if (status != Gdiplus::Ok) {
            throw std::runtime_error("Could not initialize Windows GDI+ (status " +
                                     std::to_string(static_cast<int>(status)) + ").");
        }
        started_ = true;
    }

    ~GdiPlusSession() {
        if (started_) Gdiplus::GdiplusShutdown(token_);
    }

    GdiPlusSession(const GdiPlusSession&) = delete;
    GdiPlusSession& operator=(const GdiPlusSession&) = delete;

private:
    ULONG_PTR token_ = 0;
    bool started_ = false;
};

std::size_t parseSize(const std::string& text, const std::string& label) {
    try {
        std::size_t consumed = 0;
        const unsigned long long value = std::stoull(text, &consumed);
        if (consumed != text.size() || value == 0) throw std::invalid_argument("invalid");
        return static_cast<std::size_t>(value);
    } catch (...) {
        throw std::invalid_argument(label + " must be a positive integer.");
    }
}

void printUsage() {
    std::cout <<
        "SPL face/non-face classifier (C++17, Windows GDI+, no OpenCV)\n\n"
        "Commands:\n"
        "  spl_rf extract <image.jpg> <features.csv> [tile_size]\n"
        "  spl_rf train <dataset_dir> <model_file> [tree_count] [tile_size] [criterion]\n"
        "  spl_rf predict <model_file> <image.jpg>\n"
        "  spl_rf evaluate <model_file> <test_dataset_dir>\n\n"
        "Dataset folders:\n"
        "  dataset_dir/face/      labeled face JPG/JPEG images\n"
        "  dataset_dir/non_face/  labeled non-face JPG/JPEG images\n\n"
        "Criteria: information_gain (default), gini, hellinger\n"
        "Default tile size: 64; default tree count: 100\n";
}

void exportFeatures(const std::filesystem::path& imagePath,
                    const std::filesystem::path& csvPath,
                    std::size_t tileSize) {
    const spl::GrayImage image = spl::readJpeg(imagePath);
    const spl::ImageFeatures features = spl::extractLbpAndMaxPool(image, tileSize);
    std::ofstream output(csvPath);
    if (!output) throw std::runtime_error("Could not create CSV file: " + csvPath.string());
    output << "row,tile_id,x,y";
    for (std::size_t i = 0; i < spl::kLbpFeatureCount; ++i) output << ",lbp_" << i;
    output << '\n' << std::setprecision(10);
    for (const spl::TileFeature& tile : features.tiles) {
        output << "tile," << tile.id << ',' << tile.x << ',' << tile.y;
        for (const double value : tile.values) output << ',' << value;
        output << '\n';
    }
    output << "max_pooled,-1,0,0";
    for (const double value : features.maxPooled) output << ',' << value;
    output << '\n';
    if (!output) throw std::runtime_error("Failed while writing CSV file: " + csvPath.string());

    std::cout << "Image size: " << image.width << 'x' << image.height << " pixels\n"
              << "Tiles: " << features.tiles.size() << " (" << tileSize << " pixels maximum per side)\n"
              << "Features per tile: " << spl::kLbpFeatureCount << " normalized LBP histogram bins\n"
              << "Saved tile vectors and max-pooled vector to: " << csvPath.string() << '\n';
}

void trainForest(const std::filesystem::path& datasetPath,
                 const std::filesystem::path& modelPath,
                 std::size_t treeCount,
                 std::size_t tileSize,
                 spl::SplitCriterion criterion) {
    std::cout << "Reading and describing labeled images...\n";
    spl::Dataset dataset = spl::loadLabeledDataset(datasetPath, tileSize);
    spl::ForestConfig config;
    config.treeCount = treeCount;
    config.criterion = criterion;
    spl::RandomForest forest(config);
    std::cout << "Training " << treeCount << " trees on " << dataset.samples.size()
              << " images (face=" << dataset.faceCount << ", non_face=" << dataset.nonFaceCount
              << "), feature dimension=" << spl::kLbpFeatureCount << "...\n";
    forest.train(dataset.samples, dataset.labels);
    forest.save(modelPath, tileSize);
    std::cout << "Saved model to: " << modelPath.string() << '\n'
              << "Split criterion: " << spl::criterionName(criterion) << '\n';
}

void predictImage(const std::filesystem::path& modelPath, const std::filesystem::path& imagePath) {
    std::size_t tileSize = 0;
    const spl::RandomForest forest = spl::RandomForest::load(modelPath, tileSize);
    const spl::GrayImage image = spl::readJpeg(imagePath);
    const spl::ImageFeatures features = spl::extractLbpAndMaxPool(image, tileSize);
    const auto probabilities = forest.predictProbabilities(features.maxPooled);
    const spl::Label prediction = probabilities[1] > probabilities[0] ? 1 : 0;
    std::cout << "Prediction: " << (prediction == 1 ? "FACE" : "NON-FACE") << '\n'
              << std::fixed << std::setprecision(4)
              << "Vote proportions: FACE=" << probabilities[1]
              << ", NON-FACE=" << probabilities[0] << '\n'
              << "Tiles processed: " << features.tiles.size() << '\n';
}

void evaluateForest(const std::filesystem::path& modelPath,
                    const std::filesystem::path& datasetPath) {
    std::size_t tileSize = 0;
    const spl::RandomForest forest = spl::RandomForest::load(modelPath, tileSize);
    const spl::Dataset dataset = spl::loadLabeledDataset(datasetPath, tileSize);
    const spl::EvaluationMetrics metrics = spl::evaluate(forest, dataset);
    const double total = static_cast<double>(dataset.samples.size());
    const double accuracy = static_cast<double>(metrics.truePositive + metrics.trueNegative) / total;
    const double precisionDenominator = static_cast<double>(metrics.truePositive + metrics.falsePositive);
    const double recallDenominator = static_cast<double>(metrics.truePositive + metrics.falseNegative);
    const double precision = precisionDenominator == 0.0 ? 0.0
        : static_cast<double>(metrics.truePositive) / precisionDenominator;
    const double recall = recallDenominator == 0.0 ? 0.0
        : static_cast<double>(metrics.truePositive) / recallDenominator;
    const double f1 = precision + recall == 0.0 ? 0.0 : 2.0 * precision * recall / (precision + recall);

    std::cout << std::fixed << std::setprecision(4)
              << "Evaluation images: " << dataset.samples.size() << '\n'
              << "Accuracy:  " << accuracy << '\n'
              << "Precision: " << precision << '\n'
              << "Recall:    " << recall << '\n'
              << "F1 score:  " << f1 << "\n\n"
              << "Confusion matrix (actual rows; predicted columns)\n"
              << "                 FACE  NON-FACE\n"
              << "Actual FACE      " << metrics.truePositive << "     " << metrics.falseNegative << '\n'
              << "Actual NON-FACE  " << metrics.falsePositive << "     " << metrics.trueNegative << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        GdiPlusSession gdiPlus;
        if (argc < 2) {
            printUsage();
            return 0;
        }
        const std::string command = argv[1];

        if (command == "extract") {
            if (argc < 4 || argc > 5) {
                printUsage();
                return 2;
            }
            const std::size_t tileSize = argc == 5 ? parseSize(argv[4], "tile_size") : 64;
            exportFeatures(argv[2], argv[3], tileSize);
        } else if (command == "train") {
            if (argc < 4 || argc > 7) {
                printUsage();
                return 2;
            }
            const std::size_t treeCount = argc >= 5 ? parseSize(argv[4], "tree_count") : 100;
            const std::size_t tileSize = argc >= 6 ? parseSize(argv[5], "tile_size") : 64;
            const spl::SplitCriterion criterion = argc >= 7
                ? spl::parseCriterion(argv[6])
                : spl::SplitCriterion::InformationGain;
            trainForest(argv[2], argv[3], treeCount, tileSize, criterion);
        } else if (command == "predict") {
            if (argc != 4) {
                printUsage();
                return 2;
            }
            predictImage(argv[2], argv[3]);
        } else if (command == "evaluate") {
            if (argc != 4) {
                printUsage();
                return 2;
            }
            evaluateForest(argv[2], argv[3]);
        } else if (command == "help" || command == "--help" || command == "-h") {
            printUsage();
        } else {
            printUsage();
            return 2;
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
