#include "spl/random_forest.hpp"

#include "spl/image.hpp"
#include "spl/lbp.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace spl {
namespace {

constexpr double kScoreEpsilon = 1e-12;

double entropy(std::size_t positives, std::size_t negatives) {
    const std::size_t total = positives + negatives;
    if (total == 0 || positives == 0 || negatives == 0) {
        return 0.0;
    }
    const double p = static_cast<double>(positives) / static_cast<double>(total);
    const double n = static_cast<double>(negatives) / static_cast<double>(total);
    return -p * std::log2(p) - n * std::log2(n);
}

double gini(std::size_t positives, std::size_t negatives) {
    const std::size_t total = positives + negatives;
    if (total == 0) {
        return 0.0;
    }
    const double p = static_cast<double>(positives) / static_cast<double>(total);
    const double n = static_cast<double>(negatives) / static_cast<double>(total);
    return 1.0 - p * p - n * n;
}

double splitScore(SplitCriterion criterion,
                  std::size_t parentPos,
                  std::size_t parentNeg,
                  std::size_t leftPos,
                  std::size_t leftNeg,
                  std::size_t rightPos,
                  std::size_t rightNeg) {
    const std::size_t parentCount = parentPos + parentNeg;
    const std::size_t leftCount = leftPos + leftNeg;
    const std::size_t rightCount = rightPos + rightNeg;

    if (criterion == SplitCriterion::Hellinger) {
        if (parentPos == 0 || parentNeg == 0) {
            return 0.0;
        }
        const double posLeft = static_cast<double>(leftPos) / static_cast<double>(parentPos);
        const double negLeft = static_cast<double>(leftNeg) / static_cast<double>(parentNeg);
        const double posRight = static_cast<double>(rightPos) / static_cast<double>(parentPos);
        const double negRight = static_cast<double>(rightNeg) / static_cast<double>(parentNeg);
        const double leftDelta = std::sqrt(posLeft) - std::sqrt(negLeft);
        const double rightDelta = std::sqrt(posRight) - std::sqrt(negRight);
        return 0.5 * (leftDelta * leftDelta + rightDelta * rightDelta);
    }

    const double parentImpurity = criterion == SplitCriterion::Gini
        ? gini(parentPos, parentNeg)
        : entropy(parentPos, parentNeg);
    const double leftImpurity = criterion == SplitCriterion::Gini
        ? gini(leftPos, leftNeg)
        : entropy(leftPos, leftNeg);
    const double rightImpurity = criterion == SplitCriterion::Gini
        ? gini(rightPos, rightNeg)
        : entropy(rightPos, rightNeg);
    const double weightedChildImpurity =
        static_cast<double>(leftCount) / static_cast<double>(parentCount) * leftImpurity +
        static_cast<double>(rightCount) / static_cast<double>(parentCount) * rightImpurity;
    return parentImpurity - weightedChildImpurity;
}

bool hasJpegExtension(const std::filesystem::path& path) {
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return extension == ".jpg" || extension == ".jpeg";
}

std::vector<std::filesystem::path> imagePaths(const std::filesystem::path& directory) {
    if (!std::filesystem::exists(directory) || !std::filesystem::is_directory(directory)) {
        throw std::runtime_error("Dataset class folder does not exist: " + directory.string());
    }
    std::vector<std::filesystem::path> paths;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(directory)) {
        if (entry.is_regular_file() && hasJpegExtension(entry.path())) {
            paths.push_back(entry.path());
        }
    }
    std::sort(paths.begin(), paths.end());
    return paths;
}

}  // namespace

std::string criterionName(SplitCriterion criterion) {
    switch (criterion) {
        case SplitCriterion::InformationGain: return "information_gain";
        case SplitCriterion::Gini: return "gini";
        case SplitCriterion::Hellinger: return "hellinger";
    }
    throw std::invalid_argument("Unknown split criterion.");
}

SplitCriterion parseCriterion(const std::string& name) {
    if (name == "information_gain" || name == "entropy") return SplitCriterion::InformationGain;
    if (name == "gini") return SplitCriterion::Gini;
    if (name == "hellinger") return SplitCriterion::Hellinger;
    throw std::invalid_argument("Criterion must be information_gain, gini, or hellinger.");
}

RandomForest::RandomForest(ForestConfig config) : config_(config) {
    if (config_.treeCount == 0) {
        throw std::invalid_argument("Random Forest must contain at least one tree.");
    }
    if (config_.maxDepth == 0) {
        throw std::invalid_argument("Maximum tree depth must be at least 1.");
    }
    if (config_.minSamplesLeaf == 0) {
        throw std::invalid_argument("Minimum samples per leaf must be at least 1.");
    }
}

void RandomForest::train(const std::vector<FeatureVector>& samples, const std::vector<Label>& labels) {
    if (samples.empty() || samples.size() != labels.size()) {
        throw std::invalid_argument("Training samples and labels must be non-empty and have matching sizes.");
    }
    featureCount_ = samples.front().size();
    if (featureCount_ == 0) {
        throw std::invalid_argument("Training feature vectors cannot be empty.");
    }
    for (std::size_t i = 0; i < samples.size(); ++i) {
        if (samples[i].size() != featureCount_) {
            throw std::invalid_argument("All training feature vectors must have the same dimension.");
        }
        if (labels[i] > 1) {
            throw std::invalid_argument("Labels must be 0 (NON-FACE) or 1 (FACE).");
        }
    }

    std::mt19937 random(config_.seed);
    const std::size_t featuresPerSplit = config_.maxFeaturesPerSplit == 0
        ? std::max<std::size_t>(1, static_cast<std::size_t>(std::sqrt(static_cast<double>(featureCount_))))
        : std::min(config_.maxFeaturesPerSplit, featureCount_);
    trees_.clear();
    trees_.reserve(config_.treeCount);

    for (std::size_t treeIndex = 0; treeIndex < config_.treeCount; ++treeIndex) {
        Tree tree;
        std::vector<std::size_t> bootstrap(samples.size());
        std::uniform_int_distribution<std::size_t> pickSample(0, samples.size() - 1);
        for (std::size_t& index : bootstrap) {
            index = pickSample(random);
        }

        std::function<int(const std::vector<std::size_t>&, std::size_t)> buildNode;
        buildNode = [&](const std::vector<std::size_t>& indices, std::size_t depth) -> int {
            std::size_t positives = 0;
            for (const std::size_t index : indices) {
                positives += labels[index] == 1 ? 1U : 0U;
            }
            const std::size_t negatives = indices.size() - positives;
            const Label majority = positives > negatives ? 1 : 0;
            const int nodeIndex = static_cast<int>(tree.nodes.size());
            tree.nodes.emplace_back();
            tree.nodes[static_cast<std::size_t>(nodeIndex)].prediction = majority;

            if (depth >= config_.maxDepth || indices.size() < 2 * config_.minSamplesLeaf ||
                positives == 0 || negatives == 0) {
                return nodeIndex;
            }

            std::vector<std::size_t> featureIndices(featureCount_);
            std::iota(featureIndices.begin(), featureIndices.end(), 0);
            std::shuffle(featureIndices.begin(), featureIndices.end(), random);
            featureIndices.resize(featuresPerSplit);

            double bestScore = 0.0;
            std::size_t bestFeature = 0;
            double bestThreshold = 0.0;
            bool foundSplit = false;

            for (const std::size_t feature : featureIndices) {
                std::vector<std::pair<double, Label>> ordered;
                ordered.reserve(indices.size());
                for (const std::size_t index : indices) {
                    ordered.emplace_back(samples[index][feature], labels[index]);
                }
                std::sort(ordered.begin(), ordered.end(), [](const auto& left, const auto& right) {
                    return left.first < right.first;
                });

                std::size_t leftPos = 0;
                std::size_t leftNeg = 0;
                for (std::size_t split = 1; split < ordered.size(); ++split) {
                    if (ordered[split - 1].second == 1) ++leftPos;
                    else ++leftNeg;
                    if (ordered[split - 1].first == ordered[split].first) continue;

                    const std::size_t leftCount = split;
                    const std::size_t rightCount = ordered.size() - split;
                    if (leftCount < config_.minSamplesLeaf || rightCount < config_.minSamplesLeaf) continue;
                    const std::size_t rightPos = positives - leftPos;
                    const std::size_t rightNeg = negatives - leftNeg;
                    const double score = splitScore(config_.criterion, positives, negatives,
                                                    leftPos, leftNeg, rightPos, rightNeg);
                    if (score > bestScore + kScoreEpsilon) {
                        bestScore = score;
                        bestFeature = feature;
                        bestThreshold = ordered[split - 1].first +
                            (ordered[split].first - ordered[split - 1].first) * 0.5;
                        foundSplit = true;
                    }
                }
            }

            if (!foundSplit) {
                return nodeIndex;
            }

            std::vector<std::size_t> leftSamples;
            std::vector<std::size_t> rightSamples;
            leftSamples.reserve(indices.size());
            rightSamples.reserve(indices.size());
            for (const std::size_t index : indices) {
                if (samples[index][bestFeature] <= bestThreshold) leftSamples.push_back(index);
                else rightSamples.push_back(index);
            }
            if (leftSamples.empty() || rightSamples.empty()) {
                return nodeIndex;
            }

            const int leftNode = buildNode(leftSamples, depth + 1);
            const int rightNode = buildNode(rightSamples, depth + 1);
            Node& node = tree.nodes[static_cast<std::size_t>(nodeIndex)];
            node.feature = static_cast<int>(bestFeature);
            node.threshold = bestThreshold;
            node.left = leftNode;
            node.right = rightNode;
            return nodeIndex;
        };

        buildNode(bootstrap, 0);
        trees_.push_back(std::move(tree));
    }
}

Label RandomForest::predictTree(const Tree& tree, const FeatureVector& sample) const {
    if (tree.nodes.empty()) {
        throw std::runtime_error("Random Forest contains an empty tree.");
    }
    int nodeIndex = 0;
    while (true) {
        const Node& node = tree.nodes.at(static_cast<std::size_t>(nodeIndex));
        if (node.feature < 0) {
            return node.prediction;
        }
        if (static_cast<std::size_t>(node.feature) >= sample.size()) {
            throw std::invalid_argument("Prediction feature vector has the wrong dimension.");
        }
        nodeIndex = sample[static_cast<std::size_t>(node.feature)] <= node.threshold ? node.left : node.right;
        if (nodeIndex < 0 || static_cast<std::size_t>(nodeIndex) >= tree.nodes.size()) {
            throw std::runtime_error("Random Forest model contains an invalid child node.");
        }
    }
}

std::vector<double> RandomForest::predictProbabilities(const FeatureVector& sample) const {
    if (trees_.empty()) {
        throw std::runtime_error("Train or load a Random Forest before prediction.");
    }
    if (sample.size() != featureCount_) {
        throw std::invalid_argument("Prediction feature vector has the wrong dimension.");
    }
    std::size_t votesFace = 0;
    for (const Tree& tree : trees_) {
        votesFace += predictTree(tree, sample) == 1 ? 1U : 0U;
    }
    const double total = static_cast<double>(trees_.size());
    const double faceProbability = static_cast<double>(votesFace) / total;
    return {1.0 - faceProbability, faceProbability};
}

Label RandomForest::predict(const FeatureVector& sample) const {
    const auto probabilities = predictProbabilities(sample);
    return probabilities[1] > probabilities[0] ? 1 : 0;
}

void RandomForest::save(const std::filesystem::path& path, std::size_t tileSize) const {
    if (trees_.empty() || featureCount_ == 0) {
        throw std::runtime_error("Cannot save an untrained Random Forest.");
    }
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("Could not create model file: " + path.string());
    }
    output << "SPL_RF_MODEL 1\n";
    output << "tile_size " << tileSize << '\n';
    output << "feature_count " << featureCount_ << '\n';
    output << "criterion " << criterionName(config_.criterion) << '\n';
    output << "tree_count " << trees_.size() << '\n';
    output << std::setprecision(17);
    for (std::size_t treeIndex = 0; treeIndex < trees_.size(); ++treeIndex) {
        const Tree& tree = trees_[treeIndex];
        output << "tree " << treeIndex << ' ' << tree.nodes.size() << '\n';
        for (const Node& node : tree.nodes) {
            output << node.feature << ' ' << node.threshold << ' ' << node.left << ' '
                   << node.right << ' ' << static_cast<unsigned>(node.prediction) << '\n';
        }
    }
    if (!output) {
        throw std::runtime_error("Failed while writing model file: " + path.string());
    }
}

RandomForest RandomForest::load(const std::filesystem::path& path, std::size_t& tileSize) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Could not open model file: " + path.string());
    }
    std::string marker;
    int version = 0;
    if (!(input >> marker >> version) || marker != "SPL_RF_MODEL" || version != 1) {
        throw std::runtime_error("Unsupported or invalid model file: " + path.string());
    }

    std::size_t featureCount = 0;
    std::size_t treeCount = 0;
    std::string criterion;
    std::string key;
    if (!(input >> key >> tileSize) || key != "tile_size" || tileSize < 3 ||
        !(input >> key >> featureCount) || key != "feature_count" || featureCount == 0 ||
        !(input >> key >> criterion) || key != "criterion" ||
        !(input >> key >> treeCount) || key != "tree_count" || treeCount == 0) {
        throw std::runtime_error("Model header is incomplete or invalid: " + path.string());
    }

    ForestConfig config;
    config.treeCount = treeCount;
    config.criterion = parseCriterion(criterion);
    RandomForest forest(config);
    forest.featureCount_ = featureCount;
    forest.trees_.reserve(treeCount);

    for (std::size_t treeIndex = 0; treeIndex < treeCount; ++treeIndex) {
        std::size_t storedIndex = 0;
        std::size_t nodeCount = 0;
        if (!(input >> key >> storedIndex >> nodeCount) || key != "tree" ||
            storedIndex != treeIndex || nodeCount == 0) {
            throw std::runtime_error("Invalid tree header in model file: " + path.string());
        }
        Tree tree;
        tree.nodes.resize(nodeCount);
        for (Node& node : tree.nodes) {
            unsigned prediction = 0;
            if (!(input >> node.feature >> node.threshold >> node.left >> node.right >> prediction) || prediction > 1) {
                throw std::runtime_error("Invalid tree node in model file: " + path.string());
            }
            node.prediction = static_cast<Label>(prediction);
            if (node.feature >= 0 && static_cast<std::size_t>(node.feature) >= featureCount) {
                throw std::runtime_error("Tree node feature index exceeds model feature count.");
            }
            if (node.feature >= 0 &&
                (node.left < 0 || node.right < 0 ||
                 static_cast<std::size_t>(node.left) >= nodeCount ||
                 static_cast<std::size_t>(node.right) >= nodeCount)) {
                throw std::runtime_error("Tree node has invalid child indices.");
            }
        }
        forest.trees_.push_back(std::move(tree));
    }
    return forest;
}

Dataset loadLabeledDataset(const std::filesystem::path& root, std::size_t tileSize) {
    Dataset dataset;
    const auto faceDirectory = root / "face";
    const auto nonFaceDirectory = root / "non_face";
    const auto facePaths = imagePaths(faceDirectory);
    const auto nonFacePaths = imagePaths(nonFaceDirectory);

    const auto addClass = [&](const std::vector<std::filesystem::path>& paths, Label label) {
        for (const auto& path : paths) {
            GrayImage image = readJpeg(path);
            ImageFeatures features = extractLbpAndMaxPool(image, tileSize);
            dataset.samples.push_back(std::move(features.maxPooled));
            dataset.labels.push_back(label);
            if (label == 1) ++dataset.faceCount;
            else ++dataset.nonFaceCount;
        }
    };
    addClass(facePaths, 1);
    addClass(nonFacePaths, 0);

    if (dataset.samples.empty()) {
        throw std::runtime_error("No .jpg or .jpeg files were found under the class folders.");
    }
    if (dataset.faceCount == 0 || dataset.nonFaceCount == 0) {
        throw std::runtime_error("Training/evaluation needs images in both face and non_face folders.");
    }
    return dataset;
}

EvaluationMetrics evaluate(const RandomForest& forest, const Dataset& dataset) {
    if (dataset.samples.size() != dataset.labels.size() || dataset.samples.empty()) {
        throw std::invalid_argument("Evaluation dataset is empty or malformed.");
    }
    EvaluationMetrics metrics;
    for (std::size_t i = 0; i < dataset.samples.size(); ++i) {
        const Label predicted = forest.predict(dataset.samples[i]);
        const Label actual = dataset.labels[i];
        if (actual == 1 && predicted == 1) ++metrics.truePositive;
        else if (actual == 0 && predicted == 0) ++metrics.trueNegative;
        else if (actual == 0 && predicted == 1) ++metrics.falsePositive;
        else ++metrics.falseNegative;
    }
    return metrics;
}

}  // namespace spl
