#include "doctest.h"
#include "math/HierarchicalMatrix.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

namespace {
using HMatrix = tndm::HierarchicalMatrix;
std::vector<HMatrix::Box> geometry(std::size_t n) {
    std::vector<HMatrix::Box> boxes(n);
    for (std::size_t i = 0; i < n; ++i) {
        double x = static_cast<double>(i);
        boxes[i] = {{x - 0.2, 0, 0}, {x + 0.2, 0.1, 0}};
    }
    std::mt19937 rng(17);
    std::shuffle(boxes.begin(), boxes.end(), rng);
    return boxes;
}
} // namespace

TEST_CASE("Hierarchical compression preserves ordering and bounds the full matrix error") {
    auto boxes = geometry(48);
    auto entry = [&](std::size_t i, std::size_t j) {
        double distance = boxes[i / 3].lo[0] - boxes[j / 2].lo[0];
        return (1.0 + 0.2 * (i % 3) - 0.1 * (j % 2)) / std::sqrt(0.01 + distance * distance);
    };
    for (double tolerance : {1e-3, 1e-6, 1e-9}) {
        HMatrix::Options options;
        options.leaf_facets = 4;
        options.tolerance = tolerance;
        // Force streamed large-block construction in one run as well.
        options.max_block_entries = tolerance == 1e-6 ? 1 : 262144;
        HMatrix matrix(boxes, boxes, 3, 2, options, entry);
        auto const& stats = matrix.statistics();
        CHECK(stats.low_rank_blocks > 0);
        CHECK(stats.dense_blocks > 0);
        CHECK(stats.stored_scalars < matrix.rows() * matrix.cols());
        std::vector<double> x(matrix.cols(), 0), y(matrix.rows());
        long double error = 0, norm = 0;
        for (std::size_t j = 0; j < matrix.cols(); ++j) {
            x[j] = 1;
            matrix.multiply(x.data(), y.data());
            x[j] = 0;
            for (std::size_t i = 0; i < matrix.rows(); ++i) {
                long double a = entry(i, j), difference = y[i] - a;
                error += difference * difference;
                norm += a * a;
            }
        }
        CHECK(std::sqrt(error / norm) <= tolerance);
        CHECK(std::sqrt(stats.error_squared / stats.norm_squared) <= tolerance);
    }
}

TEST_CASE("Rank limits fall back to dense storage without discarding interactions") {
    auto boxes = geometry(24);
    HMatrix::Options options;
    options.leaf_facets = 2;
    options.max_rank = 1;
    options.tolerance = 1e-12;
    auto entry = [](std::size_t i, std::size_t j) { return std::sin(0.731 * (i + 1) * (j + 1)); };
    HMatrix matrix(boxes, boxes, 1, 1, options, entry);
    CHECK(matrix.statistics().dense_blocks > 0);
    CHECK(matrix.statistics().max_rank <= 1);
    std::vector<double> x(matrix.cols()), y(matrix.rows());
    for (std::size_t j = 0; j < x.size(); ++j) {
        x[j] = std::cos(0.317 * j);
    }
    matrix.multiply(x.data(), y.data());
    for (std::size_t i = 0; i < y.size(); ++i) {
        double reference = 0;
        for (std::size_t j = 0; j < x.size(); ++j) {
            reference += entry(i, j) * x[j];
        }
        CHECK(y[i] == doctest::Approx(reference).epsilon(1e-11));
    }
}

TEST_CASE("Hierarchical zero blocks and empty row partitions") {
    auto boxes = geometry(32);
    HMatrix::Options options;
    options.leaf_facets = 2;
    auto zero = [](std::size_t, std::size_t) { return 0.0; };
    HMatrix matrix(boxes, boxes, 1, 1, options, zero);
    CHECK(matrix.statistics().norm_squared == 0);
    CHECK(matrix.statistics().low_rank_blocks > 0);
    std::vector<double> x(matrix.cols(), 1), y(matrix.rows(), 1);
    matrix.multiply(x.data(), y.data());
    for (double value : y) {
        CHECK(value == 0);
    }
    HMatrix empty({}, boxes, 1, 1, options, zero);
    empty.multiply(x.data(), nullptr);
    CHECK(empty.rows() == 0);
    CHECK(empty.cols() == boxes.size());
    options.tolerance = 0;
    CHECK_THROWS_AS(HMatrix(boxes, boxes, 1, 1, options, zero), std::invalid_argument);
    options.tolerance = 1e-6;
    auto invalid = [](std::size_t, std::size_t) {
        return std::numeric_limits<double>::quiet_NaN();
    };
    CHECK_THROWS_AS(HMatrix(boxes, boxes, 1, 1, options, invalid), std::runtime_error);
}
