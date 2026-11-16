#ifndef HIERARCHICALMATRIX_20261116_H
#define HIERARCHICALMATRIX_20261116_H

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tndm {

// CPU prototype: geometric block cluster tree and fully pivoted cross approximation.
// Input and product use the original (unpermuted) facet/component ordering.
class HierarchicalMatrix {
public:
    struct Box {
        std::array<double, 3> lo{}, hi{};
    };
    struct Options {
        double tolerance = 1e-6; // Relative Frobenius tolerance in each compressed block.
        double eta = 1.0;        // max(diameters) <= eta * distance.
        std::size_t leaf_facets = 16;
        std::size_t max_rank = 32;
        std::size_t max_block_entries = 262144; // Bound temporary residual storage.
    };
    struct Statistics {
        std::size_t dense_blocks = 0, low_rank_blocks = 0, max_rank = 0;
        std::size_t stored_scalars = 0;
        long double norm_squared = 0, error_squared = 0;
    };

    // entry(i,j) accesses a locally owned row and a global column.
    template <typename Entry>
    HierarchicalMatrix(std::vector<Box> const& rows, std::vector<Box> const& cols,
                       std::size_t row_block_size, std::size_t col_block_size, Options options,
                       Entry entry)
        : rb_(row_block_size), cb_(col_block_size), options_(options) {
        if (!(options.tolerance > 0 && options.tolerance < 1) || !std::isfinite(options.eta) ||
            options.eta <= 0 || !options.leaf_facets || !options.max_rank ||
            !options.max_block_entries || !rb_ || !cb_) {
            throw std::invalid_argument("Invalid hierarchical matrix options");
        }
        row_order_.resize(rows.size());
        col_order_.resize(cols.size());
        std::iota(row_order_.begin(), row_order_.end(), 0);
        std::iota(col_order_.begin(), col_order_.end(), 0);
        if (!rows.empty() && !cols.empty()) {
            cluster(rows, row_order_, row_tree_, 0, rows.size());
            cluster(cols, col_order_, col_tree_, 0, cols.size());
            build(0, 0, entry);
        }
    }

    Statistics const& statistics() const { return stats_; }
    std::size_t rows() const { return row_order_.size() * rb_; }
    std::size_t cols() const { return col_order_.size() * cb_; }

    void multiply(double const* x, double* y) const {
        std::vector<double> ordered_x(cols()), ordered_y(rows());
        multiply(x, y, ordered_x.data(), ordered_y.data());
    }

    // Reusable work arrays avoid allocation in the time-stepping path.
    void multiply(double const* x, double* y, double* ordered_x, double* ordered_y) const {
        for (std::size_t j = 0; j < col_order_.size(); ++j) {
            for (std::size_t c = 0; c < cb_; ++c) {
                ordered_x[j * cb_ + c] = x[col_order_[j] * cb_ + c];
            }
        }
        if (rows()) {
            std::fill(ordered_y, ordered_y + rows(), 0.0);
        }
        for (auto const& block : blocks_) {
            auto const& r = row_tree_[block.row];
            auto const& c = col_tree_[block.col];
            auto m = (r.end - r.begin) * rb_, n = (c.end - c.begin) * cb_;
            if (block.rank == 0) {
                for (std::size_t j = 0; j < n; ++j) {
                    double v = ordered_x[c.begin * cb_ + j];
                    for (std::size_t i = 0; i < m; ++i) {
                        ordered_y[r.begin * rb_ + i] += block.u[i + m * j] * v;
                    }
                }
            } else {
                for (std::size_t k = 0; k < block.rank; ++k) {
                    double v = 0;
                    for (std::size_t j = 0; j < n; ++j) {
                        v += block.v[j + n * k] * ordered_x[c.begin * cb_ + j];
                    }
                    for (std::size_t i = 0; i < m; ++i) {
                        ordered_y[r.begin * rb_ + i] += block.u[i + m * k] * v;
                    }
                }
            }
        }
        for (std::size_t i = 0; i < row_order_.size(); ++i) {
            for (std::size_t c = 0; c < rb_; ++c) {
                y[row_order_[i] * rb_ + c] = ordered_y[i * rb_ + c];
            }
        }
    }

private:
    struct Cluster {
        std::size_t begin, end;
        Box box;
        std::size_t left = 0, right = 0;
        bool leaf() const { return left == 0; }
    };
    struct Block {
        std::size_t row, col, rank;
        std::vector<double> u, v;
    };

    std::size_t cluster(std::vector<Box> const& boxes, std::vector<std::size_t>& order,
                        std::vector<Cluster>& tree, std::size_t begin, std::size_t end) {
        Box box = boxes[order[begin]];
        for (std::size_t i = begin; i < end; ++i) {
            for (std::size_t d = 0; d < 3; ++d) {
                box.lo[d] = std::min(box.lo[d], boxes[order[i]].lo[d]);
                box.hi[d] = std::max(box.hi[d], boxes[order[i]].hi[d]);
            }
        }
        auto id = tree.size();
        tree.push_back({begin, end, box});
        if (end - begin > options_.leaf_facets) {
            std::size_t axis = 0;
            for (std::size_t d = 1; d < 3; ++d) {
                if (box.hi[d] - box.lo[d] > box.hi[axis] - box.lo[axis]) {
                    axis = d;
                }
            }
            auto middle = begin + (end - begin) / 2;
            std::nth_element(order.begin() + begin, order.begin() + middle, order.begin() + end,
                             [&](auto a, auto b) {
                                 double x = boxes[a].lo[axis] + boxes[a].hi[axis];
                                 double y = boxes[b].lo[axis] + boxes[b].hi[axis];
                                 return x == y ? a < b : x < y;
                             });
            auto left = cluster(boxes, order, tree, begin, middle);
            auto right = cluster(boxes, order, tree, middle, end);
            tree[id].left = left;
            tree[id].right = right;
        }
        return id;
    }

    bool separated(Box const& a, Box const& b) const {
        double da = 0, db = 0, distance = 0;
        for (std::size_t d = 0; d < 3; ++d) {
            da += (a.hi[d] - a.lo[d]) * (a.hi[d] - a.lo[d]);
            db += (b.hi[d] - b.lo[d]) * (b.hi[d] - b.lo[d]);
            double gap = std::max({0.0, a.lo[d] - b.hi[d], b.lo[d] - a.hi[d]});
            distance += gap * gap;
        }
        return distance > 0 && std::max(da, db) <= options_.eta * options_.eta * distance;
    }

    std::size_t row_index(Cluster const& r, std::size_t i) const {
        return row_order_[r.begin + i / rb_] * rb_ + i % rb_;
    }
    std::size_t col_index(Cluster const& c, std::size_t j) const {
        return col_order_[c.begin + j / cb_] * cb_ + j % cb_;
    }
    static long double squared_norm(std::vector<double> const& a) {
        long double result = 0;
        for (double x : a) {
            result += static_cast<long double>(x) * x;
        }
        return result;
    }

    template <typename Entry> void build(std::size_t row, std::size_t col, Entry const& entry) {
        auto const& r = row_tree_[row];
        auto const& c = col_tree_[col];
        auto m = (r.end - r.begin) * rb_, n = (c.end - c.begin) * cb_;
        bool admissible = separated(r.box, c.box);
        bool terminal = r.leaf() && c.leaf();
        // Large far-field blocks use streamed residuals, preserving the hierarchy
        // without allocating a dense temporary or imposing a fixed block-size cap.
        bool large = m > options_.max_block_entries / n;
        if (admissible && large && compress_streamed(row, col, m, n, entry)) {
            return;
        }
        if ((admissible && !large) || terminal) {
            std::vector<double> a(m * n);
            for (std::size_t j = 0; j < n; ++j) {
                for (std::size_t i = 0; i < m; ++i) {
                    double value = entry(row_index(r, i), col_index(c, j));
                    if (!std::isfinite(value)) {
                        throw std::runtime_error("Non-finite Green matrix entry");
                    }
                    a[i + m * j] = value;
                }
            }
            auto norm = squared_norm(a);
            if (admissible && compress(row, col, m, n, a, norm)) {
                return;
            }
            if (terminal) {
                stats_.norm_squared += norm;
                stats_.stored_scalars += a.size();
                ++stats_.dense_blocks;
                blocks_.push_back({row, col, 0, std::move(a), {}});
                return;
            }
        }
        // Split the larger side. Failed compression is refined, never silently truncated.
        if (!r.leaf() && (c.leaf() || m >= n)) {
            build(r.left, col, entry);
            build(r.right, col, entry);
        } else {
            build(row, c.left, entry);
            build(row, c.right, entry);
        }
    }

    template <typename Entry>
    bool compress_streamed(std::size_t row, std::size_t col, std::size_t m, std::size_t n,
                           Entry const& entry) {
        auto const& r = row_tree_[row];
        auto const& c = col_tree_[col];
        Block block{row, col, 0, {}, {}};
        auto residual = [&](std::size_t i, std::size_t j) {
            long double value = entry(row_index(r, i), col_index(c, j));
            if (!std::isfinite(value)) {
                throw std::runtime_error("Non-finite Green matrix entry");
            }
            for (std::size_t k = 0; k < block.rank; ++k) {
                value -= static_cast<long double>(block.u[i + m * k]) * block.v[j + n * k];
            }
            return value;
        };
        long double norm = 0, error = 0;
        auto limit = std::min(options_.max_rank, (m * n - 1) / (m + n));
        for (;;) {
            long double pivot = 0;
            std::size_t pi = 0, pj = 0;
            error = 0;
            for (std::size_t j = 0; j < n; ++j) {
                for (std::size_t i = 0; i < m; ++i) {
                    auto value = residual(i, j);
                    error += value * value;
                    if (std::abs(value) > std::abs(pivot)) {
                        pivot = value;
                        pi = i;
                        pj = j;
                    }
                }
            }
            if (!block.rank) {
                norm = error;
            }
            if (error <= options_.tolerance * options_.tolerance * norm || block.rank == limit) {
                break;
            }
            // Evaluate both factors before changing rank: the residual references them.
            for (std::size_t i = 0; i < m; ++i) {
                block.u.push_back(static_cast<double>(residual(i, pj)));
            }
            for (std::size_t j = 0; j < n; ++j) {
                block.v.push_back(static_cast<double>(residual(pi, j) / pivot));
            }
            ++block.rank;
        }
        if (!block.rank && norm == 0 && limit > 0) {
            block.rank = 1;
            block.u.assign(m, 0.0);
            block.v.assign(n, 0.0);
        }
        if (!block.rank || !std::isfinite(error) ||
            error > options_.tolerance * options_.tolerance * norm) {
            return false;
        }
        stats_.norm_squared += norm;
        stats_.error_squared += error;
        stats_.stored_scalars += block.u.size() + block.v.size();
        stats_.max_rank = std::max(stats_.max_rank, block.rank);
        ++stats_.low_rank_blocks;
        blocks_.push_back(std::move(block));
        return true;
    }

    bool compress(std::size_t row, std::size_t col, std::size_t m, std::size_t n,
                  std::vector<double> const& a, long double norm) {
        auto residual = a;
        Block block{row, col, 0, {}, {}};
        long double error = norm;
        auto target = options_.tolerance * options_.tolerance * norm;
        // Only accept factors that occupy less storage than a dense block.
        auto limit = std::min(options_.max_rank, (m * n - 1) / (m + n));
        while (error > target && block.rank < limit) {
            auto pivot = std::max_element(residual.begin(), residual.end(), [](double a, double b) {
                return std::abs(a) < std::abs(b);
            });
            if (*pivot == 0.0) {
                break;
            }
            auto index = static_cast<std::size_t>(pivot - residual.begin());
            auto pi = index % m, pj = index / m, k = block.rank;
            double value = *pivot;
            for (std::size_t i = 0; i < m; ++i) {
                block.u.push_back(residual[i + m * pj]);
            }
            for (std::size_t j = 0; j < n; ++j) {
                block.v.push_back(residual[pi + m * j] / value);
            }
            for (std::size_t j = 0; j < n; ++j) {
                for (std::size_t i = 0; i < m; ++i) {
                    residual[i + m * j] -= block.u[i + m * k] * block.v[j + n * k];
                }
            }
            ++block.rank;
            error = squared_norm(residual);
        }
        // A zero block has an exact rank-one representation with zero factors.
        if (norm == 0 && limit > 0) {
            block.rank = 1;
            block.u.assign(m, 0.0);
            block.v.assign(n, 0.0);
        }
        if (!block.rank || error > target || !std::isfinite(error)) {
            return false;
        }
        // Verify the stored representation, rather than just the updated ACA residual.
        error = 0;
        for (std::size_t j = 0; j < n; ++j) {
            for (std::size_t i = 0; i < m; ++i) {
                long double value = a[i + m * j];
                for (std::size_t k = 0; k < block.rank; ++k) {
                    value -= static_cast<long double>(block.u[i + m * k]) * block.v[j + n * k];
                }
                error += value * value;
            }
        }
        if (error > target || !std::isfinite(error)) {
            return false;
        }
        stats_.norm_squared += norm;
        stats_.error_squared += error;
        stats_.stored_scalars += block.u.size() + block.v.size();
        stats_.max_rank = std::max(stats_.max_rank, block.rank);
        ++stats_.low_rank_blocks;
        blocks_.push_back(std::move(block));
        return true;
    }

    std::size_t rb_, cb_;
    Options options_;
    Statistics stats_;
    std::vector<std::size_t> row_order_, col_order_;
    std::vector<Cluster> row_tree_, col_tree_;
    std::vector<Block> blocks_;
};

} // namespace tndm
#endif
