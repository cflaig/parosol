#ifndef TEMPORAL_DISPLACEMENT_CACHE_H
#define TEMPORAL_DISPLACEMENT_CACHE_H

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

// Process-local host storage, bounded to two load cases. No device pointers,
// matrices or preconditioners survive a request. Exact keys avoid reusing a
// vector in a different nodal ordering. Material values deliberately are not
// part of the key: each new stiffness matrix still gets a complete PCG solve.
template <typename Real>
class TemporalDisplacementCache {
public:
    struct Signature {
        std::vector<std::uint64_t> nodes, fixed_indices, loaded_indices;
        std::vector<Real> fixed_values, loaded_values;
        std::array<std::uint64_t, 3> dimensions;
        std::array<double, 3> spacing;
        std::size_t vector_size = 0;
        bool switching = false;
        bool operator==(const Signature& other) const {
            return vector_size == other.vector_size && switching == other.switching
                && dimensions == other.dimensions && spacing == other.spacing
                && nodes == other.nodes && fixed_indices == other.fixed_indices
                && fixed_values == other.fixed_values
                && loaded_indices == other.loaded_indices && loaded_values == other.loaded_values;
        }
        std::size_t bytes() const {
            return (nodes.capacity() + fixed_indices.capacity() + loaded_indices.capacity()) * sizeof(std::uint64_t)
                + (fixed_values.capacity() + loaded_values.capacity()) * sizeof(Real);
        }
    };
    struct Entry { Signature signature; std::vector<Real> displacement; std::uint64_t age; };

    void clear() { entries_.clear(); age_ = 0; }

    Entry* find(const Signature& signature) {
        for (auto& entry : entries_) {
            if (entry.signature == signature) { entry.age = ++age_; return &entry; }
        }
        return nullptr;
    }
    bool store(Signature signature, std::vector<Real> displacement) {
        if (displacement.size() != signature.vector_size ||
            !std::all_of(displacement.begin(), displacement.end(), [](Real x) { return std::isfinite(x); }))
            return false;
        Entry* slot = find(signature);
        if (!slot) {
            if (entries_.size() < 2) {
                entries_.push_back(Entry{});
                slot = &entries_.back();
            } else {
                slot = &*std::min_element(entries_.begin(), entries_.end(),
                    [](const Entry& a, const Entry& b) { return a.age < b.age; });
            }
        }
        *slot = Entry{std::move(signature), std::move(displacement), ++age_};
        return true;
    }
    std::size_t bytes() const {
        std::size_t result = 0;
        for (const auto& entry : entries_)
            result += entry.signature.bytes() + entry.displacement.capacity() * sizeof(Real);
        return result;
    }
private:
    std::vector<Entry> entries_;
    std::uint64_t age_ = 0;
};
#endif
