#pragma once

#include "ray_types.h"
#include <algorithm>
#include <vector>

namespace MyEngine {

// BVH build configuration
struct BVHBuildConfig {
    int max_leaf_primitives;
    int morton_bits;
    int seed;

    __host__ BVHBuildConfig() : max_leaf_primitives(4), morton_bits(10), seed(12345) {}
};

// BVH statistics
struct BVHStats {
    int node_count;
    int leaf_count;
    int max_depth;
    float total_primitives;
    float total_surface_area;

    __host__ BVHStats() : node_count(0), leaf_count(0), max_depth(0),
                          total_primitives(0), total_surface_area(0) {}
};

// GPU BVH traversal
class BVHTraverser {
public:
    __device__ BVHTraverser(const BVHNode* nodes, const int* triangle_indices,
                            const Triangle* triangles)
        : _nodes(nodes), _triangle_indices(triangle_indices), _triangles(triangles) {}

    __device__ bool traverse(const Ray& ray, HitRecord& hit, float t_max = 1e20f) const {
        hit.t = t_max;
        return traverseNode(0, ray, hit, t_max);
    }

    __device__ bool traverseShadow(const Ray& ray, float t_max = 1e20f) const {
        HitRecord temp;
        temp.t = t_max;
        return traverseNode(0, ray, temp, t_max);
    }

private:
    const BVHNode* _nodes;
    const int* _triangle_indices;
    const Triangle* _triangles;

    __device__ bool intersectTriangle(const Ray& ray, HitRecord& hit, int tri_idx) const {
        const Triangle& tri = _triangles[tri_idx];

        float3 e1 = vsub(tri.v1, tri.v0);
        float3 e2 = vsub(tri.v2, tri.v0);
        float3 h = cross(ray.direction, e2);
        float a = dot(e1, h);

        if (fabsf(a) < 1e-6f) return false;

        float f = 1.0f / a;
        float3 s = vsub(ray.origin, tri.v0);
        float u = f * dot(s, h);

        if (u < 0.0f || u > 1.0f) return false;

        float3 q = cross(s, e1);
        float v = f * dot(ray.direction, q);

        if (v < 0.0f || u + v > 1.0f) return false;

        float t = f * dot(e2, q);

        if (t < 0.0001f || t >= hit.t) return false;

        hit.t = t;
        hit.point = ray.at(t);
        hit.u = u;
        hit.v = v;
        hit.material_id = tri.material_id;

        float3 normal = normalize(cross(e1, e2));
        hit.setFaceNormal(ray, normal);

        return true;
    }

    __device__ bool traverseNode(int node_idx, const Ray& ray,
                                  HitRecord& hit, float t_max) const {
        const BVHNode& node = _nodes[node_idx];

        if (!node.bounding_box.hit(ray, 0.0001f, hit.t)) {
            return false;
        }

        if (node.is_leaf) {
            for (int i = 0; i < node.count; i++) {
                int tri_idx = _triangle_indices[node.start + i];
                intersectTriangle(ray, hit, tri_idx);
            }
            return hit.t < t_max;
        }

        bool hit_left = false, hit_right = false;
        HitRecord left_hit = hit, right_hit = hit;

        if (_nodes[node.left].bounding_box.hit(ray, 0.0001f, hit.t)) {
            hit_left = traverseNode(node.left, ray, left_hit, t_max);
            if (left_hit.t < hit.t) hit = left_hit;
        }

        if (_nodes[node.right].bounding_box.hit(ray, 0.0001f, hit.t)) {
            hit_right = traverseNode(node.right, ray, right_hit, t_max);
            if (right_hit.t < hit.t) hit = right_hit;
        }

        return hit_left || hit_right;
    }
};

// BVH builder
class BVHBuilder {
public:
    BVHBuilder() : _node_capacity(0), _triangle_count(0) {}

    bool build(const Triangle* triangles, int triangle_count,
               const BVHBuildConfig& config = BVHBuildConfig()) {
        _triangle_count = triangle_count;

        _node_capacity = triangle_count * 2;
        _nodes.resize(_node_capacity);
        _triangle_indices.resize(triangle_count);

        for (int i = 0; i < triangle_count; i++) {
            _nodes[i] = BVHNode();
            _nodes[i].bounding_box = triangles[i].getBoundingBox();
            _nodes[i].start = i;
            _nodes[i].count = 1;
            _nodes[i].is_leaf = true;
            _nodes[i].left = -1;
            _nodes[i].right = -1;
            _triangle_indices[i] = i;
        }

        int node_count = buildRecursive(0, triangle_count, triangles, 0);

        _nodes.resize(node_count);
        _stats.node_count = node_count;
        _stats.leaf_count = countLeaves(0);
        _stats.max_depth = calculateMaxDepth(0);
        _stats.total_primitives = triangle_count;

        return true;
    }

    const BVHNode* getNodes() const { return _nodes.data(); }
    int getNodeCount() const { return _nodes.size(); }
    const int* getTriangleIndices() const { return _triangle_indices.data(); }
    int getTriangleIndicesCount() const { return _triangle_indices.size(); }
    const BVHStats& getStats() const { return _stats; }

    size_t getMemoryUsage() const {
        return _nodes.size() * sizeof(BVHNode) + _triangle_indices.size() * sizeof(int);
    }

private:
    std::vector<BVHNode> _nodes;
    std::vector<int> _triangle_indices;
    BVHStats _stats;
    int _node_capacity;
    int _triangle_count;

    int buildRecursive(int node_idx, int count, const Triangle* triangles, int depth) {
        if (count <= 4 || depth > 32) {
            return node_idx + 1;
        }

        AABB bounds;
        for (int i = 0; i < count; i++) {
            AABB tri_box = triangles[_triangle_indices[node_idx + i]].getBoundingBox();
            if (i == 0) {
                bounds = tri_box;
            } else {
                bounds.min = fminv(bounds.min, tri_box.min);
                bounds.max = fmaxv(bounds.max, tri_box.max);
            }
        }

        float dx = bounds.max.x - bounds.min.x;
        float dy = bounds.max.y - bounds.min.y;
        float dz = bounds.max.z - bounds.min.z;

        int split_axis = 0;
        if (dx > dy && dx > dz) split_axis = 0;
        else if (dy > dz) split_axis = 1;
        else split_axis = 2;

        int mid = count / 2;
        std::nth_element(
            &_triangle_indices[node_idx],
            &_triangle_indices[node_idx + mid],
            &_triangle_indices[node_idx + count],
            [&](int a, int b) {
                float3 ca = triangles[a].centroid();
                float3 cb = triangles[b].centroid();
                if (split_axis == 0) return ca.x < cb.x;
                if (split_axis == 1) return ca.y < cb.y;
                return ca.z < cb.z;
            }
        );

        BVHNode& node = _nodes[node_idx];
        node.bounding_box = bounds;
        node.is_leaf = false;
        node.left = node_idx + 1;
        node.start = node_idx;
        node.count = count;

        int left_end = buildRecursive(node_idx + 1, mid, triangles, depth + 1);
        node.right = left_end;
        int right_end = buildRecursive(left_end, count - mid, triangles, depth + 1);

        return right_end;
    }

    int countLeaves(int node_idx) const {
        if (_nodes[node_idx].is_leaf) return 1;
        return countLeaves(_nodes[node_idx].left) + countLeaves(_nodes[node_idx].right);
    }

    int calculateMaxDepth(int node_idx) const {
        if (_nodes[node_idx].is_leaf) return 0;
        int left_depth = calculateMaxDepth(_nodes[node_idx].left);
        int right_depth = calculateMaxDepth(_nodes[node_idx].right);
        return 1 + (left_depth > right_depth ? left_depth : right_depth);
    }
};

} // namespace MyEngine
