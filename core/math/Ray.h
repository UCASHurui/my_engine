#pragma once

#include "Vector3.h"

namespace MyEngine {

// 射线
struct Ray {
    Vector3 origin;
    Vector3 direction;

    Ray() = default;
    Ray(const Vector3& origin, const Vector3& direction)
        : origin(origin), direction(direction.normalized()) {}

    // 获取射线上指定距离的点
    Vector3 get_point(float distance) const {
        return origin + direction * distance;
    }

    // 射线与平面相交
    bool intersect_plane(const Vector3& plane_normal, float plane_d, float& out_distance) const {
        float denom = direction.dot(plane_normal);
        if (std::abs(denom) < 0.0001f) return false;

        float t = -(origin.dot(plane_normal) + plane_d) / denom;
        if (t < 0) return false;

        out_distance = t;
        return true;
    }

    // 射线与球体相交
    bool intersect_sphere(const Vector3& sphere_center, float sphere_radius, float& out_distance) const {
        Vector3 oc = origin - sphere_center;
        float a = direction.dot(direction);
        float b = 2.0f * oc.dot(direction);
        float c = oc.dot(oc) - sphere_radius * sphere_radius;
        float discriminant = b * b - 4 * a * c;

        if (discriminant < 0) return false;

        float t = (-b - std::sqrt(discriminant)) / (2.0f * a);
        if (t < 0) t = (-b + std::sqrt(discriminant)) / (2.0f * a);
        if (t < 0) return false;

        out_distance = t;
        return true;
    }

    // 射线与AABB相交
    bool intersect_aabb(const Vector3& aabb_min, const Vector3& aabb_max, float& out_distance) const {
        float tmin = (aabb_min.x - origin.x) / direction.x;
        float tmax = (aabb_max.x - origin.x) / direction.x;

        if (tmin > tmax) std::swap(tmin, tmax);

        float tymin = (aabb_min.y - origin.y) / direction.y;
        float tymax = (aabb_max.y - origin.y) / direction.y;

        if (tymin > tymax) std::swap(tymin, tymax);

        if ((tmin > tymax) || (tymin > tmax)) return false;

        if (tymin > tmin) tmin = tymin;
        if (tymax < tmax) tmax = tymax;

        float tzmin = (aabb_min.z - origin.z) / direction.z;
        float tzmax = (aabb_max.z - origin.z) / direction.z;

        if (tzmin > tzmax) std::swap(tzmin, tzmax);

        if ((tmin > tzmax) || (tzmin > tmax)) return false;

        if (tzmin > tmin) tmin = tzmin;
        if (tzmax < tmax) tmax = tzmax;

        if (tmax < 0) return false;

        out_distance = tmin >= 0 ? tmin : tmax;
        return true;
    }
};

} // namespace MyEngine
