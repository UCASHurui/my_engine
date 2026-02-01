#pragma once

#include <cuda_runtime.h>
#include <vector_functions.h>
#include "../cuda_math.h"

namespace MyEngine {

// Use CUDA built-in types
typedef float3 Float3;
typedef float2 Float2;
typedef float4 Float4;

// Ray representation
struct Ray {
    float3 origin;
    float3 direction;

    __host__ __device__ Ray() : origin(make_float3(0, 0, 0)), direction(make_float3(0, 0, 0)) {}
    __host__ __device__ Ray(float3 o, float3 d) : origin(o), direction(d) {}

    __host__ __device__ float3 at(float t) const {
        return make_float3(
            origin.x + direction.x * t,
            origin.y + direction.y * t,
            origin.z + direction.z * t
        );
    }
};

// Hit record for ray intersection
struct HitRecord {
    float3 point;
    float3 normal;
    float t;
    float u, v;
    int face_id;
    int material_id;

    __host__ __device__ HitRecord()
        : point(make_float3(0, 0, 0)), normal(make_float3(0, 0, 0)),
          t(0), u(0), v(0), face_id(-1), material_id(-1) {}

    __host__ __device__ void setFaceNormal(const Ray& r, const float3& outward_normal) {
        face_id = (dot(r.direction, outward_normal) < 0) ? 1 : 0;
        if (face_id) {
            normal = outward_normal;
        } else {
            normal = make_float3(-outward_normal.x, -outward_normal.y, -outward_normal.z);
        }
    }
};

// Bounding box
struct AABB {
    float3 min;
    float3 max;

    __host__ __device__ AABB() : min(make_float3(0, 0, 0)), max(make_float3(0, 0, 0)) {}
    __host__ __device__ AABB(float3 min_val, float3 max_val) : min(min_val), max(max_val) {}

    __host__ __device__ bool hit(const Ray& r, float t_min, float t_max) const {
        float3 invD = make_float3(
            r.direction.x != 0 ? 1.0f / r.direction.x : 0,
            r.direction.y != 0 ? 1.0f / r.direction.y : 0,
            r.direction.z != 0 ? 1.0f / r.direction.z : 0
        );

        float3 t0 = make_float3(
            (min.x - r.origin.x) * invD.x,
            (min.y - r.origin.y) * invD.y,
            (min.z - r.origin.z) * invD.z
        );
        float3 t1 = make_float3(
            (max.x - r.origin.x) * invD.x,
            (max.y - r.origin.y) * invD.y,
            (max.z - r.origin.z) * invD.z
        );

        float3 t_near = make_float3(fminf(t0.x, t1.x), fminf(t0.y, t1.y), fminf(t0.z, t1.z));
        float3 t_far = make_float3(fmaxf(t0.x, t1.x), fmaxf(t0.y, t1.y), fmaxf(t0.z, t1.z));

        t_min = fmaxf(t_near.x, fmaxf(t_near.y, fmaxf(t_near.z, t_min)));
        t_max = fminf(t_far.x, fminf(t_far.y, fminf(t_far.z, t_max)));

        return t_max > t_min && t_max > 0;
    }

    __host__ __device__ float3 centroid() const {
        return make_float3(
            (min.x + max.x) * 0.5f,
            (min.y + max.y) * 0.5f,
            (min.z + max.z) * 0.5f
        );
    }
};

// BVH node
struct BVHNode {
    AABB bounding_box;
    int left;
    int right;
    int start;
    int count;
    bool is_leaf;

    __host__ __device__ BVHNode() : left(-1), right(-1), start(0), count(0), is_leaf(true) {}
};

// Triangle vertex data
struct Triangle {
    float3 v0, v1, v2;
    int material_id;

    __host__ __device__ Triangle() : material_id(-1) {}
    __host__ __device__ Triangle(float3 a, float3 b, float3 c, int mat = -1)
        : v0(a), v1(b), v2(c), material_id(mat) {}

    __host__ __device__ AABB getBoundingBox() const {
        float3 min_val = make_float3(
            fminf(v0.x, fminf(v1.x, v2.x)),
            fminf(v0.y, fminf(v1.y, v2.y)),
            fminf(v0.z, fminf(v1.z, v2.z))
        );
        float3 max_val = make_float3(
            fmaxf(v0.x, fmaxf(v1.x, v2.x)),
            fmaxf(v0.y, fmaxf(v1.y, v2.y)),
            fmaxf(v0.z, fmaxf(v1.z, v2.z))
        );
        return AABB(min_val, max_val);
    }

    __host__ __device__ float3 centroid() const {
        return make_float3(
            (v0.x + v1.x + v2.x) * 0.333333f,
            (v0.y + v1.y + v2.y) * 0.333333f,
            (v0.z + v1.z + v2.z) * 0.333333f
        );
    }

    __host__ __device__ float3 normal() const {
        float3 e1 = make_float3(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z);
        float3 e2 = make_float3(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z);
        float3 n = cross(e1, e2);
        float len = sqrtf(dot(n, n));
        if (len > 0) {
            n = make_float3(n.x / len, n.y / len, n.z / len);
        }
        return n;
    }
};

// Material types
enum MaterialType {
    MATERIAL_LAMBERTIAN = 0,
    MATERIAL_METAL = 1,
    MATERIAL_DIELECTRIC = 2,
    MATERIAL_EMISSIVE = 3,
    MATERIAL_NONE = 255
};

// Material data
struct Material {
    MaterialType type;
    float3 albedo;
    float roughness;
    float metallic;
    float3 emissive;

    __host__ __device__ Material()
        : type(MATERIAL_NONE), albedo(make_float3(0, 0, 0)),
          roughness(1.0f), metallic(0), emissive(make_float3(0, 0, 0)) {}
};

// Sampler state for path tracing
struct SamplerState {
    unsigned int seed;
    unsigned int sample_index;

    __host__ __device__ SamplerState() : seed(0), sample_index(0) {}

    __host__ __device__ void init(unsigned int frame) {
        seed = frame * 0x9e3779b9u + 0x85ebca6bu;
        sample_index = frame;
    }

    __host__ __device__ float nextFloat() {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        return (seed & 0xFFFFFFFu) / 16777216.0f;
    }

    __host__ __device__ float3 nextFloat3() {
        return make_float3(nextFloat(), nextFloat(), nextFloat());
    }

    __host__ __device__ float3 randomInUnitSphere() {
        float3 p;
        do {
            p = make_float3(
                2.0f * nextFloat() - 1.0f,
                2.0f * nextFloat() - 1.0f,
                2.0f * nextFloat() - 1.0f
            );
        } while (dot(p, p) >= 1.0f);
        return p;
    }

    __host__ __device__ float3 randomUnitVector() {
        float a = nextFloat() * 2.0f * 3.14159265f;
        float z = nextFloat() * 2.0f - 1.0f;
        float r = sqrtf(1.0f - z * z);
        return make_float3(r * cosf(a), r * sinf(a), z);
    }

    __host__ __device__ float3 randomOnHemisphere(const float3& normal) {
        float3 on_unit_sphere = randomUnitVector();
        float d = dot(on_unit_sphere, normal);
        if (d > 0) return on_unit_sphere;
        return make_float3(-on_unit_sphere.x, -on_unit_sphere.y, -on_unit_sphere.z);
    }
};

// Camera for ray generation
struct Camera {
    float3 position;
    float3 lower_left;
    float3 horizontal;
    float3 vertical;
    float3 u, v, w;
    float lens_radius;
    float aspect_ratio;
    int sample_count;

    __host__ __device__ Camera() : position(make_float3(0, 0, 0)), sample_count(1) {}

    __host__ __device__ void initialize(float3 cam_pos, float3 look_at, float3 up,
                                float vfov, float aspect, float aperture = 0.0f) {
        position = cam_pos;
        aspect_ratio = aspect;
        lens_radius = aperture * 0.5f;

        float theta = vfov * 3.14159265f / 180.0f;
        float h = tanf(theta * 0.5f);

        float viewport_height = 2.0f * h;
        float viewport_width = aspect_ratio * viewport_height;

        float3 cam_w = make_float3(
            cam_pos.x - look_at.x,
            cam_pos.y - look_at.y,
            cam_pos.z - look_at.z
        );
        float len_w = sqrtf(dot(cam_w, cam_w));
        cam_w = make_float3(cam_w.x / len_w, cam_w.y / len_w, cam_w.z / len_w);

        float3 cam_u = cross(up, cam_w);
        float len_u = sqrtf(dot(cam_u, cam_u));
        if (len_u > 0) {
            cam_u = make_float3(cam_u.x / len_u, cam_u.y / len_u, cam_u.z / len_u);
        } else {
            cam_u = make_float3(1, 0, 0);
        }

        float3 cam_v = cross(cam_w, cam_u);

        horizontal = make_float3(
            cam_u.x * viewport_width,
            cam_u.y * viewport_width,
            cam_u.z * viewport_width
        );
        vertical = make_float3(
            cam_v.x * viewport_height,
            cam_v.y * viewport_height,
            cam_v.z * viewport_height
        );

        lower_left = make_float3(
            position.x - horizontal.x * 0.5f - vertical.x * 0.5f - cam_w.x,
            position.y - horizontal.y * 0.5f - vertical.y * 0.5f - cam_w.y,
            position.z - horizontal.z * 0.5f - vertical.z * 0.5f - cam_w.z
        );

        u = cam_u;
        v = cam_v;
        w = cam_w;
    }

    __host__ __device__ Ray getRay(float s, float t, SamplerState& sampler) const {
        float3 rd = sampler.randomInUnitSphere();
        rd = make_float3(
            rd.x * lens_radius,
            rd.y * lens_radius,
            rd.z * lens_radius
        );

        float3 offset = make_float3(
            u.x * rd.x + v.x * rd.y,
            u.y * rd.x + v.y * rd.y,
            u.z * rd.x + v.z * rd.y
        );

        float3 ray_origin = make_float3(
            position.x + offset.x,
            position.y + offset.y,
            position.z + offset.z
        );

        float3 ray_dir = make_float3(
            lower_left.x + horizontal.x * s + vertical.x * t - ray_origin.x,
            lower_left.y + horizontal.y * s + vertical.y * t - ray_origin.y,
            lower_left.z + horizontal.z * s + vertical.z * t - ray_origin.z
        );

        return Ray(ray_origin, ray_dir);
    }

    __host__ __device__ Ray getRaySimple(float s, float t) const {
        return Ray(position, make_float3(
            lower_left.x + horizontal.x * s + vertical.x * t - position.x,
            lower_left.y + horizontal.y * s + vertical.y * t - position.y,
            lower_left.z + horizontal.z * s + vertical.z * t - position.z
        ));
    }
};

} // namespace MyEngine
