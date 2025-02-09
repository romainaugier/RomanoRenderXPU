#pragma once

#if !defined(__ROMANORENDER_BVH)
#define __ROMANORENDER_BVH

#include "romanorender/geometry.h"
#include "romanorender/bbox.h"
#include "romanorender/ray.h"

#include "stdromano/logger.h"

#include <iterator>

ROMANORENDER_NAMESPACE_BEGIN

enum BVHBuildFlags_ : uint32_t
{
    BVHBuildFlags_BVHSimple = 0,
    BVHBuildFlags_BVH4 = 1,
    BVHBuildFlags_BVH8 = 2,
};

class ROMANORENDER_API Accelerator
{
    static constexpr uint32_t TRAVERSAL_STACK_SIZE = 64;
    struct alignas(32) BVHLinearNode
    {
        BBox bounds;

        union 
        {
            uint32_t second_child_offset = 0;
            uint32_t primitives_offset;
        };

        uint16_t level = 0;
        uint8_t axis = 0;
        uint8_t num_primitives = 0;
    };

    using BVHLinearNodes = stdromano::Vector<BVHLinearNode>;

    struct PrimitivePoint
    {
        Vec3F center;
        float radius;

        PrimitivePoint(const Vec3F& center, const float radius) : center(center), radius(radius) {}
        PrimitivePoint(const Vec3F* center, const float* radius) : center(*center), radius(*radius) {}
    };

    struct PrimitiveTriangle
    {
        Vec3F v0, v1, v2;

        PrimitiveTriangle(const Vec3F& v0, const Vec3F& v1, const Vec3F& v2) : v0(v0), v1(v1), v2(v2) {}
        PrimitiveTriangle(const Vec3F* v0, const Vec3F* v1, const Vec3F* v2) : v0(*v0), v1(*v1), v2(*v2) {}
    };

    class PrimitiveBuffer 
    {
    public:
        ROMANORENDER_PACKED_STRUCT(
        struct PrimitiveHeader 
        {
            uint16_t stride;
            uint8_t  type;
            uint32_t geom_id;
            uint32_t prim_id;
        });

    private:
        static constexpr size_t INITIAL_SIZE = 4096;
        static constexpr size_t ALIGNMENT = 64;
        
        uint8_t* buffer = nullptr;
        size_t capacity = 0;
        size_t size = 0;

    public:
        PrimitiveBuffer() { this->resize(INITIAL_SIZE); }

        ~PrimitiveBuffer() 
        {
            if(this->buffer != nullptr) 
            {
                stdromano::mem_aligned_free(buffer);
            }
        }

        ROMANORENDER_FORCE_INLINE size_t get_size() const noexcept { return this->size; }

        void clear() noexcept;

        uint32_t add_triangle(uint32_t geom_id, 
                              uint32_t prim_id,
                              const Vec3F* v0, 
                              const Vec3F* v1, 
                              const Vec3F* v2) noexcept;

        uint32_t add_point(uint32_t geom_id, 
                           uint32_t prim_id,
                           const Vec3F* center,
                           float radius) noexcept; 

        /* TODO: add other geom types */

        class Iterator 
        {
        public:
            using value_type = PrimitiveHeader;
            using difference_type = std::ptrdiff_t;
            using iterator_category = std::forward_iterator_tag;
            using pointer = value_type*;
            using reference = value_type&;

        private:
            const uint8_t* ptr;

        public:
            explicit Iterator(const uint8_t* p) : ptr(p) {}

            Iterator& operator++() 
            {
                auto header = reinterpret_cast<const PrimitiveHeader*>(ptr);
                ptr += header->stride;
                return *this;
            }

            const value_type operator*() const 
            {
                return *reinterpret_cast<const PrimitiveHeader*>(ptr);
            }

            const PrimitiveHeader* operator->() const 
            {
                return reinterpret_cast<const PrimitiveHeader*>(ptr);
            }

            bool operator!=(const Iterator& other) const 
            {
                return ptr != other.ptr;
            }
        };

        Iterator begin() const { return Iterator(buffer); }
        Iterator end() const { return Iterator(buffer + size); }

        const uint8_t* get_ptr_at(const size_t at) const noexcept
        {
            return this->buffer + at;
        }

    private:
        ROMANORENDER_FORCE_INLINE void check_capacity(const size_t required) noexcept
        {
            if(this->size + required > this->capacity) 
            {
                this->resize(this->capacity * 2 + required);
            }
        }

        void resize(size_t new_capacity) noexcept;
    };

    BVHLinearNodes lnodes;
    PrimitiveBuffer primitives;

    static bool intersect_triangle(const PrimitiveTriangle* triangle, 
                                   RayHit& rayhit) noexcept;

    static bool intersect_point(const PrimitivePoint* point,
                                RayHit& rayhit) noexcept;

public:
    bool build(const Geometries& geometries, const uint32_t flags) noexcept;

    bool intersect(RayHit& rayhit) const noexcept;

    bool occlude(RayHit& rayhit) const noexcept;
};

ROMANORENDER_NAMESPACE_END

#endif /* !defined(__ROMANORENDER_BVH) */