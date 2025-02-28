#pragma once

#if !defined(__ROMANORENDER_OBJECT)
#define __ROMANORENDER_OBJECT

#include "romanorender/vec4.h"
#include "romanorender/mat44.h"

#include "stdromano/vector.h"
#include "stdromano/string.h"

#include "tiny_bvh.h"

ROMANORENDER_NAMESPACE_BEGIN

class ROMANORENDER_API Object 
{
    stdromano::Vector<Vec4F> _vertices;
    stdromano::Vector<uint32_t> _indices;

    tinybvh::BVH8_CPU _blas;

    Mat44F _transform;

    uint32_t _id;
    stdromano::String<> _name;

public:
    Object() {}

    static Object cube(const Vec3F& center, const Vec3F& scale) noexcept;

    void build_blas() noexcept
    {
        this->_blas.Build((tinybvh::bvhvec4*)this->_vertices.data(), this->_indices.data(), this->_indices.size() / 3);
    }

    void set_transform(const Mat44F& transform) 
    {
        this->_transform = transform;
    }

    ROMANORENDER_FORCE_INLINE const stdromano::Vector<Vec4F>& get_vertices() const noexcept { return this->_vertices; };
    ROMANORENDER_FORCE_INLINE stdromano::Vector<Vec4F>& get_vertices() noexcept { return this->_vertices; };
    ROMANORENDER_FORCE_INLINE const stdromano::Vector<uint32_t>& get_indices() const noexcept { return this->_indices; };
    ROMANORENDER_FORCE_INLINE stdromano::Vector<uint32_t>& get_indices() noexcept { return this->_indices; };
    ROMANORENDER_FORCE_INLINE const tinybvh::BVH8_CPU& get_blas() const noexcept { return this->_blas; }
    ROMANORENDER_FORCE_INLINE const Mat44F& get_transform() const noexcept { return this->_transform; }
    ROMANORENDER_FORCE_INLINE uint32_t get_id() const noexcept { return this->_id; }
    ROMANORENDER_FORCE_INLINE const stdromano::String<>& get_name() const noexcept { return this->_name; }

    ROMANORENDER_FORCE_INLINE void set_id(uint32_t id) { this->_id = id; }
    ROMANORENDER_FORCE_INLINE void set_name(const stdromano::String<>& name) { this->_name = name; }
};

bool objects_from_obj_file(const char* file_path, stdromano::Vector<Object>& object) noexcept;

ROMANORENDER_NAMESPACE_END

#endif /* !defined(__ROMANORENDER_OBJECT) */