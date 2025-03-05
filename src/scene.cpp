#include "romanorender/scene.h"

#include "stdromano/logger.h"

ROMANORENDER_NAMESPACE_BEGIN

void Scene::add_object(Object& obj) noexcept
{
    const uint32_t id = this->_blasses.size();
    obj.set_id(id);

    if(obj.get_name().size() == 0)
    {
        obj.set_name(std::move(stdromano::String<>("object{}", id)));
    }

    this->_objects_lookup.emplace_back(id);

    this->_instances.emplace_back(id);
    std::memcpy(this->_instances.back().transform, obj.get_transform().data(), 16 * sizeof(float));

    this->_blasses.push_back(&obj.get_blas());

    this->_objects.push_back(&obj);

    stdromano::log_debug("Added a new object to the scene: {} (id: {})", obj.get_name(), obj.get_id());
}

const Object* Scene::get_object(const uint32_t instance_id) noexcept
{
    return instance_id >= this->_objects_lookup.size() ? nullptr : this->_objects[this->_objects_lookup[instance_id]];
}

void Scene::add_instance(const Object* obj, const Mat44F& transform) noexcept
{
    if(obj->get_id() == INVALID_OBJECT_ID)
    {
        return;
    }

    this->_objects_lookup.emplace_back(obj->get_id());

    this->_instances.emplace_back(obj->get_id());
    std::memcpy(this->_instances.back().transform, transform.data(), 16 * sizeof(float));

    stdromano::log_debug("Added a new instance to the scene: {} (id: {})", obj->get_name(), obj->get_id());
}

void Scene::build_tlas() noexcept
{
    this->_tlas.Build(this->_instances.data(),
                      this->_instances.size(),
                      const_cast<tinybvh::BVHBase**>(this->_blasses.data()),
                      this->_blasses.size());

    stdromano::log_debug("Built scene TLAS. Bounds:\nmin({})\nmax({})", this->_tlas.aabbMin, this->_tlas.aabbMax);
}

ROMANORENDER_NAMESPACE_END