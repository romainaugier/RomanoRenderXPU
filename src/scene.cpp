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

    stdromano::log_debug("Added a new object to the scene: {} (id: {})", obj.get_name(), obj.get_id());

    this->_instances.emplace_back(this->_blasses.size());

    this->_blasses.push_back(&obj.get_blas());

    this->_objects.push_back(&obj);
}

const Object* Scene::get_object(const uint32_t id) noexcept
{
    return id >= this->_objects.size() ? nullptr : this->_objects[id];
}

void Scene::add_instance(const Object* obj, const Mat44F& transform) noexcept {}

void Scene::build_tlas() noexcept
{
    this->_tlas.Build(this->_instances.data(),
                      this->_instances.size(),
                      const_cast<tinybvh::BVHBase**>(this->_blasses.data()),
                      this->_blasses.size());

    stdromano::log_debug("Built scene TLAS. Bounds:\nmin({})\nmax({})", this->_tlas.aabbMin, this->_tlas.aabbMax);
}

ROMANORENDER_NAMESPACE_END