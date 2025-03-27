#include "romanorender/object_algos.h"
#include "romanorender/scenegraph.h"


#include "stdromano/logger.h"

#include <regex>

ROMANORENDER_NAMESPACE_BEGIN

class ROMANORENDER_API SceneGraphNode_Output : public SceneGraphNode
{
public:
    SceneGraphNode_Output() : SceneGraphNode(1, 0) {}

    virtual const char* get_input_name(const uint32_t input) const noexcept override
    {
        return "objects";
    }

    virtual const char* get_type_name() const noexcept override { return "__output"; }

    virtual bool execute() override
    {
        for(const SceneGraphNode* input : this->get_inputs())
        {
            for(const Object* object : input->get_objects())
            {
                this->get_objects().emplace_back(object->reference());
            }
        }

        return true;
    }
};

class ROMANORENDER_API SceneGraphNode_Mesh : public SceneGraphNode
{
public:
    SceneGraphNode_Mesh() : SceneGraphNode(0)
    {
        this->add_parameter("path_pattern", ParameterType_String, ".*");
    }

    virtual const char* get_input_name(const uint32_t input) const noexcept override { return ""; }

    virtual const char* get_type_name() const noexcept override { return "mesh"; }

    virtual bool execute() override
    {
        std::regex path_regex(this->get_parameter("path_pattern")->get_string().c_str());

        for(const Object* object : ObjectsManager::get_instance().get_objects())
        {
            if(const ObjectMesh* mesh = dynamic_cast<const ObjectMesh*>(object))
            {
                std::cmatch cm;

                if(std::regex_search(object->get_path().c_str(), cm, path_regex))
                {
                    this->get_objects().emplace_back(object->reference());
                }
            }
        }

        return true;
    }
};

class ROMANORENDER_API SceneGraphNode_Camera : public SceneGraphNode
{
public:
    SceneGraphNode_Camera() : SceneGraphNode(0)
    {
        this->add_parameter("path_pattern", ParameterType_String, ".*");

        this->add_parameter("focal", ParameterType_Float, 50.0f);

        this->add_parameter("posx", ParameterType_Float, 0.0f);
        this->add_parameter("posy", ParameterType_Float, 0.0f);
        this->add_parameter("posz", ParameterType_Float, 0.0f);

        this->add_parameter("rotx", ParameterType_Float, 0.0f);
        this->add_parameter("roty", ParameterType_Float, 0.0f);
        this->add_parameter("rotz", ParameterType_Float, 0.0f);
    }

    virtual const char* get_input_name(const uint32_t input) const noexcept override { return ""; }

    virtual const char* get_type_name() const noexcept override { return "camera"; }

    virtual bool execute() override
    {
        std::regex path_regex(this->get_parameter("path_pattern")->get_string().c_str());

        Object* object = nullptr;
        ObjectsMatchingPatternIterator it = 0;
        ObjectCamera* camera = nullptr;

        while(objects_manager().get_objects_matching_pattern(it, path_regex, &object))
        {
            if(const ObjectCamera* cam = dynamic_cast<const ObjectCamera*>(object))
            {
                camera = const_cast<ObjectCamera*>(cam);
                break;
            }
        }

        if(camera == nullptr)
        {
            this->set_error(stdromano::String<>("Cannot find any camera matching pattern: {}",
                                                this->get_parameter("path_pattern")->get_string()));
            return false;
        }

        ObjectCamera* node_camera = camera->reference();

        node_camera->set_focal(this->get_parameter("focal")->get_float());

        const Vec3F t(this->get_parameter("posx")->get_float(),
                      this->get_parameter("posy")->get_float(),
                      this->get_parameter("posz")->get_float());

        const Vec3F r(this->get_parameter("rotx")->get_float(),
                      this->get_parameter("roty")->get_float(),
                      this->get_parameter("rotz")->get_float());

        const Vec3F s(1.0f);

        const Mat44F transform = Mat44F::from_trs(t, r, s);

        node_camera->set_transform(transform);

        this->get_objects().emplace_back(node_camera);

        return true;
    }
};

class ROMANORENDER_API SceneGraphNode_Merge : public SceneGraphNode
{
public:
    SceneGraphNode_Merge() : SceneGraphNode(2) {}

    virtual const char* get_input_name(const uint32_t input) const noexcept override
    {
        switch(input)
        {
        case 0:
            return "objects to merge";
        case 1:
            return "objects to merge";
        default:
            return "";
        }

        return "";
    }

    virtual const char* get_type_name() const noexcept override { return "merge"; }

    virtual bool execute() override
    {
        for(const SceneGraphNode* input : this->get_inputs())
        {
            for(const Object* object : input->get_objects())
            {
                this->get_objects().emplace_back(object->reference());
            }
        }

        return true;
    }
};

class ROMANORENDER_API SceneGraphNode_SetTransform : public SceneGraphNode
{
public:
    SceneGraphNode_SetTransform() : SceneGraphNode(1)
    {
        this->add_parameter("posx", ParameterType_Float, 0.0f);
        this->add_parameter("posy", ParameterType_Float, 0.0f);
        this->add_parameter("posz", ParameterType_Float, 0.0f);

        this->add_parameter("rotx", ParameterType_Float, 0.0f);
        this->add_parameter("roty", ParameterType_Float, 0.0f);
        this->add_parameter("rotz", ParameterType_Float, 0.0f);

        this->add_parameter("scalex", ParameterType_Float, 1.0f);
        this->add_parameter("scaley", ParameterType_Float, 1.0f);
        this->add_parameter("scalez", ParameterType_Float, 1.0f);
    }

    virtual const char* get_input_name(const uint32_t input) const noexcept override
    {
        return "objects";
    }

    virtual const char* get_type_name() const noexcept override { return "set_transform"; }

    virtual bool execute() override
    {
        const Vec3F t(this->get_parameter("posx")->get_float(),
                      this->get_parameter("posy")->get_float(),
                      this->get_parameter("posz")->get_float());

        const Vec3F r(this->get_parameter("rotx")->get_float(),
                      this->get_parameter("roty")->get_float(),
                      this->get_parameter("rotz")->get_float());

        const Vec3F s(this->get_parameter("scalex")->get_float(),
                      this->get_parameter("scaley")->get_float(),
                      this->get_parameter("scalez")->get_float());

        const Mat44F transform = Mat44F::from_trs(t, r, s);

        for(const Object* object : this->get_inputs()[0]->get_objects())
        {
            Object* obj = object->reference();
            obj->set_transform(transform);
            this->get_objects().emplace_back(obj);
        }

        return true;
    }
};

class ROMANORENDER_API SceneGraphNode_Attributes : public SceneGraphNode
{
public:
    SceneGraphNode_Attributes() : SceneGraphNode(1)
    {
        this->add_parameter("visible", ParameterType_Bool, true);
        this->add_parameter("smooth_normals", ParameterType_Bool, false);
        this->add_parameter("subdivision_level", ParameterType_Int, 0);
    }

    virtual const char* get_input_name(const uint32_t input) const noexcept override
    {
        return "objects";
    }

    virtual const char* get_type_name() const noexcept override { return "attributes"; }

    virtual bool execute() override
    {
        const bool visible = this->get_parameter("visible")->get_bool();
        const bool smooth_normals = this->get_parameter("smooth_normals")->get_bool();
        const uint32_t subdivision_level = this->get_parameter("subdivision_level")->get_int();

        for(const Object* object : this->get_inputs()[0]->get_objects())
        {
            if(const ObjectMesh* mesh = dynamic_cast<const ObjectMesh*>(object))
            {
                ObjectMesh* mod_mesh = mesh->reference();

                mod_mesh->set_is_visible(visible);

                if(subdivision_level > 0)
                {
                    object_algos::subdivide(mod_mesh, subdivision_level);
                }

                if(smooth_normals)
                {
                    object_algos::smooth_normals(mod_mesh);
                }

                this->get_objects().emplace_back(mod_mesh);
            }
            else
            {
                this->set_error("Cannot set attributes on objects different than meshes");
                return false;
            }
        }

        return true;
    }
};

class ROMANORENDER_API SceneGraphNode_Instancer : public SceneGraphNode
{
public:
    SceneGraphNode_Instancer() : SceneGraphNode(1)
    {
        this->add_parameter("use_attribute_if_found", ParameterType_Bool, true);
        this->add_parameter("orient_name_attribute", ParameterType_String, "orient");
    }

    virtual const char* get_input_name(const uint32_t input) const noexcept override
    {
        switch(input)
        {
        case 0:
            return "Object";
        case 1:
            return "Point Cloud";
        }

        return "";
    }

    virtual const char* get_type_name() const noexcept override { return "instancer"; }

    virtual bool execute() override
    {
        if(this->get_inputs().empty() || this->get_inputs()[0] == nullptr || this->get_inputs()[1] == nullptr)
        {
            this->set_error("Instancer node requires an input mesh and pointcloud");
            return false;
        }

        const stdromano::Vector<Object*>& input_objects = this->get_inputs()[0]->get_objects();

        if(input_objects.empty())
        {
            this->set_error("Input pointcloud node has no objects");
            return false;
        }

        ObjectMesh* object_to_instance = nullptr;

        for(Object* obj : input_objects)
        {
            if(ObjectMesh* mesh = dynamic_cast<ObjectMesh*>(obj))
            {
                object_to_instance = mesh;
                break;
            }
        }

        if(!object_to_instance)
        {
            this->set_error("No ObjectMesh found in input Object");
            return false;
        }

        const stdromano::Vector<Object*>& input_point_clouds = this->get_inputs()[1]->get_objects();

        ObjectMesh* point_cloud = nullptr;

        for(Object* obj : input_objects)
        {
            if(ObjectMesh* mesh = dynamic_cast<ObjectMesh*>(obj))
            {
                point_cloud = mesh;
                break;
            }
        }

        if(point_cloud == nullptr)
        {
            this->set_error("No ObjectMesh found in input Point Cloud");
            return false;
        }

        Vertices& positions = point_cloud->get_vertices();

        if(positions.empty())
        {
            this->set_error("Input pointcloud has no vertices.");
            return false;
        }

        const AttributeBuffer* orient_buffer = point_cloud->get_vertex_attribute_buffer("orient");

        for(size_t i = 0; i < positions.size(); ++i)
        {
            ObjectInstance* instance = new ObjectInstance();

            const Vec3F position(positions[i].x, positions[i].y, positions[i].z);

            Mat44F transform = Mat44F::from_translation(position);

            instance->set_instanced(object_to_instance);
            instance->set_transform(transform);

            this->get_objects().push_back(instance);
        }

        return true;
    }
};

void register_builtin_nodes(SceneGraphNodesManager& manager) noexcept
{
    manager.register_node_type(stdromano::String<>::make_ref("__output", 8),
                               []() -> SceneGraphNode* { return new SceneGraphNode_Output; });

    manager.register_node_type(stdromano::String<>::make_ref("mesh", 4),
                               []() -> SceneGraphNode* { return new SceneGraphNode_Mesh; });

    manager.register_node_type(stdromano::String<>::make_ref("camera", 6),
                               []() -> SceneGraphNode* { return new SceneGraphNode_Camera; });

    manager.register_node_type(stdromano::String<>::make_ref("merge", 5),
                               []() -> SceneGraphNode* { return new SceneGraphNode_Merge; });

    manager.register_node_type(stdromano::String<>::make_ref("set_transform", 13),
                               []() -> SceneGraphNode* { return new SceneGraphNode_SetTransform; });

    manager.register_node_type(stdromano::String<>::make_ref("attributes", 10),
                               []() -> SceneGraphNode* { return new SceneGraphNode_Attributes; });

    manager.register_node_type(stdromano::String<>::make_ref("instancer", 9),
                               []() -> SceneGraphNode* { return new SceneGraphNode_Instancer; });
}

ROMANORENDER_NAMESPACE_END