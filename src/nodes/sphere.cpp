#include "romanorender/rendergraph.h"
#include "romanorender/geometry.h"

ROMANORENDER_NAMESPACE_BEGIN

class ROMANORENDER_API SphereNode : public Node
{
public:
    SphereNode()
    {
        this->set_num_inputs(0);

        this->set_parameter<Vec3F>("position", Vec3F(0.0f));
        this->set_parameter<float>("radius", 1.0f);
    }

    void compute() override
    {
        Geometry* single_sphere = new Geometry(GeometryType_Point);
        Vec3F* positions = (Vec3F*)single_sphere->add_geometry_buffer(GeometryBufferType_Vertex, 
                                                                      GeometryBufferFormat_Float3,
                                                                      sizeof(Vec3F),
                                                                      1);
        positions[0] = this->get_parameter<Vec3F>("position");

        float* radius = (float*)single_sphere->add_geometry_buffer(GeometryBufferType_VertexAttributeRadius,
                                                                   GeometryBufferFormat_Float1,
                                                                   sizeof(float),
                                                                   1);
        radius[0] = this->get_parameter<float>("radius");

        this->set_data_block((void*)single_sphere, sizeof(Geometry*), NodeDataBlockType_Geometry);
    }
};

ROMANORENDER_NAMESPACE_END

ROMANORENDER_API void register_node(romanorender::NodeManager& manager)
{
    manager.register_node_type("Sphere", [](){
        return new romanorender::SphereNode();
    });
}
