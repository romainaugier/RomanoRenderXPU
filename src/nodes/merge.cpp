#include "romanorender/rendergraph.h"

ROMANORENDER_NAMESPACE_BEGIN

class ROMANORENDER_API MergeNode : public Node
{
public:
    MergeNode()
    {
        this->set_num_inputs(2);
    }

    void compute() override
    {
        Node* input_1 = this->get_input(0);
        Node* input_2 = this->get_input(1);

        const uint32_t type_1 = input_1->get_data_block().get_type();
        const uint32_t type_2 = input_2->get_data_block().get_type();

        if(type_1 != type_2)
        {
            this->set_error("Cannot merge different data types");
            return;
        }

        switch(type_1)
        {
            case NodeDataBlockType_Geometry:
                /* code */
                break;
            
            default:
                break;
        }
    }
};

ROMANORENDER_NAMESPACE_END

ROMANORENDER_API void register_node(romanorender::NodeManager& manager)
{
    manager.register_node_type("Merge", [](){
        return new romanorender::MergeNode();
    });
}