#include "romanorender/app.h"
#include <imgui.h>
#include <imnodes.h>

ROMANORENDER_NAMESPACE_BEGIN

ROMANORENDER_API void draw_scenegraph(SceneGraph& graph) noexcept
{
    ImGui::Begin("Scenegraph");

    ImNodes::BeginNodeEditor();

    if(ImGui::IsWindowHovered() && ImNodes::IsEditorHovered() && ImGui::IsMouseClicked(1))
    {
        ImGui::OpenPopup("Create Node");
    }

    if(ImGui::BeginPopup("Create Node"))
    {
        const ImVec2 pos = ImGui::GetMousePosOnOpeningCurrentPopup();

        if(ImGui::BeginMenu("Add Node"))
        {
            for(const stdromano::String<>& node_type : SceneGraphNodesManager::get_instance().get_types())
            {
                if(node_type.startswith("__"))
                {
                    continue;
                }

                if(ImGui::MenuItem(node_type.c_str()))
                {
                    SceneGraphNode* node = SceneGraphNodesManager::get_instance().create_node(node_type);
                    graph.add_node(node);
                }
            }

            ImGui::EndMenu();
        }

        ImGui::EndPopup();
    }

    for(SceneGraphNode* node : graph.get_nodes())
    {
        ImNodes::BeginNode(node->get_id());

        ImNodes::BeginNodeTitleBar();
        ImGui::TextUnformatted(node->get_type_name());
        ImNodes::EndNodeTitleBar();

        for(uint32_t i = 0; i < node->get_num_inputs(); ++i)
        {
            const int pin_id = node->get_id() * 1000 + i;
            ImNodes::BeginInputAttribute(pin_id);
            ImGui::Text("%s", node->get_input_name(i));
            ImNodes::EndInputAttribute();
        }

        const int out_pin = node->get_id() * 1000 + 999;
        ImNodes::BeginOutputAttribute(out_pin);
        ImGui::Text("Output");
        ImNodes::EndOutputAttribute();

        ImNodes::EndNode();
    }

    for(SceneGraphNode* node : graph.get_nodes())
    {
        auto& inputs = node->get_inputs();
        for(uint32_t i = 0; i < inputs.size(); ++i)
        {
            if(inputs[i] != nullptr)
            {
                const int src_pin = inputs[i]->get_id() * 1000 + 999;
                const int dst_pin = node->get_id() * 1000 + i;
                ImNodes::Link(i, src_pin, dst_pin);
            }
        }
    }

    ImNodes::EndNodeEditor();

    int start, end;
    if(ImNodes::IsLinkCreated(&start, &end))
    {
        int output_pin, input_pin;
        if(start % 1000 == 999)
        {
            output_pin = start;
            input_pin = end;
        }
        else
        {
            output_pin = end;
            input_pin = start;
        }

        const int src_node = output_pin / 1000;
        const int dst_node = input_pin / 1000;
        const uint32_t input_idx = input_pin % 1000;

        SceneGraphNode* source = nullptr;
        SceneGraphNode* dest = nullptr;
        for(auto* node : graph.get_nodes())
        {
            if(node->get_id() == src_node)
                source = node;
            if(node->get_id() == dst_node)
                dest = node;
        }

        if(source && dest)
        {
            if(input_idx < dest->get_num_inputs() && !dest->get_inputs()[input_idx])
            {
                graph.connect_nodes(source, dest, input_idx);
            }
        }
    }

    ImGui::End();
}

ROMANORENDER_NAMESPACE_END