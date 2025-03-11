#include "romanorender/app_widgets.h"

#include "stdromano/logger.h"

#include <imgui.h>
#include <imnodes.h>

ROMANORENDER_NAMESPACE_BEGIN

IconsManager::IconsManager()
{
    ImGuiIO& io = ImGui::GetIO();
    io.Fonts->AddFontFromFileTTF("res/material_icons.ttf", 16.0f);

    std::FILE* file = std::fopen("res/material_icons.codepoints", "r");

    if(file == nullptr)
    {
        stdromano::log_error("Cannot load codepoints file");
        return;
    }

    std::fseek(file, 0, SEEK_END);
    const size_t file_size = std::ftell(file);
    std::rewind(file);
    stdromano::String<> content = stdromano::String<>::make_zeroed(file_size);
    std::fread(content.c_str(), sizeof(char), file_size, file);
    std::fclose(file);

    stdromano::String<> split;
    stdromano::String<>::split_iterator split_it = 0;

    while(content.split("\n", split_it, split))
    {
        const int32_t space_pos = split.find(" ");

        stdromano::String<> icon_name("{}", fmt::string_view(split.data(), space_pos));
        stdromano::String<> icon_cp("\\u{}",
                                    fmt::string_view(split.data() + space_pos + 1, split.size() - space_pos - 1));

        this->_icons_lookup.insert(std::make_pair(icon_name, icon_cp));
    }

    stdromano::log_debug("Loaded {} icons", this->_icons_lookup.size());
}

/* Objects */
void draw_objects() noexcept
{
    stdromano::String<> window_name = IconsManager::get_instance().get_code_point("deployed_code");
    window_name.appendc(" Objects");

    ImGui::Begin(window_name.c_str());

    // Object list
    const float icon_size = 24.0f;
    static Object* selected_object = nullptr;

    for(Object* obj : ObjectsManager::get_instance().get_objects())
    {
        ImGui::PushID(obj);

        // Icon and name
        // ImGui::Image(obj->get_icon(), ImVec2(icon_size, icon_size));
        // ImGui::SameLine();

        // Selectable text
        // if(ImGui::Selectable(obj->name.c_str(), selected_object == obj))
        // {
        //     selected_object = obj;
        // }

        // Context menu
        if(ImGui::BeginPopupContextItem())
        {
            if(ImGui::MenuItem("Delete"))
            {
                // ObjectsManager::get_instance().remove_object(obj);
            }
            ImGui::EndPopup();
        }

        ImGui::PopID();
    }

    // Create buttons
    ImGui::Separator();
    if(ImGui::Button("Create Mesh"))
    {
        ObjectsManager::get_instance().add_object(new ObjectMesh());
    }
    ImGui::SameLine();
    if(ImGui::Button("Create Camera"))
    {
        ObjectsManager::get_instance().add_object(new ObjectCamera());
    }

    ImGui::End();
}

namespace
{
struct NodeLink
{
    int id;
    int output_pin;
    int input_pin;
    int input_idx;
    SceneGraphNode* source;
    SceneGraphNode* dest;
};
}

int32_t get_input_hash(const SceneGraphNode* node, const uint32_t input_id) noexcept
{
    const stdromano::String<128> uuid("{}{}_in{}", node->get_type_name(), node->get_id(), input_id);
    return (int32_t)stdromano::hash_fnv1a(uuid.c_str());
}

int32_t get_output_hash(const SceneGraphNode* node, const uint32_t output_id) noexcept
{
    const stdromano::String<128> uuid("{}{}_out{}", node->get_type_name(), node->get_id(), output_id);
    return (int32_t)stdromano::hash_fnv1a(uuid.c_str());
}

ROMANORENDER_API void draw_scenegraph(SceneGraph& graph) noexcept
{
    ImGui::Begin("Scenegraph");
    ImNodes::BeginNodeEditor();

    const bool is_editor_hovered = ImNodes::IsEditorHovered();
    const bool right_click = ImGui::IsMouseClicked(ImGuiMouseButton_Right);

    if(is_editor_hovered && right_click)
    {
        ImGui::OpenPopup("NodeContextMenu");
    }

    if(ImGui::BeginPopup("NodeContextMenu"))
    {
        const ImVec2 click_pos = ImGui::GetMousePosOnOpeningCurrentPopup();

        if(ImGui::BeginMenu("Add Node"))
        {
            for(const auto& node_type : SceneGraphNodesManager::get_instance().get_types())
            {
                if(node_type.startswith("__"))
                    continue;

                if(ImGui::MenuItem(node_type.c_str()))
                {
                    SceneGraphNode* node = SceneGraphNodesManager::get_instance().create_node(node_type);
                    graph.add_node(node);
                    ImNodes::SetNodeScreenSpacePos(node->get_id(), click_pos);
                }
            }
            ImGui::EndMenu();
        }

        ImGui::EndPopup();
    }

    static stdromano::HashMap<int, SceneGraphNode*> pin_map;
    static stdromano::Vector<NodeLink> links;
    links.clear();
    int link_id = 0;

    for(SceneGraphNode* node : graph.get_nodes())
    {
        ImNodes::BeginNode(node->get_id());

        ImNodes::BeginNodeTitleBar();
        ImGui::TextUnformatted(node->get_type_name());
        ImNodes::EndNodeTitleBar();

        for(uint32_t i = 0; i < node->get_num_inputs(); i++)
        {
            const int pin_id = get_input_hash(node, i);
            ImNodes::BeginInputAttribute(pin_id);
            ImGui::Text("%s", node->get_input_name(i));
            ImNodes::EndInputAttribute();
            pin_map[pin_id] = node;
        }

        if(node->get_num_outputs() > 0)
        {
            const int pin_id = get_output_hash(node, 0);
            ImNodes::BeginOutputAttribute(pin_id);
            ImGui::Text("Output");
            ImNodes::EndOutputAttribute();
            pin_map[pin_id] = node;
        }

        ImNodes::EndNode();

        auto& inputs = node->get_inputs();

        for(uint32_t i = 0; i < inputs.size(); ++i)
        {
            if(inputs[i])
            {
                links.push_back({link_id++,
                                 get_output_hash(inputs[i], 0),
                                 get_input_hash(node, i),
                                 static_cast<int>(i),
                                 inputs[i],
                                 node});
            }
        }
    }

    for(const auto& link : links)
    {
        ImNodes::Link(link.id, link.output_pin, link.input_pin);
    }

    ImNodes::EndNodeEditor();

    int start_pin, end_pin;

    if(ImNodes::IsLinkCreated(&start_pin, &end_pin))
    {
        SceneGraphNode* start_node = pin_map[start_pin];
        SceneGraphNode* end_node = pin_map[end_pin];

        if(start_node && end_node)
        {
            bool valid = false;
            int input_idx = -1;

            if(start_pin == get_output_hash(start_node, 0))
            {
                valid = true;
                for(uint32_t i = 0; i < end_node->get_num_inputs(); ++i)
                {
                    if(end_pin == get_input_hash(end_node, i))
                    {
                        input_idx = i;
                        break;
                    }
                }
            }

            if(valid && input_idx >= 0)
            {
                if(std::strcmp(end_node->get_type_name(), "__output") == 0)
                {
                    graph.connect_nodes(start_node, end_node, input_idx);
                }
                else if(!end_node->get_inputs()[input_idx])
                {
                    graph.connect_nodes(start_node, end_node, input_idx);
                }
            }
        }
    }

    const int num_selected = ImNodes::NumSelectedNodes();

    if(num_selected > 0 && ImGui::IsKeyPressed(ImGuiKey_Delete))
    {
        static stdromano::Vector<int> selected_nodes;

        for(uint32_t i = 0; i < num_selected; i++)
        {
            selected_nodes.push_back(0);
        }

        ImNodes::GetSelectedNodes(selected_nodes.data());

        for(const int node_id : selected_nodes)
        {
            graph.remove_node(node_id);
        }
    }

    if(ImNodes::NumSelectedLinks() > 0 && ImGui::IsKeyPressed(ImGuiKey_Delete))
    {
        for(const auto& link : links)
        {
            if(ImNodes::IsLinkSelected(link.id))
            {
                if(link.dest && link.input_idx < link.dest->get_inputs().size())
                {
                    link.dest->get_inputs()[link.input_idx] = nullptr;
                    link.dest->set_dirty();
                    link.source->remove_output(link.dest);
                }
            }
        }
    }

    if(ImGui::Button("Execute"))
    {
        graph.execute();
    }

    ImGui::End();
}

ROMANORENDER_NAMESPACE_END