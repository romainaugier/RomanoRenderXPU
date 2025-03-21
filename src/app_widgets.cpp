#include "romanorender/app_widgets.h"
#include "romanorender/app_icons.h"

#include "stdromano/filesystem.h"
#include "stdromano/logger.h"

#define STDROMANO_ENABLE_PROFILING
#include "stdromano/profiling.h"

#include <imgui.h>
#include <imnodes.h>

#include <regex>
#include <unordered_map>
#include <unordered_set>

ROMANORENDER_NAMESPACE_BEGIN

/* Resources Manager */

void UIResourcesManager::load_fonts() noexcept
{
    ImGuiIO& io = ImGui::GetIO();

    static const ImWchar icons_ranges[] = {ICON_MIN_FK, ICON_MAX_FK, 0};

    const stdromano::String<> icons_path = stdromano::expand_from_executable_dir("res/forkawesome-webfont.ttf");
    const stdromano::String<> fonts_dir = stdromano::expand_from_executable_dir("res");

    stdromano::ListDirIterator it;

    while(stdromano::fs_list_dir(it, fonts_dir, stdromano::ListDirFlags_ListFiles))
    {
        const stdromano::String<> current_file_path = std::move(it.get_current_path());

        if(current_file_path.endswith(".ttf"))
        {
            const stdromano::String<> current_file_name = stdromano::fs_filename(current_file_path);

            if(current_file_name.startswith("Roboto"))
            {
                const stdromano::String<> font_name("{}",
                                                    fmt::string_view(current_file_name.data() + 7,
                                                                     current_file_name.size() - 11));

                ImFont* main_font = io.Fonts->AddFontFromFileTTF(current_file_path.c_str(), 16.0f);

                ImFontConfig icons_config;
                icons_config.MergeMode = true;
                icons_config.PixelSnapH = true;
                icons_config.GlyphOffset.y = 1.0f;

                io.Fonts->AddFontFromFileTTF(icons_path.c_str(), 16.0f, &icons_config, icons_ranges);

                this->_fonts.insert(std::make_pair(font_name.lower(), main_font));
            }
        }
    }

    io.Fonts->Build();

    stdromano::log_debug("Loaded {} fonts", this->_fonts.size() + 1);
}

void UIResourcesManager::load_imgui() const noexcept {
    const stdromano::String<> ini_path = stdromano::expand_from_executable_dir("res/imgui.ini");

    ImGui::LoadIniSettingsFromDisk(ini_path.c_str());
}

void UIResourcesManager::save_imgui() const noexcept
{
    const stdromano::String<> ini_path = stdromano::expand_from_executable_dir("res/imgui.ini");

    ImGui::SaveIniSettingsToDisk(ini_path.c_str());
}

UIResourcesManager::UIResourcesManager()
{
    SCOPED_PROFILE_START(stdromano::ProfileUnit::Seconds, ui_res_manager_startup);

    this->load_imgui();

    ImGuiIO& io = ImGui::GetIO();

    this->load_fonts();
}

UIResourcesManager::~UIResourcesManager()
{
}

/* State */

UIState::UIState() { this->_states[UIStateFlag_Show] = 1; }

/* Helpers */

#define FONT(call, font_name)                                                                      \
    ImGui::PushFont(ui_res_manager().get_font(font_name));                                         \
    call;                                                                                          \
    ImGui::PopFont()

#define BOLD(call) FONT(call, "bold")
#define REGULAR(call) FONT(call, "regular")
#define ITALIC(call) FONT(call, "italic")
#define LIGHT(call) FONT(call, "light")
#define THIN(call) FONT(call, "thin")

#define PUSH_FONT(font_name) ImGui::PushFont(ui_res_manager().get_font(font_name))
#define PUSH_BOLD() PUSH_FONT("bold")
#define PUSH_REGULAR() PUSH_FONT("regular")
#define PUSH_ITALIC() PUSH_FONT("italic")
#define PUSH_LIGHT() PUSH_FONT("light")
#define PUSH_THIN() PUSH_FONT("thin")
#define POP_FONT() ImGui::PopFont()

/* Objects */

void draw_objects() noexcept
{
    bool show = (bool)ui_state().get(UIStateFlag_Show);

    if(!show)
    {
        return;
    }

    BOLD(ImGui::Begin(ICON_FK_CUBES " Objects", &show));

    // Object list
    static Object* selected_object = nullptr;

    for(Object* obj : ObjectsManager::get_instance().get_objects())
    {
        ImGui::PushID(obj);

        PUSH_REGULAR();

        if(ObjectMesh* mesh = dynamic_cast<ObjectMesh*>(obj))
        {
            ImGui::Text(ICON_FK_CUBE " ");
            ImGui::SameLine();
        }
        else if(ObjectCamera* cam = dynamic_cast<ObjectCamera*>(obj))
        {
            ImGui::Text(ICON_FK_VIDEO_CAMERA " ");
            ImGui::SameLine();
        }

        if(ImGui::Selectable(obj->get_name().c_str(), selected_object == obj))
        {
            selected_object = obj;
        }

        if(ImGui::BeginDragDropSource(ImGuiDragDropFlags_SourceAllowNullID))
        {
            ImGui::SetDragDropPayload("OBJECT_PAYLOAD", &obj, sizeof(Object*));

            ImGui::Text("Dragging %s", obj->get_name().c_str());
            ImGui::EndDragDropSource();
        }

        // Context menu
        if(ImGui::BeginPopupContextItem())
        {
            if(ImGui::MenuItem("Delete"))
            {
                // ObjectsManager::get_instance().remove_object(obj);
            }

            ImGui::EndPopup();
        }

        POP_FONT();

        ImGui::PopID();
    }

    ImGui::End();
}

/* Nodegraph */

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

void draw_scenegraph(SceneGraph& graph, SceneGraphNode** current_node) noexcept
{
    bool show = (bool)ui_state().get(UIStateFlag_Show);

    if(!show)
    {
        return;
    }

    BOLD(ImGui::Begin(ICON_FK_HASHNODE " Scenegraph", &show));

    ImNodes::BeginNodeEditor();

    const bool is_editor_hovered = ImNodes::IsEditorHovered();
    const bool right_click = ImGui::IsMouseClicked(ImGuiMouseButton_Right);

    PUSH_REGULAR();

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

    POP_FONT();

    static stdromano::HashMap<int, SceneGraphNode*> pin_map;
    static stdromano::Vector<NodeLink> links;
    links.clear();
    int link_id = 0;

    for(SceneGraphNode* node : graph.get_nodes())
    {
        ImNodes::BeginNode(node->get_id());

        ImNodes::BeginNodeTitleBar();
        REGULAR(ImGui::TextUnformatted(node->get_name().c_str()));
        ImNodes::EndNodeTitleBar();

        for(uint32_t i = 0; i < node->get_num_inputs(); i++)
        {
            const int pin_id = get_input_hash(node, i);
            ImNodes::BeginInputAttribute(pin_id);
            ITALIC(ImGui::Text("%s", node->get_input_name(i)));
            ImNodes::EndInputAttribute();
            pin_map[pin_id] = node;
        }

        if(node->get_num_outputs() > 0)
        {
            const int pin_id = get_output_hash(node, 0);
            ImNodes::BeginOutputAttribute(pin_id);
            ITALIC(ImGui::Text("output"));
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

    if(ImGui::BeginDragDropTarget())
    {
        if(const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("OBJECT_PAYLOAD"))
        {
            Object* dropped_obj = *static_cast<Object**>(payload->Data);

            SceneGraphNode* node = SceneGraphNodesManager::get_instance().create_node("mesh");
            node->set_name(stdromano::String<>("mesh_{}", dropped_obj->get_name()));

            graph.add_node(node);

            const ImVec2 click_pos = ImGui::GetMousePos();

            ImNodes::SetNodeScreenSpacePos(node->get_id(), click_pos);
        }

        ImGui::EndDragDropTarget();
    }

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

    if(num_selected > 0)
    {
        static stdromano::Vector<int> selected_nodes;
        selected_nodes.clear();

        for(uint32_t i = 0; i < num_selected; i++)
        {
            selected_nodes.push_back(0);
        }

        ImNodes::GetSelectedNodes(selected_nodes.data());

        if(ImGui::IsKeyPressed(ImGuiKey_Delete) || ImGui::IsKeyPressed(ImGuiKey_X))
        {
            for(const int node_id : selected_nodes)
            {
                graph.remove_node(node_id);
            }
        }
        else
        {
            *current_node = graph.get_node_by_id((uint32_t)selected_nodes[0]);
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

    ImGui::End();
}

/* Node Params */

struct parameter_group
{
    stdromano::String<> base_name;
    stdromano::Vector<Parameter*> params;
    stdromano::Vector<stdromano::String<> > component_names;
};

stdromano::Vector<parameter_group> detect_parameter_groups(const stdromano::Vector<Parameter>& params)
{
    std::unordered_map<stdromano::String<>, parameter_group> groups;

    static const std::regex vec_pattern("([a-zA-Z0-9_]+)(x|y|z|w|r|g|b|a|u|v|w)$");

    for(uint32_t i = 0; i < params.size(); ++i)
    {
        const Parameter& param = params[i];
        stdromano::String<> param_name = param.get_name();
        std::string param_name_std = param_name.c_str();
        std::cmatch match;

        if(std::regex_match(param_name.c_str(), match, vec_pattern) && match.size() > 2)
        {
            stdromano::String<> base_name = match[1].str().c_str();
            stdromano::String<> component_name = match[2].str().c_str();

            if(groups.find(base_name) == groups.end())
            {
                parameter_group new_group;
                new_group.base_name = base_name;
                groups[base_name] = new_group;
            }

            groups[base_name].params.push_back(const_cast<Parameter*>(&param));
            groups[base_name].component_names.push_back(component_name);
        }
    }

    stdromano::Vector<parameter_group> result;

    for(auto& pair : groups)
    {
        if(pair.second.params.size() >= 2)
        {
            result.push_back(pair.second);
        }
    }

    return result;
}

void render_parameter(Parameter& param)
{
    const char* name = param.get_name().c_str();
    ImGui::PushID(name);

    switch(param.get_type())
    {
    case ParameterType_Int:
    {
        int value = param.get_int();
        if(ImGui::InputInt(name, &value))
        {
            param.set_int(value);
        }
        break;
    }
    case ParameterType_Float:
    {
        float value = param.get_float();
        if(ImGui::InputFloat(name, &value, 0.01f, 0.1f, "%.3f"))
        {
            param.set_float(value);
        }
        break;
    }
    case ParameterType_Bool:
    {
        bool value = param.get_bool();
        if(ImGui::Checkbox(name, &value))
        {
            param.set_bool(value);
        }
        break;
    }
    case ParameterType_String:
    {
        char buffer[256];
        const stdromano::String<>& str = param.get_string();
        strncpy(buffer, str.c_str(), sizeof(buffer) - 1);
        buffer[sizeof(buffer) - 1] = '\0';

        if(ImGui::InputText(name, buffer, sizeof(buffer)))
        {
            param.set_string(buffer);
        }
        break;
    }
    }

    ImGui::PopID();
}

void standardize_component_order(stdromano::Vector<Parameter*>& params,
                                 stdromano::Vector<stdromano::String<> >& components)
{
    if(components.size() == 2)
    {
        stdromano::Vector<stdromano::String<> > order;

        order.push_back("x");
        order.push_back("y");
        order.emplace_back("u");
        order.emplace_back("v");

        stdromano::Vector<Parameter*> ordered_params;
        stdromano::Vector<stdromano::String<> > ordered_components;

        for(uint32_t i = 0; i < order.size(); ++i)
        {
            for(uint32_t j = 0; j < components.size(); ++j)
            {
                if(components[j] == order[i])
                {
                    ordered_params.push_back(params[j]);
                    ordered_components.push_back(components[j]);
                    break;
                }
            }
        }

        if(ordered_params.size() == params.size())
        {
            params = ordered_params;
            components = ordered_components;
        }
    }
    else if(components.size() == 3)
    {
        stdromano::Vector<stdromano::String<> > order;

        order.emplace_back("x");
        order.emplace_back("y");
        order.emplace_back("z");
        order.emplace_back("r");
        order.emplace_back("g");
        order.emplace_back("b");
        order.emplace_back("u");
        order.emplace_back("v");
        order.emplace_back("w");

        stdromano::Vector<Parameter*> ordered_params;
        stdromano::Vector<stdromano::String<> > ordered_components;

        for(uint32_t i = 0; i < order.size(); ++i)
        {
            for(uint32_t j = 0; j < components.size(); ++j)
            {
                if(components[j] == order[i])
                {
                    ordered_params.push_back(params[j]);
                    ordered_components.push_back(components[j]);
                    break;
                }
            }
        }

        if(ordered_params.size() == params.size())
        {
            params = ordered_params;
            components = ordered_components;
        }
    }
    // TODO: 4 components
}

void render_parameter_group(parameter_group& group)
{
    standardize_component_order(group.params, group.component_names);

    ParameterType type = group.params[0]->get_type();

    ImGui::Text("%s:", group.base_name.c_str());
    ImGui::SameLine();
    ImGui::PushID(group.base_name.c_str());

    if(type == ParameterType_Float)
    {
        if(group.params.size() == 2)
        {
            float values[2] = {group.params[0]->get_float(), group.params[1]->get_float()};

            if(ImGui::InputFloat2("", values))
            {
                group.params[0]->set_float(values[0]);
                group.params[1]->set_float(values[1]);
            }
        }
        else if(group.params.size() == 3)
        {
            float values[3] = {group.params[0]->get_float(),
                               group.params[1]->get_float(),
                               group.params[2]->get_float()};

            if(ImGui::InputFloat3("", values))
            {
                group.params[0]->set_float(values[0]);
                group.params[1]->set_float(values[1]);
                group.params[2]->set_float(values[2]);
            }
        }
        else if(group.params.size() == 4)
        {
            float values[4] = {group.params[0]->get_float(),
                               group.params[1]->get_float(),
                               group.params[2]->get_float(),
                               group.params[3]->get_float()};

            if(ImGui::InputFloat4("", values))
            {
                group.params[0]->set_float(values[0]);
                group.params[1]->set_float(values[1]);
                group.params[2]->set_float(values[2]);
                group.params[3]->set_float(values[3]);
            }
        }
    }
    else if(type == ParameterType_Int)
    {
        if(group.params.size() == 2)
        {
            int values[2] = {group.params[0]->get_int(), group.params[1]->get_int()};

            if(ImGui::InputInt2("", values))
            {
                group.params[0]->set_int(values[0]);
                group.params[1]->set_int(values[1]);
            }
        }
        else if(group.params.size() == 3)
        {
            int values[3] = {group.params[0]->get_int(),
                             group.params[1]->get_int(),
                             group.params[2]->get_int()};

            if(ImGui::InputInt3("", values))
            {
                group.params[0]->set_int(values[0]);
                group.params[1]->set_int(values[1]);
                group.params[2]->set_int(values[2]);
            }
        }
        else if(group.params.size() == 4)
        {
            int values[4] = {group.params[0]->get_int(),
                             group.params[1]->get_int(),
                             group.params[2]->get_int(),
                             group.params[3]->get_int()};

            if(ImGui::InputInt4("", values))
            {
                group.params[0]->set_int(values[0]);
                group.params[1]->set_int(values[1]);
                group.params[2]->set_int(values[2]);
                group.params[3]->set_int(values[3]);
            }
        }
    }

    ImGui::PopID();
}

void draw_node_params(SceneGraphNode* selected_node) noexcept
{
    ImGui::Begin("Parameters");

    if(selected_node == nullptr)
    {
        ImGui::Text("No node selected");
        ImGui::End();
        return;
    }

    BOLD(ImGui::Text("%s", selected_node->get_name().c_str()));
    ImGui::SameLine();
    ImGui::Spacing();
    ImGui::SameLine();
    ITALIC(ImGui::Text("%s", selected_node->get_type_name()));

    ImGui::Separator();

    const stdromano::Vector<Parameter>& params = selected_node->get_parameters();

    auto groups = detect_parameter_groups(params);

    std::unordered_set<const Parameter*> grouped_params;
    for(uint32_t i = 0; i < groups.size(); ++i)
    {
        const parameter_group& group = groups[i];
        for(uint32_t j = 0; j < group.params.size(); ++j)
        {
            grouped_params.insert(group.params[j]);
        }
    }

    if(groups.size() > 0)
    {
        for(uint32_t i = 0; i < groups.size(); ++i)
        {
            render_parameter_group(groups[i]);
        }

        ImGui::Separator();
    }

    for(uint32_t i = 0; i < params.size(); ++i)
    {
        Parameter& param = const_cast<Parameter&>(params[i]);

        if(grouped_params.find(&param) != grouped_params.end())
        {
            continue;
        }

        render_parameter(param);
    }

    ImGui::End();
}

/* Debug Window */

void draw_debug(RenderEngine& engine) noexcept
{
    bool show = (bool)ui_state().get(UIStateFlag_Show);

    if(!show)
    {
        return;
    }

    BOLD(ImGui::Begin("Debug"));

    PUSH_REGULAR();
    ImGui::Text("FPS : %0.3f", ImGui::GetIO().Framerate);
    ImGui::Text("Sample: %u", engine.get_current_sample());
    ImGui::Text("SceneGraph state: %s", engine.get_scene_graph().is_dirty() ? "Dirty" : "Ready");

    if(ImGui::Button(engine.is_rendering() ? "Stop Render" : "Start Render"))
    {
        if(engine.is_rendering())
        {
            engine.stop_rendering();
        }
        else
        {
            engine.start_rendering(integrator_debug);
        }
    }

    int32_t current_backend = (int32_t)engine.get_setting(RenderEngineSetting_Device) - 1;

    if(ImGui::Combo("Backend", &current_backend, "CPU\0GPU"))
    {
        engine.set_setting(RenderEngineSetting_Device, (uint32_t)current_backend + 1);
    }

    POP_FONT();

    ImGui::End();
}

ROMANORENDER_NAMESPACE_END