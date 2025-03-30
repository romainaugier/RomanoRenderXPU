#include "romanorender/app_widgets.h"
#include "romanorender/app_icons.h"

#include "stdromano/filesystem.h"
#include "stdromano/logger.h"

#define STDROMANO_ENABLE_PROFILING
#include "stdromano/profiling.h"

#include <imgui.h>
#include <imnodes.h>

#include "GLFW/glfw3.h"

#include <regex>
#include <unordered_map>
#include <unordered_set>

ROMANORENDER_NAMESPACE_BEGIN

/* Resources Manager */

struct CachedFontInfo
{
    char name[64];
    size_t data_size;
};

bool cache_font_files(const char* cache_path,
                      const stdromano::HashMap<stdromano::String<>, std::pair<unsigned char*, size_t> >& font_data)
{
    FILE* file = std::fopen(cache_path, "wb");

    if(file == nullptr)
    {
        return false;
    }

    const size_t font_count = font_data.size();

    if(std::fwrite(&font_count, sizeof(font_count), 1, file) != 1)
    {
        std::fclose(file);
        return false;
    }

    for(const auto& font_pair : font_data)
    {
        CachedFontInfo info;
        std::memset(info.name, 0, sizeof(info.name));
        std::strncpy(info.name, font_pair.first.c_str(), sizeof(info.name) - 1);
        info.data_size = font_pair.second.second;

        if(std::fwrite(&info, sizeof(info), 1, file) != 1)
        {
            std::fclose(file);
            return false;
        }

        if(std::fwrite(font_pair.second.first, 1, info.data_size, file) != info.data_size)
        {
            std::fclose(file);
            return false;
        }
    }

    std::fclose(file);

    return true;
}

void UIResourcesManager::load_fonts() noexcept
{
    ImGuiIO& io = ImGui::GetIO();
    const stdromano::String<>
        cache_path = stdromano::expand_from_executable_dir("res/font_cache.bin");
    stdromano::HashMap<stdromano::String<>, std::pair<unsigned char*, size_t> > font_data;
    bool cache_exists = false;

    FILE* file = std::fopen(cache_path.c_str(), "rb");

    if(file != nullptr)
    {
        size_t font_count = 0;

        if(std::fread(&font_count, sizeof(font_count), 1, file) == 1)
        {
            cache_exists = true;

            for(size_t i = 0; i < font_count; i++)
            {
                CachedFontInfo info;

                if(std::fread(&info, sizeof(info), 1, file) != 1)
                {
                    cache_exists = false;
                    break;
                }

                unsigned char* data = static_cast<unsigned char*>(stdromano::mem_calloc(info.data_size,
                                                                                        sizeof(unsigned char)));

                if(std::fread(data, sizeof(unsigned char), info.data_size, file) != info.data_size)
                {
                    cache_exists = false;
                    break;
                }

                font_data[stdromano::String<>(info.name)] = std::make_pair(data, info.data_size);
            }
        }

        std::fclose(file);

        stdromano::log_debug("Loaded {} fonts from cache", font_data.size());
    }

    if(!cache_exists)
    {
        static const ImWchar icons_ranges[] = {ICON_MIN_FK, ICON_MAX_FK, 0};
        const stdromano::String<>
            icons_path = stdromano::expand_from_executable_dir("res/forkawesome-webfont.ttf");
        const stdromano::String<> fonts_dir = stdromano::expand_from_executable_dir("res");

        FILE* icon_file = std::fopen(icons_path.c_str(), "rb");

        if(icon_file != nullptr)
        {
            std::fseek(icon_file, 0, SEEK_END);
            const size_t icons_size = std::ftell(icon_file);
            std::rewind(icon_file);

            unsigned char* icons_data = static_cast<unsigned char*>(stdromano::mem_calloc(icons_size,
                                                                                          sizeof(unsigned char)));
            std::fread(icons_data, sizeof(unsigned char), icons_size, icon_file);
            std::fclose(icon_file);

            font_data["icons"] = std::make_pair(icons_data, icons_size);
        }

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

                    FILE* font_file = std::fopen(current_file_path.c_str(), "rb");

                    if(font_file != nullptr)
                    {
                        std::fseek(font_file, 0, SEEK_END);
                        const size_t font_size = ftell(font_file);
                        std::rewind(font_file);

                        unsigned char* font_data_buffer = static_cast<unsigned char*>(stdromano::mem_calloc(font_size,
                                                                                                            sizeof(unsigned char)));
                        std::fread(font_data_buffer, sizeof(unsigned char), font_size, font_file);
                        std::fclose(font_file);

                        font_data[font_name.lower()] = std::make_pair(font_data_buffer, font_size);
                    }
                }
            }
        }

        cache_font_files(cache_path.c_str(), font_data);
    }

    for(const auto& font_pair : font_data)
    {
        if(font_pair.first == "icons")
        {
            continue;
        }

        ImFontConfig font_config;
        font_config.FontDataOwnedByAtlas = false;

        ImFont* main_font = io.Fonts->AddFontFromMemoryTTF(font_pair.second.first,
                                                           font_pair.second.second,
                                                           16.0f,
                                                           &font_config);

        if(font_data.find("icons") != font_data.end())
        {
            ImFontConfig icons_config;
            icons_config.MergeMode = true;
            icons_config.PixelSnapH = true;
            icons_config.GlyphOffset.y = 1.0f;
            icons_config.FontDataOwnedByAtlas = false;
            static const ImWchar icons_ranges[] = {ICON_MIN_FK, ICON_MAX_FK, 0};

            io.Fonts->AddFontFromMemoryTTF(font_data["icons"].first, font_data["icons"].second, 16.0f, &icons_config, icons_ranges);
        }

        this->_fonts.insert(std::make_pair(font_pair.first, main_font));
    }

    io.Fonts->Build();

    for(auto& data : font_data)
    {
        stdromano::mem_free(data.second.first);
    }

    stdromano::log_debug("Loaded {} fonts", this->_fonts.size());
}

void UIResourcesManager::load_imgui() const noexcept
{
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

UIResourcesManager::~UIResourcesManager() {}

/* State */

ROMANORENDER_FORCE_INLINE uint32_t float_to_uint32(float value)
{
    return *reinterpret_cast<uint32_t*>(&value);
}

ROMANORENDER_FORCE_INLINE float uint32_to_float(uint32_t value)
{
    return *reinterpret_cast<float*>(&value);
}

UIState::UIState()
{
    this->_states[UIStateFlag_Show] = 1;

    this->set(UIStateFlag_ZoomLevel, float_to_uint32(1.0f));
    this->set(UIStateFlag_PanX, float_to_uint32(0.0f));
    this->set(UIStateFlag_PanY, float_to_uint32(0.0f));
}

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

/* Style */

void set_style() noexcept
{
    ImGuiStyle* style = &ImGui::GetStyle();

    style->WindowBorderSize = 0.0f;
    style->FrameBorderSize = 0.0f;
    style->ChildBorderSize = 0.0f;
    style->PopupBorderSize = 0.0f;
    style->TabBorderSize = 0.0f;
    style->WindowPadding = ImVec2(8, 8);
    style->WindowRounding = 0.0f;
    style->FramePadding = ImVec2(20, 1);
    style->FrameRounding = 0.0f;
    style->ItemSpacing = ImVec2(12, 8);
    style->ItemInnerSpacing = ImVec2(8, 6);
    style->IndentSpacing = 25.0f;
    style->ScrollbarSize = 15.0f;
    style->ScrollbarRounding = 0.0f;
    style->GrabMinSize = 5.0f;
    style->GrabRounding = 0.0f;
    style->ChildRounding = 0.0f;
    style->TabRounding = 0.0f;
    style->PopupRounding = 0.0f;
    style->GrabRounding = 0.0f;
    style->LogSliderDeadzone = 0.0f;
    style->ScrollbarRounding = 0.0f;
    style->DisplaySafeAreaPadding = ImVec2(0.0f, 0.0f);
    style->WindowMenuButtonPosition = ImGuiDir_None;

    style->Colors[ImGuiCol_Text] = ImVec4(0.80f, 0.80f, 0.83f, 1.00f);
    style->Colors[ImGuiCol_TextDisabled] = ImVec4(0.24f, 0.23f, 0.29f, 1.00f);
    style->Colors[ImGuiCol_WindowBg] = ImVec4(0.06f, 0.05f, 0.07f, 1.00f);
    style->Colors[ImGuiCol_ChildBg] = ImVec4(0.07f, 0.07f, 0.09f, 1.00f);
    style->Colors[ImGuiCol_PopupBg] = ImVec4(0.07f, 0.07f, 0.09f, 1.00f);
    style->Colors[ImGuiCol_Border] = ImVec4(0.80f, 0.80f, 0.83f, 0.88f);
    style->Colors[ImGuiCol_BorderShadow] = ImVec4(0.92f, 0.91f, 0.88f, 0.00f);
    style->Colors[ImGuiCol_FrameBg] = ImVec4(0.10f, 0.09f, 0.12f, 1.00f);
    style->Colors[ImGuiCol_FrameBgHovered] = ImVec4(0.24f, 0.23f, 0.29f, 1.00f);
    style->Colors[ImGuiCol_FrameBgActive] = ImVec4(0.56f, 0.56f, 0.58f, 1.00f);
    style->Colors[ImGuiCol_TitleBg] = ImVec4(0.10f, 0.09f, 0.12f, 1.00f);
    style->Colors[ImGuiCol_TitleBgCollapsed] = ImVec4(1.00f, 0.98f, 0.95f, 0.75f);
    style->Colors[ImGuiCol_TitleBgActive] = ImVec4(0.07f, 0.07f, 0.09f, 1.00f);
    style->Colors[ImGuiCol_MenuBarBg] = ImVec4(0.10f, 0.09f, 0.12f, 1.00f);
    style->Colors[ImGuiCol_ScrollbarBg] = ImVec4(0.10f, 0.09f, 0.12f, 1.00f);
    style->Colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.80f, 0.80f, 0.83f, 0.31f);
    style->Colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.56f, 0.56f, 0.58f, 1.00f);
    style->Colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.06f, 0.05f, 0.07f, 1.00f);
    style->Colors[ImGuiCol_CheckMark] = ImVec4(0.80f, 0.80f, 0.83f, 0.31f);
    style->Colors[ImGuiCol_SliderGrab] = ImVec4(0.80f, 0.80f, 0.83f, 0.31f);
    style->Colors[ImGuiCol_SliderGrabActive] = ImVec4(0.06f, 0.05f, 0.07f, 1.00f);
    style->Colors[ImGuiCol_Button] = ImVec4(0.10f, 0.09f, 0.12f, 1.00f);
    style->Colors[ImGuiCol_ButtonHovered] = ImVec4(0.24f, 0.23f, 0.29f, 1.00f);
    style->Colors[ImGuiCol_ButtonActive] = ImVec4(0.56f, 0.56f, 0.58f, 1.00f);
    style->Colors[ImGuiCol_Header] = ImVec4(0.10f, 0.09f, 0.12f, 1.00f);
    style->Colors[ImGuiCol_HeaderHovered] = ImVec4(0.56f, 0.56f, 0.58f, 1.00f);
    style->Colors[ImGuiCol_HeaderActive] = ImVec4(0.06f, 0.05f, 0.07f, 1.00f);
    style->Colors[ImGuiCol_ResizeGrip] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    style->Colors[ImGuiCol_ResizeGripHovered] = ImVec4(0.56f, 0.56f, 0.58f, 1.00f);
    style->Colors[ImGuiCol_ResizeGripActive] = ImVec4(0.06f, 0.05f, 0.07f, 1.00f);
    style->Colors[ImGuiCol_PlotLines] = ImVec4(0.40f, 0.39f, 0.38f, 0.63f);
    style->Colors[ImGuiCol_PlotLinesHovered] = ImVec4(0.25f, 1.00f, 0.00f, 1.00f);
    style->Colors[ImGuiCol_PlotHistogram] = ImVec4(0.40f, 0.39f, 0.38f, 0.63f);
    style->Colors[ImGuiCol_PlotHistogramHovered] = ImVec4(0.25f, 1.00f, 0.00f, 1.00f);
    style->Colors[ImGuiCol_TextSelectedBg] = ImVec4(0.25f, 1.00f, 0.00f, 0.43f);
    style->Colors[ImGuiCol_TabUnfocusedActive] = ImVec4(1.0f, 1.0f, 1.0f, 0.04f);
    style->Colors[ImGuiCol_TabHovered] = ImVec4(0.718f, 0.718f, 0.718f, 0.6f);
    style->Colors[ImGuiCol_TabActive] = ImVec4(1.0f, 1.0f, 1.0f, 0.2f);
    style->Colors[ImGuiCol_DockingPreview] = ImVec4(1.0f, 0.45f, 0.0f, 1.0f);
    style->Colors[ImGuiCol_DockingEmptyBg] = ImVec4(0.0f, 0.0f, 0.0f, 1.0f);
    style->Colors[ImGuiCol_MenuBarBg] = ImVec4(0.07f, 0.07f, 0.09f, 1.00f);
}

/* RenderBuffer */

void render_buffer_reset_transform() noexcept
{
    ui_state().set(UIStateFlag_ZoomLevel, float_to_uint32(1.0f));
    ui_state().set(UIStateFlag_PanX, float_to_uint32(0.0f));
    ui_state().set(UIStateFlag_PanY, float_to_uint32(0.0f));
}

void render_buffer_fit_transform(const RenderBuffer* render_buffer) noexcept
{
    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);

    const float x_zoom = (float)viewport[2] / render_buffer->get_xsize();
    const float y_zoom = (float)viewport[2] / render_buffer->get_ysize();

    ui_state().set(UIStateFlag_ZoomLevel, float_to_uint32(maths::minf(x_zoom, y_zoom)));
    ui_state().set(UIStateFlag_PanX, float_to_uint32(0.0f));
    ui_state().set(UIStateFlag_PanY, float_to_uint32(0.0f));
}

void render_buffer_handle_scroll(const double x_offset, const double y_offset, const double mouse_x, const double mouse_y) noexcept
{
    if(ImGui::GetIO().WantCaptureMouse || !ui_state().get(UIStateFlag_TextureMouseHover))
    {
        return;
    }

    float zoom_level = uint32_to_float(ui_state().get(UIStateFlag_ZoomLevel));
    float pan_x = uint32_to_float(ui_state().get(UIStateFlag_PanX));
    float pan_y = uint32_to_float(ui_state().get(UIStateFlag_PanY));

    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);

    float center_x = viewport[2] / 2.0f;
    float center_y = viewport[3] / 2.0f;

    float mouse_rel_x = static_cast<float>(mouse_x) - center_x;
    float mouse_rel_y = static_cast<float>(mouse_y) - center_y;

    float old_zoom = zoom_level;

    constexpr float zoom_speed = 0.1f;
    zoom_level += static_cast<float>(y_offset) * zoom_speed;

    zoom_level = maths::maxf(0.1f, maths::minf(10.0f, zoom_level));

    float zoom_ratio = zoom_level / old_zoom;

    pan_x = mouse_rel_x + (pan_x - mouse_rel_x) * zoom_ratio;
    pan_y = mouse_rel_y + (pan_y - mouse_rel_y) * zoom_ratio;

    ui_state().set(UIStateFlag_ZoomLevel, float_to_uint32(zoom_level));
    ui_state().set(UIStateFlag_PanX, float_to_uint32(pan_x));
    ui_state().set(UIStateFlag_PanY, float_to_uint32(pan_y));
}

void render_buffer_handle_mouse_button(const int button, const int action, const double x_pos, const double y_pos) noexcept
{
    if(ImGui::GetIO().WantCaptureMouse || !ui_state().get(UIStateFlag_TextureMouseHover))
    {
        return;
    }

    if(button == GLFW_MOUSE_BUTTON_LEFT)
    {
        if(action == GLFW_PRESS)
        {
            ui_state().set(UIStateFlag_LastMouseX, float_to_uint32(static_cast<float>(x_pos)));
            ui_state().set(UIStateFlag_LastMouseY, float_to_uint32(static_cast<float>(y_pos)));
            ui_state().set(UIStateFlag_TexturePanActive, 1);
        }
        else if(action == GLFW_RELEASE)
        {
            ui_state().set(UIStateFlag_TexturePanActive, 0);
        }
    }
}

void render_buffer_handle_mouse_move(const double x_pos, const double y_pos) noexcept
{
    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);

    bool is_hovering = !ImGui::GetIO().WantCaptureMouse && x_pos >= viewport[0]
                       && x_pos < viewport[0] + viewport[2] && y_pos >= viewport[1]
                       && y_pos < viewport[1] + viewport[3];

    ui_state().set(UIStateFlag_TextureMouseHover, is_hovering ? 1 : 0);

    if(ui_state().get(UIStateFlag_TexturePanActive))
    {
        const float last_mouse_x = uint32_to_float(ui_state().get(UIStateFlag_LastMouseX));
        const float last_mouse_y = uint32_to_float(ui_state().get(UIStateFlag_LastMouseY));

        const float dx = (float)x_pos - last_mouse_x;
        const float dy = (float)y_pos - last_mouse_y;

        const float pan_x = uint32_to_float(ui_state().get(UIStateFlag_PanX)) + dx;
        const float pan_y = uint32_to_float(ui_state().get(UIStateFlag_PanY)) + dy;

        ui_state().set(UIStateFlag_PanX, float_to_uint32(pan_x));
        ui_state().set(UIStateFlag_PanY, float_to_uint32(pan_y));
        ui_state().set(UIStateFlag_LastMouseX, float_to_uint32((float)x_pos));
        ui_state().set(UIStateFlag_LastMouseY, float_to_uint32((float)y_pos));
    }
}

void render_buffer_draw(const RenderBuffer* render_buffer) noexcept
{
    const float zoom_level = uint32_to_float(ui_state().get(UIStateFlag_ZoomLevel, float_to_uint32(1.0f)));
    const float pan_x = uint32_to_float(ui_state().get(UIStateFlag_PanX, float_to_uint32(0.0f)));
    const float pan_y = uint32_to_float(ui_state().get(UIStateFlag_PanY, float_to_uint32(0.0f)));

    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);

    glBindFramebuffer(GL_READ_FRAMEBUFFER, render_buffer->get_gl_framebuffer());
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);

    const int32_t dst_width = (int32_t)(render_buffer->get_xsize() * zoom_level);
    const int32_t dst_height = (int32_t)(render_buffer->get_ysize() * zoom_level);

    const int32_t center_x = viewport[2] / 2;
    const int32_t center_y = viewport[3] / 2;

    const int32_t dst_x = (int32_t)(center_x - (dst_width / 2) + pan_x);
    const int32_t dst_y = (int32_t)(center_y + (dst_height / 2) - pan_y);

    glBlitFramebuffer(0,
                      0,
                      render_buffer->get_xsize(),
                      render_buffer->get_ysize(),
                      dst_x,
                      dst_y,
                      dst_x + dst_width,
                      dst_y - dst_height,
                      GL_COLOR_BUFFER_BIT,
                      GL_LINEAR);

    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
}

/* Objects */

void draw_objects() noexcept
{
    bool show = (bool)ui_state().get(UIStateFlag_Show);

    if(!show)
    {
        return;
    }

    BOLD(ImGui::Begin(ICON_FK_CUBES " Objects", &show, ImGuiWindowFlags_MenuBar));

    if(ImGui::BeginMenuBar())
    {
        if(ImGui::BeginMenu(ICON_FK_PLUS " Add"))
        {
            if(ImGui::BeginMenu(ICON_FK_LIGHTBULB_O " Light"))
            {
                if(ImGui::MenuItem("Square Light"))
                {
                    objects_manager().add_light(LightType_Square);
                }
                if(ImGui::MenuItem("Dome Light"))
                {
                    objects_manager().add_light(LightType_Dome);
                }
                if(ImGui::MenuItem("Distant Light"))
                {
                    objects_manager().add_light(LightType_Distant);
                }
                if(ImGui::MenuItem("Circle Light"))
                {
                    objects_manager().add_light(LightType_Circle);
                }
                if(ImGui::MenuItem("Spherical Light"))
                {
                    objects_manager().add_light(LightType_Spherical);
                }

                ImGui::EndMenu();
            }

            if(ImGui::MenuItem(ICON_FK_VIDEO_CAMERA " Camera"))
            {
               
            }

            ImGui::EndMenu();
        }

        ImGui::EndMenuBar();
    }

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
        else if(ObjectLight* light = dynamic_cast<ObjectLight*>(obj))
        {
            ImGui::Text(ICON_FK_LIGHTBULB_O " ");
            ImGui::SameLine();
        }

        if(ImGui::Selectable(obj->get_path().c_str(), selected_object == obj))
        {
            selected_object = obj;
        }

        if(ImGui::BeginDragDropSource(ImGuiDragDropFlags_SourceAllowNullID))
        {
            ImGui::SetDragDropPayload("OBJECT_PAYLOAD", &obj, sizeof(Object*));

            ImGui::Text(obj->get_name().c_str());
            ImGui::EndDragDropSource();
        }

        if(ImGui::BeginPopupContextItem())
        {
            if(ImGui::MenuItem("Delete"))
            {
                objects_manager().remove_object(obj);

                if(selected_object == obj)
                {
                    selected_object = nullptr;
                }
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

#define DIRTY_NODE_COLOR IM_COL32(70, 70, 70, 255)
#define NONDIRTY_NODE_COLOR IM_COL32(41, 74, 122, 255)

void draw_scenegraph(SceneGraph& graph, SceneGraphNode** current_node) noexcept
{
    bool show = (bool)ui_state().get(UIStateFlag_Show);

    if(!show)
    {
        return;
    }

    BOLD(ImGui::Begin(ICON_FK_HASHNODE " Scenegraph", &show));

    if(ImGui::Button("Save"))
    {
        stdromano::log_debug("{}", serialize_graph(graph).dump(4));
    }

    ImGui::SameLine(0.0f, 50.0f);

    if(ImGui::Button("Load"))
    {

    }

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
        if(node->is_dirty())
        {
            ImNodes::PushColorStyle(ImNodesCol_TitleBar, DIRTY_NODE_COLOR);
        }
        else
        {
            ImNodes::PushColorStyle(ImNodesCol_TitleBar, NONDIRTY_NODE_COLOR);
        }

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

        ImNodes::PopColorStyle();

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

            SceneGraphNode* node = nullptr;

            if(const ObjectMesh* mesh = dynamic_cast<ObjectMesh*>(dropped_obj))
            {
                node = graph.create_node("mesh");
                node->set_name(stdromano::String<>("mesh_{}", mesh->get_name()));
                node->get_parameter("path_pattern")->set_string(mesh->get_path());
            }

            if(const ObjectCamera* mesh = dynamic_cast<ObjectCamera*>(dropped_obj))
            {
                node = graph.create_node("camera");
                node->set_name(stdromano::String<>("camera_{}", mesh->get_name()));
                node->get_parameter("path_pattern")->set_string(mesh->get_path());
            }

            if(const ObjectLight* light = dynamic_cast<ObjectLight*>(dropped_obj))
            {
                switch(light->get_light()->get_type())
                {
                    case LightType_Square:
                    {
                        node = graph.create_node("square_light");
                        node->set_name(light->get_name().copy());
                        node->get_parameter("__light_uuid")->set_int(light->get_uuid_32());
                        break;
                    }
                    case LightType_Dome:
                    {
                        node = graph.create_node("dome_light");
                        node->set_name(light->get_name().copy());
                        node->get_parameter("__light_uuid")->set_int(light->get_uuid_32());
                        break;
                    }
                    case LightType_Distant:
                    {
                        node = graph.create_node("distant_light");
                        node->set_name(light->get_name().copy());
                        node->get_parameter("__light_uuid")->set_int(light->get_uuid_32());
                        break;
                    }
                    case LightType_Circle:
                    {
                        node = graph.create_node("circle_light");
                        node->set_name(light->get_name().copy());
                        node->get_parameter("__light_uuid")->set_int(light->get_uuid_32());
                        break;
                    }
                    case LightType_Spherical:
                    {
                        node = graph.create_node("spherical_light");
                        node->set_name(light->get_name().copy());
                        node->get_parameter("__light_uuid")->set_int(light->get_uuid_32());
                        break;
                    }
                }
            }

            if(node != nullptr)
            {
                const ImVec2 click_pos = ImGui::GetMousePos();

                ImNodes::SetNodeScreenSpacePos(node->get_id(), click_pos);
            }
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

        if(is_editor_hovered && (ImGui::IsKeyPressed(ImGuiKey_Delete) || ImGui::IsKeyPressed(ImGuiKey_X)))
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

    if(ImNodes::NumSelectedLinks() > 0 && is_editor_hovered
       && (ImGui::IsKeyPressed(ImGuiKey_Delete) || ImGui::IsKeyPressed(ImGuiKey_X)))
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
    if(param.get_name().startswith("__"))
    {
        return;
    }

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
    if(group.base_name.startswith("__"))
    {
        return;
    }

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
    bool show = (bool)ui_state().get(UIStateFlag_Show);

    if(!show)
    {
        return;
    }

    ImGui::Begin(ICON_FK_WRENCH " Parameters");

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
            engine.start_rendering(integrator_pathtrace);
        }
    }

    int32_t current_backend = (int32_t)engine.get_setting(RenderEngineSetting_Device) - 1;

    if(ImGui::Combo("Backend", &current_backend, "CPU\0GPU\0"))
    {
        engine.set_setting(RenderEngineSetting_Device, (uint32_t)current_backend + 1);
    }

    int32_t xsize = (int32_t)engine.get_setting(RenderEngineSetting_XSize);

    if(ImGui::InputInt("Width", &xsize))
    {
        engine.set_setting(RenderEngineSetting_XSize, (uint32_t)xsize);
    }

    int32_t ysize = (int32_t)engine.get_setting(RenderEngineSetting_YSize);

    if(ImGui::InputInt("Height", &ysize))
    {
        engine.set_setting(RenderEngineSetting_YSize, (uint32_t)ysize);
    }

    int32_t max_bounces = engine.get_setting(RenderEngineSetting_MaxBounces);

    if(ImGui::InputInt("Max Bounces", &max_bounces))
    {
        engine.set_setting(RenderEngineSetting_MaxBounces, (uint32_t)max_bounces);
    }

    int32_t max_samples = engine.get_setting(RenderEngineSetting_MaxSamples);

    if(ImGui::InputInt("Max Samples", &max_samples))
    {
        engine.set_setting(RenderEngineSetting_MaxSamples, (uint32_t)max_samples);
    }

    ImGui::Separator();
    BOLD(ImGui::Text("Memory Usage"));

    char obj_manager_mem_usage_fmt[16];
    stdromano::format_byte_size((float)objects_manager().get_memory_usage(), obj_manager_mem_usage_fmt);

    ImGui::Text("Objects Manager: %s", obj_manager_mem_usage_fmt);

    char scenegraph_mem_usage_fmt[16];
    stdromano::format_byte_size((float)engine.get_scene_graph().get_memory_usage(), scenegraph_mem_usage_fmt);

    ImGui::Text("SceneGraph: %s", scenegraph_mem_usage_fmt);

    POP_FONT();

    ImGui::End();
}

ROMANORENDER_NAMESPACE_END