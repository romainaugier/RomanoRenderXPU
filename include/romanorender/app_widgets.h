#pragma once

#if !defined(__ROMANORENDER_APP_WIDGETS)
#define __ROMANORENDER_APP_WIDGETS

#include "romanorender/renderengine.h"

#include "imgui.h"

ROMANORENDER_NAMESPACE_BEGIN

class ROMANORENDER_API UIResourcesManager
{
public:
    static UIResourcesManager& get_instance()
    {
        static UIResourcesManager myInstance;

        return myInstance;
    }

    UIResourcesManager(UIResourcesManager const&) = delete;
    UIResourcesManager(UIResourcesManager&&) = delete;
    UIResourcesManager& operator=(UIResourcesManager const&) = delete;
    UIResourcesManager& operator=(UIResourcesManager&&) = delete;

    void initialize() noexcept { ROMANORENDER_NOOP; }

    ImFont* get_font(const stdromano::String<>& font_name) const noexcept
    {
        const auto& it = this->_fonts.find(font_name);
        return it == this->_fonts.end() ? nullptr : it->second;
    }

    void load_imgui() const noexcept;

    void save_imgui() const noexcept;

private:
    UIResourcesManager();

    ~UIResourcesManager();

    void load_fonts() noexcept;

    stdromano::HashMap<stdromano::String<>, ImFont*> _fonts;
};

#define ui_res_manager() UIResourcesManager::get_instance()

enum UIStateFlag_ : uint32_t
{
    UIStateFlag_Show,

    UIStateFlag_TexturePanActive,
    UIStateFlag_TextureMouseHover,
    UIStateFlag_ZoomLevel,
    UIStateFlag_PanX,
    UIStateFlag_PanY,
    UIStateFlag_LastMouseX,
    UIStateFlag_LastMouseY,
};

class ROMANORENDER_API UIState
{
public:
    static UIState& get_instance()
    {
        static UIState myInstance;

        return myInstance;
    }

    UIState(UIState const&) = delete;
    UIState(UIState&&) = delete;
    UIState& operator=(UIState const&) = delete;
    UIState& operator=(UIState&&) = delete;

    ROMANORENDER_FORCE_INLINE void set(const uint32_t state, const uint32_t value) noexcept { this->_states[state] = value; }

    ROMANORENDER_FORCE_INLINE uint32_t get(const uint32_t state, const uint32_t default_value = 0) const noexcept { const auto& it = this->_states.find(state); return it == this->_states.end() ? default_value : it.value(); }

    void toggle(const uint32_t state)
    {
        const auto& it = this->_states.find(state);
        const uint32_t previous = it == this->_states.end() ? 0 : it->second;
        this->_states[state] = previous ^ 1UL;
    }

private:
    UIState();

    ~UIState() {}

    stdromano::HashMap<uint32_t, uint32_t> _states;
};

#define ui_state() UIState::get_instance()

ROMANORENDER_API void set_style() noexcept;

ROMANORENDER_API void render_buffer_reset_transform() noexcept;

ROMANORENDER_API void render_buffer_fit_transform(const RenderBuffer* render_buffer) noexcept;

ROMANORENDER_API void render_buffer_handle_scroll(const double x_offset, 
                                                  const double y_offset,
                                                  const double mouse_x,
                                                  const double mouse_y) noexcept;

ROMANORENDER_API void render_buffer_handle_mouse_button(const int button, const int action, const double x_pos, const double y_pos) noexcept;

ROMANORENDER_API void render_buffer_handle_mouse_move(const double x_pos, const double y_pos) noexcept;

ROMANORENDER_API void render_buffer_draw(const RenderBuffer* render_buffer) noexcept;

ROMANORENDER_API void draw_objects() noexcept;

ROMANORENDER_API void draw_scenegraph(SceneGraph& graph, SceneGraphNode** current_node) noexcept;

ROMANORENDER_API void draw_node_params(SceneGraphNode* node) noexcept;

ROMANORENDER_API void draw_debug(RenderEngine& engine) noexcept;

ROMANORENDER_NAMESPACE_END

#endif /* !defined(__ROMANORENDER_APP_WIDGETS) */
