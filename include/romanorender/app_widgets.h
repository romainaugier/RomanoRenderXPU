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

private:
    UIResourcesManager();

    ~UIResourcesManager() {}

    void load_fonts() noexcept;

    stdromano::HashMap<stdromano::String<>, ImFont*> _fonts;
};

#define ui_res_manager() UIResourcesManager::get_instance()

enum UIStateFlag_ : uint32_t
{
    UIStateFlag_Show = 0x1,
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

    uint32_t get(const uint32_t state) const noexcept
    {
        const auto& it = this->_states.find(state);
        return it == this->_states.end() ? 0 : it->second;
    }

    void set(const uint32_t state, const uint32_t value) noexcept { this->_states[state] = value; }

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

ROMANORENDER_API void draw_objects() noexcept;

ROMANORENDER_API void draw_scenegraph(SceneGraph& graph, SceneGraphNode** current_node) noexcept;

ROMANORENDER_API void draw_node_params(SceneGraphNode* node) noexcept;

ROMANORENDER_API void draw_debug(RenderEngine& engine) noexcept;

ROMANORENDER_NAMESPACE_END

#endif /* !defined(__ROMANORENDER_APP_WIDGETS) */
