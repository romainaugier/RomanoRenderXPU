#pragma once

#if !defined(__ROMANORENDER_APP_WIDGETS)
#define __ROMANORENDER_APP_WIDGETS

#include "romanorender/scenegraph.h"

ROMANORENDER_NAMESPACE_BEGIN

class ROMANORENDER_API IconsManager
{
public:
    static IconsManager& get_instance()
    {
        static IconsManager myInstance;

        return myInstance;
    }

    IconsManager(IconsManager const&) = delete;
    IconsManager(IconsManager&&) = delete;
    IconsManager& operator=(IconsManager const&) = delete;
    IconsManager& operator=(IconsManager&&) = delete;

    stdromano::String<> get_code_point(const stdromano::String<>& name) const noexcept
    {
        const auto& it = this->_icons_lookup.find(name);
        return it == this->_icons_lookup.end() ? "" : it->second;
    }

    void initialize() noexcept { ROMANORENDER_NOOP; }

private:
    IconsManager();

    ~IconsManager() {}

    stdromano::HashMap<stdromano::String<>, stdromano::String<> > _icons_lookup;
};

ROMANORENDER_API void draw_objects() noexcept;

ROMANORENDER_API void draw_scenegraph(SceneGraph& graph) noexcept;

ROMANORENDER_NAMESPACE_END

#endif /* !defined(__ROMANORENDER_APP_WIDGETS) */
