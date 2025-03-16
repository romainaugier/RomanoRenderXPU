#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "imnodes.h"

#include <stdio.h>

#include <GL/glew.h>

#include "romanorender/app.h"
#include "romanorender/app_widgets.h"
#include "romanorender/renderengine.h"

#define FLYTHROUGH_CAMERA_IMPLEMENTATION
#include "romanorender/flythrough_camera.h"

#include "stdromano/logger.h"

ROMANORENDER_NAMESPACE_BEGIN

// GLFW Callbacks

void glfw_error_callback(int error, const char* desc) noexcept { stdromano::log_error("GLFW: {} ({})", desc, error); }

void glfw_drop_event_callback(GLFWwindow* user_ptr, int count, const char** paths) noexcept
{
    for(size_t i = 0; i < (size_t)count; i++)
    {
        const stdromano::String<> file_path = stdromano::String<>::make_ref(paths[i], std::strlen(paths[i]));

        if(file_path.endswith(".obj"))
        {
            if(!objects_from_obj_file(file_path.data()))
            {
                stdromano::log_error("Error while loading OBJ file: {}", file_path);
            }
        }
        else if(file_path.endswith(".abc"))
        {
            if(!objects_from_abc_file(file_path.data()))
            {
                stdromano::log_error("Error while loading Alembic file: {}", file_path);
            }
        }
    }
}

void glfw_key_event_callback(GLFWwindow* user_ptr, int key, int scancode, int action, int mods) noexcept
{
    switch(key)
    {
    case GLFW_KEY_I:
    {
        if(action == GLFW_PRESS)
        {
            ui_state().toggle(UIStateFlag_Show);
        }
    }
    }
}

// APP entry point

int application(int argc, char** argv)
{
    stdromano::set_log_level(stdromano::LogLevel::Debug);

    glfwSetErrorCallback(glfw_error_callback);

    if(!glfwInit())
    {
        return 1;
    }

    const char* glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

    constexpr int xres = 1280;
    constexpr int yres = 720;

    // Create window with graphics context
    GLFWwindow* window = glfwCreateWindow(xres, yres, "RomanoRenderXPU", NULL, NULL);

    if(window == NULL)
    {
        return 1;
    }

    glfwSetKeyCallback(window, glfw_key_event_callback);
    glfwSetDropCallback(window, glfw_drop_event_callback);

    glfwMakeContextCurrent(window);
    glfwSwapInterval(0); // Enable vsync

    // Initialize OpenGL loader
    GLenum err = glewInit();

    if(err != GLEW_OK)
    {
        stdromano::log_error("Failed to initialize OpenGL : {}",
                             reinterpret_cast<const char*>(glewGetErrorString(err)));
        glfwTerminate();
        return 1;
    }

    RenderEngine render_engine(xres, yres, false);

    glfwSetWindowUserPointer(window, (void*)&render_engine);

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImNodes::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    (void)io;

    UIResourcesManager::get_instance().initialize();

    ImGuiStyle* style = &ImGui::GetStyle();

    style->Colors[ImGuiCol_DockingEmptyBg] = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
    style->Colors[ImGuiCol_WindowBg] = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);

    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

    io.FontDefault = ui_res_manager().get_font("regular");

    // Setup Platform/Renderer bindings
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // Our state
    bool show_demo_window = true;
    bool show_another_window = false;
    ImVec4 clear_color = ImVec4(0.06f, 0.05f, 0.04f, 0.00f);

    double oldCursorX, oldCursorY;
    glfwGetCursorPos(window, &oldCursorX, &oldCursorY);

    // Main loop
    while(!glfwWindowShouldClose(window))
    {
        glfwPollEvents();
        glClearColor(clear_color.x, clear_color.y, clear_color.z, clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);

        if(display_w != render_engine.get_setting(RenderEngineSetting_XSize)
           || display_h != render_engine.get_setting(RenderEngineSetting_YSize))
        {
            render_engine.set_setting(RenderEngineSetting_XSize, display_w, true);
            render_engine.set_setting(RenderEngineSetting_YSize, display_h, false);
        }

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        if(render_engine.is_rendering())
        {
            render_engine.update_gl_texture();
            render_engine.get_renderbuffer()->blit_default_gl_buffer();
        }

        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.1f, 0.1f, 0.1f, 1.0f));

        draw_debug(render_engine);

        draw_objects();

        draw_scenegraph(render_engine.get_scene_graph());

        ImGui::PopStyleColor();

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        ImGui::EndFrame();

        glfwSwapBuffers(window);
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImNodes::DestroyContext();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}

ROMANORENDER_NAMESPACE_END