#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <stdio.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#if defined(_MSC_VER) && (_MSC_VER >= 1900) && !defined(IMGUI_DISABLE_WIN32_FUNCTIONS)
#pragma comment(lib, "legacy_stdio_definitions")
#endif

#include "romanorender/app.h"
#include "romanorender/renderengine.h"

#define FLYTHROUGH_CAMERA_IMPLEMENTATION
#include "romanorender/flythrough_camera.h"

ROMANORENDER_NAMESPACE_BEGIN

// GLFW Callbacks and shortcuts handling
inline void glfw_error_callback(int error, const char* description) noexcept
{
    fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

int application(int argc, char** argv)
{
    // Setup window
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit())
        return 1;

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

    glfwMakeContextCurrent(window);
    glfwSwapInterval(0); // Enable vsync

    // Initialize OpenGL loader
    bool err = glewInit() != 0;

    if (err)
    {
        fprintf(stderr, "Failed to initialize OpenGL loader!\n");
        return 1;
    }

    RenderEngine render_engine(xres, yres);

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;

    ImGuiStyle* style = &ImGui::GetStyle();

    style->Colors[ImGuiCol_DockingEmptyBg] = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);
    style->Colors[ImGuiCol_WindowBg] = ImVec4(0.0f, 0.0f, 0.0f, 0.0f);

    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

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
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();
        glClearColor(clear_color.x, clear_color.y, clear_color.z, clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);

        if(display_w != render_engine.get_setting(RenderEngineSetting_XSize) || 
           display_h != render_engine.get_setting(RenderEngineSetting_YSize))
        {
            render_engine.set_setting(RenderEngineSetting_XSize, display_w, true);
            render_engine.set_setting(RenderEngineSetting_YSize, display_h, false);
        }

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        ImGui::DockSpaceOverViewport(ImGui::GetMainViewport());

        render_engine.render_sample(nullptr);
        render_engine.get_renderbuffer()->blit_default_gl_buffer();

        // Info Window
        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.1f, 0.1f, 0.1f, 1.0f));
        ImGui::Begin("Debug");
        {
            ImGui::Text("FPS : %0.3f", ImGui::GetIO().Framerate);
            ImGui::Text("Sample: %u", render_engine.get_current_sample());

            ImGui::End();
        }
        ImGui::PopStyleColor();

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        ImGui::EndFrame();
        
        glfwSwapBuffers(window);
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}

ROMANORENDER_NAMESPACE_END