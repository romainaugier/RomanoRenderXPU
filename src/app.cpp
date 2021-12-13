#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <stdio.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#if defined(_MSC_VER) && (_MSC_VER >= 1900) && !defined(IMGUI_DISABLE_WIN32_FUNCTIONS)
#pragma comment(lib, "legacy_stdio_definitions")
#endif

#include "app.h"

#define FLYTHROUGH_CAMERA_IMPLEMENTATION
#include "flythrough_camera.h"

#define NUM_SPHERES 200
#define NUM_MATS 120

// GLFW Callbacks and shortcuts handling
inline void glfw_error_callback(int error, const char* description) noexcept
{
    fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

int application(int argc, char** argv)
{
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

    // Build scene
    Allocators allocators;

    Material** materials = new Material*[NUM_MATS];

    MaterialDiffuse* diffuse = new MaterialDiffuse();
    diffuse->m_Color = vec3(1.0f);
    diffuse->m_Id = 0;
    diffuse->m_Type = MaterialType_Diffuse;
    materials[0] = diffuse;

    MaterialDielectric* dielectric = new MaterialDielectric();
    dielectric->m_Color = vec3(1.0f);
    dielectric->m_Id = 1;
    dielectric->m_Type = MaterialType_Dielectric;
    materials[1] = dielectric;

    MaterialReflective* reflective = new MaterialReflective();
    reflective->m_Color = vec3(1.0f);
    reflective->m_Roughness = 0.0f;
    reflective->m_Id = 2;
    reflective->m_Type = MaterialType_Reflective;
    materials[2] = reflective;

    for(uint32_t i = 3; i < NUM_MATS; i++)
    {
        const uint8_t type = maths::to_int(WangHashSampler(i + 3829) * 2);

        const vec3 randomColor = vec3(maths::fit01(WangHashSampler(i + 3214), 0.0f, 1.1f), 
                                      maths::fit01(WangHashSampler(i + 43723), 0.0f, 1.1f),
                                      maths::fit01(WangHashSampler(i + 3217), 0.0f, 1.1f));

        if(type == 0)
        {
            diffuse = new MaterialDiffuse();
            diffuse->m_Color = randomColor;
            diffuse->m_Id = i;
            diffuse->m_Type = MaterialType_Diffuse;
            materials[i] = diffuse;
        }
        else if(type == 1)
        {
            reflective = new MaterialReflective();
            reflective->m_Color = vec3(maths::fit01(WangHashSampler(i + 84329), 0.5f, 1.0f), 
                                       maths::fit01(WangHashSampler(i + 73281), 0.5f, 1.0f),
                                       maths::fit01(WangHashSampler(i + 32190), 0.5f, 1.0f));

            reflective->m_Roughness = maths::fit01(WangHashSampler(i + 83123), 0.0f, 0.1f);
            reflective->m_Id = i;
            reflective->m_Type = MaterialType_Reflective;
            materials[i] = reflective;
        }
        else if(type == 2)
        {
            dielectric = new MaterialDielectric();
            dielectric->m_Color = vec3(maths::fit01(WangHashSampler(i + 47382), 0.5f, 1.0f), 
                                       maths::fit01(WangHashSampler(i + 9328), 0.5f, 1.0f),
                                       maths::fit01(WangHashSampler(i + 24821), 0.5f, 1.0f));
            dielectric->m_Ior = 1.33f;
            dielectric->m_Id = i;
            dielectric->m_Type = MaterialType_Dielectric;
            materials[i] = dielectric;
        }
    }

    std::vector<Sphere> spheres;
    spheres.reserve(NUM_SPHERES);

    spheres.emplace_back(vec3(0.0f, -10000.0f, 0.0f), 10000.0f, 0, 0);

    spheres.emplace_back(vec3(0.0f, 5.0f, 0.0f), 5.0f, 1, 0);
    spheres.emplace_back(vec3(-15.0f, 5.0f, 0.0f), 5.0f, 2, 1);
    spheres.emplace_back(vec3(15.0f, 5.0f, 0.0f), 5.0f, 3, 2);

    for (uint32_t i = 4; i < NUM_SPHERES; i++)
    {
        const float radius = 1.0f;

        const uint32_t materialId = maths::to_int(WangHashSampler(i + 481923) * ((NUM_MATS - 1)));

        const vec3 position = vec3(maths::fit01(WangHashSampler(i), -50.0f, 50.0f), 
                                   radius,
                                   maths::fit01(WangHashSampler((i + 1 )* 321), -50.0f, 50.0f));

        spheres.emplace_back(Sphere(position, radius, i, materialId));
    }

    auto start = get_time();
    
    Accelerator accelerator = BuildAccelerator(spheres, allocators);

    auto end = get_time();

    float elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    printf("Acceleration structure building time : %0.3f ms\n", elapsed);
    
    // Setup window
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit())
        return 1;

    const char* glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

    constexpr int xres = 1280;
    constexpr int yres = 720;

    const uint32_t* blueNoisePtr = LoadBlueNoise();

    // Create window with graphics context
    GLFWwindow* window = glfwCreateWindow(xres, yres, "SphereTracer", NULL, NULL);
    if (window == NULL)
        return 1;
    glfwMakeContextCurrent(window);
    glfwSwapInterval(0); // Enable vsync

    // Initialize OpenGL loader
    bool err = glewInit() != 0;

    if (err)
    {
        fprintf(stderr, "Failed to initialize OpenGL loader!\n");
        return 1;
    }

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


    Camera cam(vec3(0.0f, 0.0f, 30.0f), vec3(0.0f, 0.0f, 0.0f), 50, xres, yres);
    cam.SetTransform();

    Settings settings;
    settings.xres = xres;
    settings.yres = yres;


    color* renderBuffer = new color[xres * yres];

    Tiles tiles;
    GenerateTiles(tiles, settings);
    
    GLuint render_view_texture;

    glGenTextures(1, &render_view_texture);
    glBindTexture(GL_TEXTURE_2D, render_view_texture);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, xres, yres, 0, GL_RGB, GL_FLOAT, renderBuffer);

    glBindTexture(GL_TEXTURE_2D, 0);

    GLuint myFbo{};
    glGenFramebuffers(1, &myFbo);
    glBindFramebuffer(GL_FRAMEBUFFER, myFbo);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, render_view_texture, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    Shader shader;

    uint32_t samples = 1;
    bool edited = 0;
    bool render = false;
    static bool drawBvh = false;
    float elapsedBvh = 0.0f;
    elapsed = 0.0f;
    float renderSeconds = 0.0f;

    // Flythrough camera
    float pos[3] = { 0.0f, 0.0f, 0.0f };
    float look[3] = { 0.0f, 0.0f, 1.0f };
    const float up[3] = { 0.0f, 1.0f, 0.0f };

    double oldCursorX, oldCursorY;
    glfwGetCursorPos(window, &oldCursorX, &oldCursorY);

    // Main loop
    while (!glfwWindowShouldClose(window))
    {
        // Reset the render accumulation buffer if something has been edited
        if (edited)
        {
            samples = 1;
            renderSeconds = 0.0f;
            memset(renderBuffer, 0.0f, xres * yres * sizeof(color));
            edited = false;
        }

        // Update flythrough camera

        double cursorX, cursorY;
        glfwGetCursorPos(window, &cursorX, &cursorY);

        const float activated = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS ? 1.0f : 0.0f;

        const float delta_time_sec = 60.0f / ImGui::GetIO().Framerate;

        if (activated) edited = true;

        float view[16];
        flythrough_camera_update(
            pos, look, up, view,
            delta_time_sec,
            1.0f * (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ? 2.0f : 1.0f) * activated,
            0.5f * activated,
            80.0f,
            cursorX - oldCursorX, cursorY - oldCursorY,
            glfwGetKey(window, GLFW_KEY_W), glfwGetKey(window, GLFW_KEY_A), glfwGetKey(window, GLFW_KEY_S), glfwGetKey(window, GLFW_KEY_D),
            glfwGetKey(window, GLFW_KEY_SPACE), glfwGetKey(window, GLFW_KEY_LEFT_CONTROL),
            0);

        cam.pos = vec3(pos[0], pos[1], pos[2]);

        cam.SetTransformFromCam(mat44(view[0], view[1], view[2], view[3],
            view[4], view[5], view[6], view[7],
            view[8], view[9], view[10], view[11],
            view[12], view[13], view[14], view[15]));


        ////////////////////////////////

        glfwPollEvents();
        glClearColor(clear_color.x, clear_color.y, clear_color.z, clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        ImGui::DockSpaceOverViewport(ImGui::GetMainViewport());

        if(render)
        {
            auto startRender = get_time();

            Render(renderBuffer, accelerator, materials, blueNoisePtr, ImGui::GetFrameCount(), samples, tiles, cam, settings);

            auto endRender = get_time();

            elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(endRender - startRender).count();
            renderSeconds += elapsed;

            glBindTexture(GL_TEXTURE_2D, render_view_texture);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, xres, yres, GL_RGB, GL_FLOAT, renderBuffer);
            glBindTexture(GL_TEXTURE_2D, 0);

        }
        
        glBindFramebuffer(GL_READ_FRAMEBUFFER, myFbo);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0); // default fbo  
        glBlitFramebuffer(0, 0, xres, yres,
            0, display_h, display_w, 0, GL_COLOR_BUFFER_BIT, GL_LINEAR); // or GL_NEAREST

        glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);

        // Info Window
        ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.1f, 0.1f, 0.1f, 1.0f));
        ImGui::Begin("Debug");
        {
            ImGui::Text("FPS : %0.3f", ImGui::GetIO().Framerate);
            ImGui::Text("Frame time : %0.3f ms", elapsed);
            ImGui::Text("Render time : %0.1f s", renderSeconds / 1000.0f);
            ImGui::Text("Samples : %d", samples);
            if (ImGui::Button("Render"))
            {
                if (render) render = false;
                else render = true;
            }
            ImGui::DragFloat("Gamma", &settings.gamma, 0.05f, 1.0f, 3.0f);

            ImGui::End();
        }
        ImGui::PopStyleColor();

        if (render)
        {
            samples++;
            oldCursorX = cursorX;
            oldCursorY = cursorY;
        }

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        ImGui::EndFrame();
        
        glfwSwapBuffers(window);
    }

    for(int i = 0; i < NUM_MATS; i++) delete materials[i];
    delete[] materials;

    ReleaseAccelerator(accelerator, allocators);
    ReleaseTiles(tiles);

    delete[] renderBuffer;

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}