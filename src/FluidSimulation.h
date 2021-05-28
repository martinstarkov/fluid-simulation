#pragma once

#include <vector> // std::vector
#include <algorithm> // std::fill
#include <cstdlib> // std::size_t
#include <cstdint> // std::uint32_t, etc

#include <protegon.h>

#include "FluidContainer.h"
#include "SYCLFluidContainer.h"

template <typename FluidContainerType>
class FluidSimulation : public engine::Engine {
public:

    const int SCALE{ 2 };
    FluidContainerType fluid{ 300, 0.2f, 0, 0.0000001f }; // Dt, Diffusion, Viscosity

    V2_float gravity; // Initial gravity

    float gravity_increment{ 1.0f }; // Increment by which gravity increases / decreases
    int radius{ 1 };

    engine::Texture texture;

    void Init() {
        texture = { Engine::GetDisplay().second, { fluid.size, fluid.size }, engine::PixelFormat::ARGB8888, engine::TextureAccess::STREAMING };
        previous_mouse = engine::InputHandler::GetMousePosition();
    }
    V2_int previous_mouse;
    void Update() {
        // Reset the screen.
        if (engine::InputHandler::KeyDown(engine::Key::SPACE)) {
            fluid.Reset();
        }
        // Reset gravity.
        if (engine::InputHandler::KeyDown(engine::Key::R)) {
            gravity = {};
        }
        // Increment gravity.
        if (engine::InputHandler::KeyDown(engine::Key::DOWN)) {
            gravity.y += gravity_increment;
        } else if (engine::InputHandler::KeyDown(engine::Key::UP)) {
            gravity.y -= gravity_increment;
        } else if (engine::InputHandler::KeyDown(engine::Key::LEFT)) {
            gravity.x -= gravity_increment;
        } else if (engine::InputHandler::KeyDown(engine::Key::RIGHT)) {
            gravity.x += gravity_increment;
        }
        // Add fluid.
        auto current_mouse{ engine::InputHandler::GetMousePosition() };

        auto amount{ current_mouse - previous_mouse };

        fluid.AddVelocity(current_mouse.x / SCALE, current_mouse.y / SCALE, amount.x, amount.y);

        previous_mouse = current_mouse;
        //if (engine::InputHandler::MousePressed(engine::Mouse::LEFT)) {
            // Add dye.
            //auto mouse_position{ engine::InputHandler::GetMousePosition() };
            fluid.AddDensity(current_mouse.x / SCALE, current_mouse.y / SCALE, 200, radius * SCALE);
            // Add gravity vector.
            //fluid.AddVelocity(mouse_position.x / SCALE, mouse_position.y / SCALE, gravity.x, gravity.y);
        //}

        // Fade overall dye levels slowly over time.
        fluid.DecreaseDensity(0.99);

        // Update fluid.
        fluid.Update();
        AllocationMetrics::PrintMemoryUsage();
    }

    void Render() {
        engine::Timer timer;
        timer.Start();
        bool density_graph{ false };
        if (engine::InputHandler::KeyDown(engine::Key::D)) {
            density_graph = !density_graph;
        }
        int pitch;
        void* pixel_array;
        texture.Lock(&pixel_array, &pitch);
        auto pixels{ static_cast<std::uint32_t*>(pixel_array) };
        auto blue{ 0 };
        auto alpha{ (255 << 24) };
        for (int j{ 0 }; j < fluid.size; ++j) {
            auto row{ j * fluid.size };
            for (int i{ 0 }; i < fluid.size; ++i) {
                auto index{ i + row };
                auto density{ fluid.density[index] };
                blue = density > 255 ? 255 : engine::math::Round<std::uint8_t>(density);
                /*if (density_graph) {
                    color.g = engine::math::Round<std::uint8_t>(density);
                    if (density < 255.0f * 2.0f && density > 255.0f) {
                        color.g -= 255;
                    }
                }*/
                pixels[index] = (blue << 16) + alpha;
            }
        }
        texture.Unlock();
        engine::Renderer::DrawTexture(texture, {}, { fluid.size * SCALE, fluid.size * SCALE });
        engine::PrintLine("Render time: ", timer.Elapsed<engine::milliseconds>().count());
    }

};