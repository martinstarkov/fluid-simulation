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
    FluidContainerType fluid{ 300, 0.1f, 0.0001f, 0.000001f }; // Dt, Diffusion, Viscosity

    V2_float gravity; // Initial gravity

    float gravity_increment{ 1.0f }; // Increment by which gravity increases / decreases
    int radius{ 4 };

    engine::Texture texture;

    void Init() {
        texture = { Engine::GetDisplay().second, { fluid.size, fluid.size }, engine::PixelFormat::BGRA8888, engine::TextureAccess::STREAMING };
    }

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
        if (engine::InputHandler::MousePressed(engine::Mouse::LEFT)) {
            // Add dye.
            auto mouse_position{ engine::InputHandler::GetMousePosition() };
            fluid.AddDensity(mouse_position.x / SCALE, mouse_position.y / SCALE, 1000, radius);
            // Add gravity vector.
            fluid.AddVelocity(mouse_position.x / SCALE, mouse_position.y / SCALE, gravity.x, gravity.y);
        }

        // Fade overall dye levels slowly over time.
        fluid.DecreaseDensity();

        // Update fluid.
        fluid.Update();
    }

    void Render() {
        static bool density_graph{ false };
        if (engine::InputHandler::KeyDown(engine::Key::D)) {
            density_graph = !density_graph;
        }
        int pitch;
        void* pixels;
        texture.Lock(&pixels, &pitch);
        auto pixels_ = static_cast<std::uint32_t*>(pixels);
        for (int i{ 0 }; i < fluid.size; ++i) {
            for (int j{ 0 }; j < fluid.size; ++j) {
                V2_int position{ i * SCALE, j * SCALE };
                engine::Color color{ 0, 0, 0, 255 };

                auto index = fluid.IX(i, j, fluid.size);

                auto density{ fluid.density[index] };

                color.b = density > 255 ? 255 : engine::math::Round<std::uint8_t>(density);

                if (density_graph) {
                    color.g = engine::math::Round<std::uint8_t>(density);
                    if (density < 255.0f * 2.0f && density > 255.0f) {
                        color.g -= 255;
                    }
                }

                pixels_[index] = color.ToUint32();
            }
        }
        texture.Unlock();

        engine::Renderer::DrawTexture(texture, {}, { fluid.size * SCALE, fluid.size * SCALE });
    }

};