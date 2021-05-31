#pragma once

#include <cstdint> // std::uint32_t, etc

#include <protegon.h>

#include "FluidContainer.h"
#include "SYCLFluidContainer.h"

template <typename FluidContainerType>
class FluidSimulation : public engine::Scene {
public:

    const int SCALE{ 2 };
    FluidContainerType fluid{ 300, 0.2f, 0.0f, 0.0000001f }; // Size, Dt, Diffusion, Viscosity

    engine::Texture texture;
    V2_int previous_mouse;

    void Enter() {
        texture = engine::Renderer::CreateTexture({ fluid.GetSize(), fluid.GetSize() },
                                                  engine::PixelFormat::ARGB8888,
                                                  engine::TextureAccess::STREAMING);
        previous_mouse = engine::InputHandler::GetMousePosition();
    }

    void Update() {
        // Reset the flluid if space is pressed.
        if (engine::InputHandler::KeyDown(engine::Key::SPACE)) {
            fluid.Reset();
        }

        auto mouse{ engine::InputHandler::GetMousePosition() };
        V2_float amount{ mouse - previous_mouse };

        // Add velocity vectors in direction of mouse travel.
        fluid.AddVelocity(mouse.x / SCALE, mouse.y / SCALE, amount.x, amount.y);
        // Add density at mouse cursor location.
        fluid.AddDensity(mouse.x / SCALE, mouse.y / SCALE, 200, 2);

        // Fade overall dye levels slowly over time.
        fluid.DecreaseDensity(0.99f);

        // Update fluid.
        fluid.Update();
        previous_mouse = mouse;
    }

    void Render() {
        int pitch{ 0 };
        void* pixel_array{ nullptr };
        texture.Lock(&pixel_array, &pitch);
        auto pixels{ static_cast<std::uint32_t*>(pixel_array) };
        auto blue{ 0 };
        static const auto alpha{ (255 << 24) };
        auto size{ fluid.GetSize() };
        for (int j{ 0 }; j < size; ++j) {
            auto row{ j * size };
            for (int i{ 0 }; i < size; ++i) {
                auto index{ i + row };
                auto density{ fluid.GetDensity(index) };
                blue = density > 255 ? 255 : engine::math::Round<std::uint8_t>(density);
                pixels[index] = (blue << 16) + alpha;
            }
        }
        texture.Unlock();
        engine::Renderer::DrawTexture(texture, {}, { size * SCALE, size * SCALE });
    }

};