#pragma once

#include <cstdlib> // std::size_t



#include "FluidContainer.h"
#include "SYCLFluidContainer.h"

constexpr std::size_t SIZE{ 300 };

class FluidSimulation : public ci::app::App {
public:
	FluidSimulation() : size_{ SIZE }, fluid_{ size_, 0.2f, 0.0f, 0.0000001f } {}
	void setup() override {
		texture_ = ci::gl::Texture2d::create(nullptr, GL_RGBA, size_, size_,
												ci::gl::Texture2d::Format()
												.dataType(GL_UNSIGNED_BYTE)
												.internalFormat(GL_RGBA));
	}
	void update() override {

		if (engine::InputHandler::KeyDown(engine::Key::SPACE)) {
			fluid_.Reset();
        }
        
		// Add density at mouse cursor location.
		fluid_.AddDensity(x, y, 200, 2);

        // Fade overall dye levels slowly over time.
        fluid.DecreaseDensity(0.99f);


		fluid_.Update();
	}
	void draw() override {
		ci::gl::clear();

		// Update GL texture with new calculation data
		fluid.with_data([&](sycl::cl_uchar4 const* data) {
			texture_->update(data, GL_RGBA, GL_UNSIGNED_BYTE, 0, size_, size_);
		});

		ci::Rectf screen(0, 0, getWindow()->getWidth(), getWindow()->getHeight());
		ci::gl::draw(texture_, screen);
	}
private:
	void mouseDrag(ci::app::MouseEvent event) override {
		float x{ event.getX() / double(size_) };
		float y{ event.getY() / double(size_) };

		float amount_x{ x - previous_x };
		float amount_y{ y - previous_y };

		// Add velocity vectors in direction of mouse travel.
		fluid_.AddVelocity(x, y, amount_x, amount_y);
	}
	SYCLFluidContainer fluid_;
	std::size_t size_{ 0 };
	double previous_x{ 0 };
	double previous_y{ 0 };
	ci::gl::Texture2dRef texture_;
};

CINDER_APP(FluidSimulation, ci::app::RendererGl(ci::app::RendererGl::Options{}))

//template <typename FluidContainerType>
//class FluidSimulation : public engine::Scene {
//public:
//
//    const int SCALE{ 2 };
//    FluidContainerType fluid{ 300, 0.2f, 0.0f, 0.0000001f }; // Size, Dt, Diffusion, Viscosity
//
//    engine::Texture texture;
//    V2_int previous_mouse;
//
//    void Enter() {
//        texture = engine::Renderer::CreateTexture({ fluid.GetSize(), fluid.GetSize() },
//                                                  engine::PixelFormat::ARGB8888,
//                                                  engine::TextureAccess::STREAMING);
//        previous_mouse = engine::InputHandler::GetMousePosition();
//    }
//
//    void Update() {
//        // Reset the flluid if space is pressed.
//        if (engine::InputHandler::KeyDown(engine::Key::SPACE)) {
//            fluid.Reset();
//        }
//
//        auto mouse{ engine::InputHandler::GetMousePosition() };
//        V2_float amount{ mouse - previous_mouse };
//
//        // Add velocity vectors in direction of mouse travel.
//        fluid.AddVelocity(mouse.x / SCALE, mouse.y / SCALE, amount.x, amount.y);
//        // Add density at mouse cursor location.
//        fluid.AddDensity(mouse.x / SCALE, mouse.y / SCALE, 200, 2);
//
//        // Fade overall dye levels slowly over time.
//        fluid.DecreaseDensity(0.99f);
//
//        // Update fluid.
//        fluid.Update();
//        previous_mouse = mouse;
//    }
//
//    void Render() {
//        int pitch{ 0 };
//        void* pixel_array{ nullptr };
//        texture.Lock(&pixel_array, &pitch);
//        auto pixels{ static_cast<std::uint32_t*>(pixel_array) };
//        auto blue{ 0 };
//        static const auto alpha{ (255 << 24) };
//        auto size{ fluid.GetSize() };
//        for (int j{ 0 }; j < size; ++j) {
//            auto row{ j * size };
//            for (int i{ 0 }; i < size; ++i) {
//                auto index{ i + row };
//                auto density{ fluid.GetDensity(index) };
//                blue = density > 255 ? 255 : engine::math::Round<std::uint8_t>(density);
//                pixels[index] = (blue << 16) + alpha;
//            }
//        }
//        texture.Unlock();
//        engine::Renderer::DrawTexture(texture, {}, { size * SCALE, size * SCALE });
//    }
//
//};