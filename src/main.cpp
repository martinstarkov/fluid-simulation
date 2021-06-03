#include <cstdlib> // std::size_t

#include <cinder/app/App.h>
#include <cinder/app/RendererGl.h>
#include <cinder/gl/gl.h>

#include "fluid.h"
#include "sycl_fluid.h"

constexpr std::size_t SIZE{ 300 };

class FluidSimulation : public ci::app::App {
public:
	FluidSimulation() : size_{ SIZE }, fluid_{ size_, 0.2f, 0.0f, 0.0000001f } {
		image.resize(size_* size_, { 0, 0, 0, 255 });
	}
	void setup() override {
		texture_ = ci::gl::Texture2d::create(nullptr, GL_RGBA, size_, size_,
											 ci::gl::Texture2d::Format()
											 .dataType(GL_UNSIGNED_BYTE)
											 .internalFormat(GL_RGBA));
	}
	void update() override {

		/*if (engine::InputHandler::KeyDown(engine::Key::SPACE)) {
			fluid_.Reset();
		}*/

		// Add density at mouse cursor location.
		//fluid_.AddDensity(x, y, 200, 2);

		// Fade overall dye levels slowly over time.
		fluid_.DecreaseDensity(0.99f);


		fluid_.Update();

		auto size{ fluid_.GetSize() };
		for (std::size_t j{ 0 }; j < size; ++j) {
			auto row{ j * size };
			for (std::size_t i{ 0 }; i < size; ++i) {
				auto index{ i + row };
				auto density{ fluid_.GetDensity(index) };
				auto blue{ density > 255 ? 255 : static_cast<std::uint8_t>(std::round(density)) };
				std::get<0>(image[index]) = blue;
			}
		}
	}
	void draw() override {
		ci::gl::clear();

		texture_->update(image.data(), GL_RGBA, GL_UNSIGNED_BYTE, 0, size_, size_);

		ci::Rectf screen{ 0, 0, getWindow()->getWidth(), getWindow()->getHeight() };
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
	std::vector<std::tuple<std::uint8_t, std::uint8_t, std::uint8_t, std::uint8_t>> image;
	SYCLFluidContainer fluid_;
	std::size_t size_{ 0 };
	double previous_x{ 0 };
	double previous_y{ 0 };
	ci::gl::Texture2dRef texture_;
};

CINDER_APP(FluidSimulation, ci::app::RendererGl(ci::app::RendererGl::Options{}))