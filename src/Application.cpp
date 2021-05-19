#include <vector> // std::vector
#include <algorithm> // std::fill
#include <cstdlib> // std::size_t
#include <cstdint> // std::uint32_t, etc

#include "FluidContainer.h"

#include <CL/sycl.hpp>
#include <protegon.h>

// Kernel declarations.
class fluid_linear_solve;
class fluid_add_velocity;
class fluid_project1;
class fluid_project2;
class fluid_project3;
class fluid_project4;
class fluid_project5;
class fluid_advect;
class fluid_whole;
template <std::size_t SIZE>
class fluid_kernel;

auto exception_handler = [](sycl::exception_list exceptions) {
    for (std::exception_ptr const& e : exceptions) {
        try {
            std::rethrow_exception(e);
        } catch (sycl::exception const& e) {
            std::cout << "Caught asynchronous SYCL exception:\n"
                << e.what() << std::endl;
        }
    }
};

class SYCLFluidContainer {
public:

    using read_write_accessor
        = cl::sycl::accessor<float, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::global_buffer, cl::sycl::access::placeholder::false_t>;

    // Member variables.

    std::size_t size{ 0 };
    std::size_t velocity_iterations{ 4 };
    std::size_t density_iterations{ 4 };

    float dt{ 0.0f };
    float diffusion{ 0.0f };
    float viscosity{ 0.0f };
    float a_velocity{ 0.0f };
    float a_density{ 0.0f };
    float c_reciprocal_velocity{ 0.0f };
    float c_reciprocal_density{ 0.0f };
    float c_reciprocal_project{ 1.0f / 6.0f };
    float dt0{ 0.0f };

    std::vector<float> px;
    std::vector<float> py;
    std::vector<float> x;
    std::vector<float> y;
    std::vector<float> previous_density;
    std::vector<float> density;

    SYCLFluidContainer(std::size_t size, float dt, float diffusion, float viscosity) :
        size{ size }, dt{ dt }, diffusion{ diffusion }, viscosity{ viscosity } {
        auto s{ size * size };
        a_velocity = dt * viscosity * (size - 2) * (size - 2);
        c_reciprocal_velocity = 1.0f / (1.0f + 6.0f * a_velocity);
        a_density = dt * diffusion * (size - 2) * (size - 2);
        c_reciprocal_density = 1.0f / (1.0f + 6.0f * a_density);
        dt0 = dt * size;
        px.resize(s);
        py.resize(s);
        x.resize(s);
        y.resize(s);
        previous_density.resize(s);
        density.resize(s);
        cl::sycl::property_list props{ cl::sycl::property::buffer::use_host_ptr() };
        x_b = { x.data(), x.size(), props };
        y_b = { y.data(), y.size(), props };
        px_b = { px.data(), px.size(), props };
        py_b = { py.data(), py.size(), props };
        previous_density_b = { previous_density.data(), previous_density.size(), props };
        density_b = { density.data(), density.size(), props };
    }
    ~SYCLFluidContainer() = default;

    // Reset fluid to empty.
    void Reset() {
        std::fill(px.begin(), px.end(), 0.0f);
        std::fill(py.begin(), py.end(), 0.0f);
        std::fill(x.begin(), x.end(), 0.0f);
        std::fill(y.begin(), y.end(), 0.0f);
        std::fill(previous_density.begin(), previous_density.end(), 0.0f);
        std::fill(density.begin(), density.end(), 0.0f);
    }

    // Fade density over time.
    void DecreaseDensity(float fraction = 0.999) {
        for (auto& d : density) {
            d *= fraction;
        }
    }

    // Get clamped index based off of coordinates.
    static std::size_t IX(std::size_t x, std::size_t y, std::size_t N) {
        // Clamp coordinates.
        if (x < 0) { x = 0; }
        if (x > N - 1) { x = N - 1; }
        if (y < 0) { y = 0; }
        if (y > N - 1) { y = N - 1; }

        return (y * N) + x;
    }

    // Add density to the density field.
    void AddDensity(std::size_t x, std::size_t y, float amount, int radius = 0) {
        if (radius > 0) {
            for (int i{ -radius }; i <= radius; ++i) {
                for (int j{ -radius }; j <= radius; ++j) {
                    if (i * i + j * j <= radius * radius) {
                        auto index{ IX(x + i, y + j, size) };
                        this->density[index] += amount;
                    }
                }
            }
        } else {
            auto index{ IX(x, y, size) };
            this->density[index] += amount;
        }
    }

    // Add velocity to the velocity field.
    void AddVelocity(std::size_t x, std::size_t y, float px, float py) {
        auto index{ IX(x, y, size) };
        this->x[index] += px;
        this->y[index] += py;
    }
 
    // SYCL Code

    cl::sycl::queue queue{ cl::sycl::host_selector{}, exception_handler };

    cl::sycl::buffer<float, 1> x_b;
    cl::sycl::buffer<float, 1> y_b;
    cl::sycl::buffer<float, 1> px_b;
    cl::sycl::buffer<float, 1> py_b;
    cl::sycl::buffer<float, 1> previous_density_b;
    cl::sycl::buffer<float, 1> density_b;

    // Set boundaries to opposite of adjacent layer.
    static void SetBoundaryConditions(int b, std::vector<float>& x, std::size_t N) {
        for (std::size_t i{ 1 }; i < N - 1; ++i) {
            auto top{ IX(i, 1, N) };
            auto bottom{ IX(i, N - 2, N) };
            x[IX(i, 0, N)] = b == 2 ? -x[top] : x[top];
            x[IX(i, N - 1, N)] = b == 2 ? -x[bottom] : x[bottom];
        }

        for (std::size_t j{ 1 }; j < N - 1; ++j) {
            auto left{ IX(1, j, N) };
            auto right{ IX(N - 2, j, N) };
            x[IX(0, j, N)] = b == 1 ? -x[left] : x[left];
            x[IX(N - 1, j, N)] = b == 1 ? -x[right] : x[right];
        }

        // Set corner boundaries
        x[IX(0, 0, N)] = 0.33f * (x[IX(1, 0, N)] + x[IX(0, 1, N)] + x[IX(0, 0, N)]);
        x[IX(0, N - 1, N)] = 0.33f * (x[IX(1, N - 1, N)] + x[IX(0, N - 2, N)] + x[IX(0, N - 1, N)]);
        x[IX(N - 1, 0, N)] = 0.33f * (x[IX(N - 2, 0, N)] + x[IX(N - 1, 1, N)] + x[IX(N - 1, 0, N)]);
        x[IX(N - 1, N - 1, N)] = 0.33f * (x[IX(N - 2, N - 1, N)] + x[IX(N - 1, N - 2, N)] + x[IX(N - 1, N - 1, N)]);
    }

    // Set boundaries to opposite of adjacent layer.
    static void SetBoundaryConditions(int b, read_write_accessor x, std::size_t N, cl::sycl::handler& cgh) {
        cgh.single_task<fluid_kernel<0>>([=]() {
            for (std::size_t i{ 1 }; i < N - 1; ++i) {
                auto top{ IX(i, 1, N) };
                auto bottom{ IX(i, N - 2, N) };
                x[IX(i, 0, N)] = b == 2 ? -x[top] : x[top];
                x[IX(i, N - 1, N)] = b == 2 ? -x[bottom] : x[bottom];
            }

            for (std::size_t j{ 1 }; j < N - 1; ++j) {
                auto left{ IX(1, j, N) };
                auto right{ IX(N - 2, j, N) };
                x[IX(0, j, N)] = b == 1 ? -x[left] : x[left];
                x[IX(N - 1, j, N)] = b == 1 ? -x[right] : x[right];
            }

            // Set corner boundaries
            x[IX(0, 0, N)] = 0.33f * (x[IX(1, 0, N)] + x[IX(0, 1, N)] + x[IX(0, 0, N)]);
            x[IX(0, N - 1, N)] = 0.33f * (x[IX(1, N - 1, N)] + x[IX(0, N - 2, N)] + x[IX(0, N - 1, N)]);
            x[IX(N - 1, 0, N)] = 0.33f * (x[IX(N - 2, 0, N)] + x[IX(N - 1, 1, N)] + x[IX(N - 1, 0, N)]);
            x[IX(N - 1, N - 1, N)] = 0.33f * (x[IX(N - 2, N - 1, N)] + x[IX(N - 1, N - 2, N)] + x[IX(N - 1, N - 1, N)]);
        });
    }

    // Solve linear differential equation of density / velocity.
    static void LinearSolve(int b, std::vector<float>& x, std::vector<float>& x0, float a, float c_reciprocal, std::size_t N) {
        for (std::size_t j{ 1 }; j < N - 1; ++j) {
            for (std::size_t i{ 1 }; i < N - 1; ++i) {
                auto index{ IX(i, j, N) };
                x[index] = (x0[index] +
                            a * (
                                x[IX(i + 1, j, N)]
                                + x[IX(i - 1, j, N)]
                                + x[IX(i, j + 1, N)]
                                + x[IX(i, j - 1, N)]
                                + x[index]
                                + x[index]
                                )
                            ) * c_reciprocal;
            }
        }
    }

    // Solve linear differential equation of density / velocity.
    static void LinearSolve(int b, read_write_accessor x, read_write_accessor x0, float a, float c_reciprocal, std::size_t N, cl::sycl::handler& cgh) {
        cgh.parallel_for<fluid_linear_solve>(cl::sycl::range<2>(N - 2, N - 2), [=](cl::sycl::item<2> item) {
            auto i{ 1 + item.get_id(0) };
            auto j{ 1 + item.get_id(1) };
            auto index{ IX(i, j, N) };
            x[index] = (x0[index] +
                        a * (
                            x[IX(i + 1, j, N)]
                            + x[IX(i - 1, j, N)]
                            + x[IX(i, j + 1, N)]
                            + x[IX(i, j - 1, N)]
                            + x[index]
                            + x[index]
                            )
                        ) * c_reciprocal;
        });
    }

    static void Project(std::vector<float>& vx, std::vector<float>& vy, std::vector<float>& p, std::vector<float>& div, std::size_t iterations, std::size_t N) {
        for (std::size_t j{ 1 }; j < N - 1; ++j) {
            for (std::size_t i{ 1 }; i < N - 1; ++i) {
                auto index{ IX(i, j, N) };
                div[index] = -0.5f * (
                    vx[IX(i + 1, j, N)]
                    - vx[IX(i - 1, j, N)]
                    + vy[IX(i, j + 1, N)]
                    - vy[IX(i, j - 1, N)]
                    ) / N;
                p[index] = 0;
            }
        }
    }

    // Converse 'mass' of density / velocity fields.
    static void Project1(read_write_accessor vx, read_write_accessor vy, read_write_accessor p, read_write_accessor div, std::size_t N, cl::sycl::handler& cgh) {
        cgh.parallel_for<fluid_project1>(cl::sycl::range<2>(N - 2, N - 2), [=](cl::sycl::item<2> item) {
            auto i{ 1 + item.get_id(0) };
            auto j{ 1 + item.get_id(1) };
            auto index{ IX(i, j, N) };
            div[index] = -0.5f * (
                vx[IX(i + 1, j, N)]
                - vx[IX(i - 1, j, N)]
                + vy[IX(i, j + 1, N)]
                - vy[IX(i, j - 1, N)]
                ) / N;
            p[index] = 0;
        });
    }

    // Converse 'mass' of density / velocity fields.
    static void Project2(std::vector<float>&vx, std::vector<float>&vy, std::vector<float>&p, std::size_t N) {
        for (std::size_t j{ 1 }; j < N - 1; ++j) {
            for (std::size_t i{ 1 }; i < N - 1; ++i) {
                auto index{ IX(i, j, N) };
                vx[index] -= 0.5f * (p[IX(i + 1, j, N)] - p[IX(i - 1, j, N)]) * N;
                vy[index] -= 0.5f * (p[IX(i, j + 1, N)] - p[IX(i, j - 1, N)]) * N;
            }
        }
    }

    // Converse 'mass' of density / velocity fields.
    static void Project2(read_write_accessor vx, read_write_accessor vy, read_write_accessor p, std::size_t N, cl::sycl::handler& cgh) {
        cgh.parallel_for<fluid_project2>(cl::sycl::range<2>(N - 2, N - 2), [=](cl::sycl::item<2> item) {
            auto i{ 1 + item.get_id(0) };
            auto j{ 1 + item.get_id(1) };
            auto index{ IX(i, j, N) };
            vx[index] -= 0.5f * (p[IX(i + 1, j, N)] - p[IX(i - 1, j, N)]) * N;
            vy[index] -= 0.5f * (p[IX(i, j + 1, N)] - p[IX(i, j - 1, N)]) * N;
        });
    }

    // Move density / velocity within the field to the next step.
    static void Advect(int b, std::vector<float>& d, std::vector<float>& d0, std::vector<float>& u, std::vector<float>& v, float dt0, std::size_t N) {
        for (std::size_t i{ 1 }; i < N - 1; ++i) {
            for (std::size_t j{ 1 }; j < N - 1; ++j) {
                auto index{ IX(i, j, N) };

                float x{ i - dt0 * u[index] };
                float y{ j - dt0 * v[index] };
                x = Clamp(x, 0.5f, N + 0.5f);
                auto i0 = (int)x;
                auto i1 = i0 + 1;
                y = Clamp(y, 0.5f, N + 0.5f);
                auto j0 = (int)y;
                auto j1 = j0 + 1;
                float s1{ x - i0 };
                float s0{ 1 - s1 };
                float t1{ y - j0 };
                float t0{ 1 - t1 };
                d[index] = s0 * (t0 * d0[IX(i0, j0, N)] + t1 * d0[IX(i0, j1, N)]) +
                    s1 * (t0 * d0[IX(i1, j0, N)] + t1 * d0[IX(i1, j1, N)]);
            }
        }
    }

    // Move density / velocity within the field to the next step.
    static void Advect(int b, read_write_accessor d, read_write_accessor d0, read_write_accessor u, read_write_accessor v, float dt0, std::size_t N, cl::sycl::handler& cgh) {
        cgh.parallel_for<fluid_advect>(cl::sycl::range<2>(N - 2, N - 2), [=](cl::sycl::item<2> item) {
            auto i{ 1 + item.get_id(0) };
            auto j{ 1 + item.get_id(1) };
            auto index{ IX(i, j, N) };

            float x{ i - dt0 * u[index] };
            float y{ j - dt0 * v[index] };
            x = Clamp(x, 0.5f, N + 0.5f);
            auto i0 = (int)x;
            auto i1 = i0 + 1;
            y = Clamp(y, 0.5f, N + 0.5f);
            auto j0 = (int)y;
            auto j1 = j0 + 1;
            float s1{ x - i0 };
            float s0{ 1 - s1 };
            float t1{ y - j0 };
            float t0{ 1 - t1 };
            d[index] = s0 * (t0 * d0[IX(i0, j0, N)] + t1 * d0[IX(i0, j1, N)]) +
                s1 * (t0 * d0[IX(i1, j0, N)] + t1 * d0[IX(i1, j1, N)]);
        });
    }

    void Update() {
        for (std::size_t iteration{ 0 }; iteration < velocity_iterations; ++iteration) {
            Submit(queue, [&](cl::sycl::handler& cgh, auto x_a, auto px_a, auto y_a, auto py_a) {
                LinearSolve(1, px_a, x_a, a_velocity, c_reciprocal_velocity, size, cgh);
                LinearSolve(2, py_a, y_a, a_velocity, c_reciprocal_velocity, size, cgh);
            }, x_b, px_b, y_b, py_b);
            Submit(queue, [&](cl::sycl::handler& cgh, auto x_a, auto px_a, auto y_a, auto py_a) {
                SetBoundaryConditions(1, px_a, size, cgh);
                SetBoundaryConditions(2, py_a, size, cgh);
            }, x_b, px_b, y_b, py_b);
        }

        Submit(queue, [&](cl::sycl::handler& cgh, auto x_a, auto px_a, auto y_a, auto py_a) {
            Project1(px_a, py_a, x_a, y_a, size, cgh);
        }, x_b, px_b, y_b, py_b);
        Submit(queue, [&](cl::sycl::handler& cgh, auto x_a, auto px_a, auto y_a, auto py_a) {
            SetBoundaryConditions(0, y_a, size, cgh);
            SetBoundaryConditions(0, x_a, size, cgh);
        }, x_b, px_b, y_b, py_b);

        for (std::size_t iteration{ 0 }; iteration < velocity_iterations; ++iteration) {
            Submit(queue, [&](cl::sycl::handler& cgh, auto x_a, auto y_a) {
                LinearSolve(0, x_a, y_a, 1.0f, c_reciprocal_project, size, cgh);
            }, x_b, y_b);
            Submit(queue, [&](cl::sycl::handler& cgh, auto x_a) {
                SetBoundaryConditions(0, x_a, size, cgh);
            }, x_b);
        }

        Submit(queue, [&](cl::sycl::handler& cgh, auto x_a, auto px_a, auto py_a) {
            Project2(px_a, py_a, x_a, size, cgh);
        }, x_b, px_b, py_b);

        Submit(queue, [&](cl::sycl::handler& cgh, auto px_a, auto py_a) {
            SetBoundaryConditions(1, px_a, size, cgh);
            SetBoundaryConditions(2, py_a, size, cgh);
        }, px_b, py_b);

        Submit(queue, [&](cl::sycl::handler& cgh, auto x_a, auto px_a, auto y_a, auto py_a) {
            Advect(1, x_a, px_a, px_a, py_a, dt0, size, cgh);
            Advect(2, y_a, py_a, px_a, py_a, dt0, size, cgh);
        }, x_b, px_b, y_b, py_b);
        Submit(queue, [&](cl::sycl::handler& cgh, auto x_a, auto y_a) {
            SetBoundaryConditions(1, x_a, size, cgh);
            SetBoundaryConditions(2, y_a, size, cgh);
        }, x_b, y_b);

        Submit(queue, [&](cl::sycl::handler& cgh, auto x_a, auto px_a, auto y_a, auto py_a) {
            Project1(x_a, y_a, px_a, py_a, size, cgh);
        }, x_b, px_b, y_b, py_b);
        Submit(queue, [&](cl::sycl::handler& cgh, auto x_a, auto px_a, auto y_a, auto py_a) {
            SetBoundaryConditions(0, py_a, size, cgh);
            SetBoundaryConditions(0, px_a, size, cgh);
        }, x_b, px_b, y_b, py_b);

        for (std::size_t iteration{ 0 }; iteration < velocity_iterations; ++iteration) {
            Submit(queue, [&](cl::sycl::handler& cgh, auto px_a, auto py_a) {
                LinearSolve(0, px_a, py_a, 1.0f, c_reciprocal_project, size, cgh);
            }, px_b, py_b);
            Submit(queue, [&](cl::sycl::handler& cgh, auto px_a) {
                SetBoundaryConditions(0, px_a, size, cgh);
            }, px_b);
        }

        Submit(queue, [&](cl::sycl::handler& cgh, auto x_a, auto y_a, auto px_a) {
            Project2(x_a, y_a, px_a, size, cgh);
        }, x_b, y_b, px_b);

        Submit(queue, [&](cl::sycl::handler& cgh, auto x_a, auto y_a) {
            SetBoundaryConditions(1, x_a, size, cgh);
            SetBoundaryConditions(2, y_a, size, cgh);
        }, x_b, y_b);

        for (std::size_t iteration{ 0 }; iteration < density_iterations; ++iteration) {
            Submit(queue, [&](cl::sycl::handler& cgh, auto density_a, auto previous_density_a) {
                LinearSolve(0, previous_density_a, density_a, a_density, c_reciprocal_density, size, cgh);
            }, density_b, previous_density_b);
            Submit(queue, [&](cl::sycl::handler& cgh, auto previous_density_a) {
                SetBoundaryConditions(0, previous_density_a, size, cgh);
            }, previous_density_b);
        }

        Submit(queue, [&](cl::sycl::handler& cgh, auto density_a, auto previous_density_a, auto x_a, auto y_a) {
            Advect(0, density_a, previous_density_a, x_a, y_a, dt0, size, cgh);
        }, density_b, previous_density_b, x_b, y_b);

        Submit(queue, [&](cl::sycl::handler& cgh, auto density_a) {
            SetBoundaryConditions(0, density_a, size, cgh);
        }, density_b);


    }

    template <typename T>
    static read_write_accessor CreateAccessor(cl::sycl::handler& cgh, T& buffer) {
        return buffer.template get_access<cl::sycl::access::mode::read_write>(cgh);
    }

    template <typename T, typename ...Ts>
    static void Submit(cl::sycl::queue& queue, T lambda, Ts&... buffers) {
        queue.submit([&](cl::sycl::handler& cgh){
            lambda(cgh, CreateAccessor(cgh, buffers)...);
        });
        queue.wait();
    }

private:
    // Clamp value to a range.
    template <typename T>
    static T Clamp(T value, T low, T high) {
        return value >= high ? high : value <= low ? low : value;
    }
};

class FluidSimulation : public engine::Engine {
public:

    const int SCALE{ 2 };
    SYCLFluidContainer fluid{ 300, 0.1f, 0.0001f, 0.000001f }; // Dt, Diffusion, Viscosity

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

int main(int c, char** v) {

    engine::Engine::Start<FluidSimulation>("Fluid Simulation", { 600, 600 }, 60);

	return 0;
}