#include <cstdlib> // std::size_t
#include <vector> // std::vector
#include <algorithm> // std::fill

class FluidContainer {
public:
    // Size of square container
    std::size_t size{ 0 };
    // Delta timestep
    float dt{ 0.0f };
    // Rate of diffusion.
    float diffusion{ 0.0f };
    // Viscosity of fluid.
    float viscosity{ 0.0f };

    // Previous velocity components.
    std::vector<float> px;
    std::vector<float> py;
    // Current velocity components.
    std::vector<float> x;
    std::vector<float> y;
    // Previous density at a point.
    std::vector<float> previous_density;
    // Current density at a point.
    std::vector<float> density;

    FluidContainer(std::size_t size, float dt, float diffusion, float viscosity) :
        size{ size }, dt{ dt }, diffusion{ diffusion }, viscosity{ viscosity } {
        auto s{ size * size };
        px.resize(s, 0);
        py.resize(s, 0);
        x.resize(s, 0);
        y.resize(s, 0);
        previous_density.resize(s, 0);
        density.resize(s, 0);
    }

    ~FluidContainer() = default;

    // Reset fluid to empty.
    void Reset() {
        std::fill(px.begin(), px.end(), 0.0f);
        std::fill(py.begin(), py.end(), 0.0f);
        std::fill(x.begin(), x.end(), 0.0f);
        std::fill(y.begin(), y.end(), 0.0f);
        std::fill(previous_density.begin(), previous_density.end(), 0.0f);
        std::fill(density.begin(), density.end(), 0.0f);
    }

    // Fade density over time by a fraction.
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
        // Add density in a circle around the poin5.
        if (radius > 0) {
            for (int i{ -radius }; i <= radius; ++i) {
                for (int j{ -radius }; j <= radius; ++j) {
                    if (i * i + j * j <= radius * radius) {
                        auto index{ IX(x + i, y + j, size) };
                        this->density[index] += amount;
                    }
                }
            }
        // Add density at point.
        } else {
            auto index{ IX(x, y, size) };
            this->density[index] += amount;
        }
    }

    // Add velocity to the velocity field.
    void AddVelocity(std::size_t x, std::size_t y, float amount_x, float amount_y) {
        auto index{ IX(x, y, size) };
        this->x[index] += amount_x;
        this->y[index] += amount_y;
    }

    // Fluid specific operations.

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

    // Solve linear differential equation of density / velocity.
    static void LinearSolve(int b, std::vector<float>& x, std::vector<float>& x0, float a, float c, std::size_t iterations, std::size_t N) {
        float c_reciprocal{ 1.0f / c };
        for (std::size_t iteration{ 0 }; iteration < iterations; ++iteration) {
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
            SetBoundaryConditions(b, x, N);
        }
    }

    // Diffuse density / velocity outward at each step.
    static void Diffuse(int b, std::vector<float>& x, std::vector<float>& x0, float diffusion, float dt, std::size_t iterations, std::size_t N) {
        float a{ dt * diffusion * (N - 2) * (N - 2) };
        LinearSolve(b, x, x0, a, 1 + 6 * a, iterations, N);
    }

    // Converse 'mass' of density / velocity fields.
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

        SetBoundaryConditions(0, div, N);
        SetBoundaryConditions(0, p, N);

        LinearSolve(0, p, div, 1, 6, iterations, N);

        for (std::size_t j{ 1 }; j < N - 1; ++j) {
            for (std::size_t i{ 1 }; i < N - 1; ++i) {
                auto index{ IX(i, j, N) };
                vx[index] -= 0.5f * (p[IX(i + 1, j, N)] - p[IX(i - 1, j, N)]) * N;
                vy[index] -= 0.5f * (p[IX(i, j + 1, N)] - p[IX(i, j - 1, N)]) * N;
            }
        }

        SetBoundaryConditions(1, vx, N);
        SetBoundaryConditions(2, vy, N);
    }

    // Move density / velocity within the field to the next step.
    static void Advect(int b, std::vector<float>& d, std::vector<float>& d0, std::vector<float>& u, std::vector<float>& v, float dt, std::size_t N) {
        float dt0{ dt * N };
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
        SetBoundaryConditions(b, d, N);
    }

    // Update the fluid.
    void Update() {
        Diffuse(1, px, x, viscosity, dt, 4, size);
        Diffuse(2, py, y, viscosity, dt, 4, size);
        Project(px, py, x, y, 4, size);
        Advect(1, x, px, px, py, dt, size);
        Advect(2, y, py, px, py, dt, size);
        Project(x, y, px, py, 4, size);
        Diffuse(0, previous_density, density, diffusion, dt, 4, size);
        Advect(0, density, previous_density, x, y, dt, size);
    }
private:
    // Clamp value to a range.
    template <typename T>
    inline T Clamp(T value, T low, T high) {
        return value >= high ? high : value <= low ? low : value;
    }
};