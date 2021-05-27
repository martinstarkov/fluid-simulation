#include "FluidSimulation.h"

int main(int c, char** v) {

    engine::Engine::Start<FluidSimulation<SYCLFluidContainer>>("Fluid Simulation", { 600, 600 }, 60);

	return 0;
}