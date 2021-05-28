#include "FluidSimulation.h"

int main(int c, char** v) {

    engine::Engine::Start<FluidSimulation<SYCLFluidContainer>>("fluid", "Fluid Simulation", { 600, 600 }, 60);

	return 0;
}