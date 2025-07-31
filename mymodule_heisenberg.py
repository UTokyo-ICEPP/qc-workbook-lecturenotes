import numpy as np
from qiskit import QuantumCircuit


def trotter_twopi_heisenberg(state_register, energy_norm, g, num_steps):
    """Return a function that implements a single Trotter step for the Heisenberg model.

    The Heisenberg model Hamiltonian is
    H = -J * sum_of_sigmas = hbar*ω * Θ

    The returned circuit implements a negative time evolution
    U = exp(-i H*(-τ)/hbar)
    where τ = 2π / ω, which leads to
    U = exp(i 2π Θ).

    Because we employ the Suzuki-Trotter decomposition, the actual circuit corresponds to
    U = [exp(i 2π/num_steps Θ)]^num_steps.

    Args:
        state_register (QuantumRegister): Register to perform the Suzuki-Trotter simulation.
        energy_norm (float): J/(hbar*ω).
        g (float): External field strength relative to the coupling constant J.
        num_steps (float): Number of steps to divide the time evolution of ωτ=2π.

    Returns:
        QuantumCircuit: A quantum circuit implementing the Trotter simulation of the Heisenberg
        model.
    """
    circuit = QuantumCircuit(state_register, name='ΔU')

    n_spins = state_register.size
    step_size = 2. * np.pi / num_steps

    # Implement the circuit corresponding to exp(i*step_size*Θ) below, where Θ is defined by
    # Θ = -J/(hbar*ω) * sum_of_sigmas = -energy_norm * sum_of_sigmas
    ##################
    ### EDIT BELOW ###
    ##################

    phase = -energy_norm * step_size

    # circuit.?
    for j in range(n_spins):
        # ZZ
        # exp(i dphi ZZ) -> phase +dphi if parity is even
        circuit.cx(j, (j + 1) % n_spins)
        # j + 1 is |0> if parity is even -> apply Rz(-2*dphi) = exp(dphi Z)
        circuit.rz(-2. * phase, (j + 1) % n_spins)
        circuit.cx(j, (j + 1) % n_spins)

        # XX
        circuit.h(j)
        circuit.h((j + 1) % n_spins)
        circuit.cx(j, (j + 1) % n_spins)
        circuit.rz(-2. * phase, (j + 1) % n_spins)
        circuit.cx(j, (j + 1) % n_spins)
        circuit.h(j)
        circuit.h((j + 1) % n_spins)

        # YY
        circuit.p(-np.pi / 2., j)
        circuit.p(-np.pi / 2., (j + 1) % n_spins)
        circuit.h(j)
        circuit.h((j + 1) % n_spins)
        circuit.cx(j, (j + 1) % n_spins)
        circuit.rz(-2. * phase, (j + 1) % n_spins)
        circuit.cx(j, (j + 1) % n_spins)
        circuit.h(j)
        circuit.h((j + 1) % n_spins)
        circuit.p(np.pi / 2., j)
        circuit.p(np.pi / 2., (j + 1) % n_spins)

        # gZ
        if g != 0.:
            # exp(i dphi gZ)
            circuit.rz(-2. * g * phase, j)

    ##################
    ### EDIT ABOVE ###
    ##################

    circuit = circuit.repeat(num_steps)
    circuit.name = 'U'

    return circuit
