from typing import Any, Sequence, Optional
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.opflow import PauliSumOp
from qiskit.providers import Backend
from qiskit.algorithms.optimizers import COBYLA

def energy_expval(
    circuit: QuantumCircuit,
    hamiltonian_op: PauliSumOp,
    backend: Backend,
    shots: int = 2000
) -> float:
    """Compute the expectation value of the given Hamiltonian for the state defined by the given circuit.
    
    Args:
        circuit: The circuit with whose final state we evaluate the expectation value.
        hamiltonian_op: The Hamiltonian expressed as a PauliSumOp.
        backend: Backend to run the circuit and evaluate the expectation values.
        shots: Number of shots to use for expectation value calculations.
        
    Returns:
        The energy expectation value.
    """
    
    energy = 0.
    
    ##################
    ### EDIT BELOW ###
    ##################
    
    # ノートブックの energy_expval の中身を貼り付けてください
    
    circuit_iz = circuit.copy()
    circuit_iz.measure_all()

    circuit_iz = transpile(circuit_iz, backend=backend)
    counts_iz = backend.run(circuit_iz, shots=shots).result().get_counts()

    energy = 0.

    for pauli_term in hamiltonian_op:
        # paulisがXXYYやIZIZなどのパウリ積を表す文字列
        paulis = pauli_term.primitive.paulis.to_labels()[0]
        # coefficientがこの項の係数
        coefficient = pauli_term.coeffs[0].real

        if 'X' in paulis or 'Y' in paulis:
            circ = circuit.copy()

            for iq, label in enumerate(paulis[::-1]):
                if label == 'X':
                    circ.h(iq)
                elif label == 'Y':
                    circ.sdg(iq)
                    circ.h(iq)

            circ.measure_all()

            circ = transpile(circ, backend=backend)
            counts = backend.run(circ, shots=shots).result().get_counts()

        else:
            counts = counts_iz

        expval = 0.
        for key, value in counts.items():
            sign = 1.
            for bit, label in zip(key, paulis):
                if bit == '1' and label != 'I':
                    sign *= -1.

            expval += sign * value

        expval /= shots

        energy += coefficient * expval

    ##################
    ### EDIT ABOVE ###
    ##################
    
    return energy


def main(
    backend: Backend,
    user_messenger: Any,
    ansatz: QuantumCircuit,
    hamiltonian_op: PauliSumOp,
    initial: Optional[Sequence] = None,
    shots: int = 2000,
    maxiter: int = 100,
    tol: Optional[float] = None,
    report_every: int = 10
) -> dict:
    """The main program function.
    
    The function internally defines an objective function, which is in turn passed to the COBYLA optimizer
    for energy minimization. The objective function takes the value of the ansatz parameters as an argument
    and returns the energy expectation value of the Hamiltonian with respect to the ansatz circuit with the
    given parameter values.
    
    Args:
        backend: Backend to run the circuit and evaluate the expectation values.
        user_messenger: An object that publishes interim results.
        ansatz: Parametrized quantum circuit with respect to which the energy expectation values are computed.
        hamiltonian_op: The Hamiltonian expressed as a PauliSumOp.
        initial: An array of parameter initial values. If None, all parameters are initialized to value 0.01.
        shots: Number of shots to use for expectation value calculations.
        maxiter: Maximum number of COBYLA iterations.
        tol: COBYLA tolerance parameter.
        report_every: The number of objective function calls to make before publishing an interim result.
        
    Returns:
        A dictionary containing the final parameter values, minimized energy, and number of function calls used
        for minimization.
    """
    num_calls = 0

    # COBYLAで最小化する量に対応する関数
    def objective_function(param_values):
        nonlocal num_calls
        
        energy = 0.
        
        ##################
        ### EDIT BELOW ###
        ##################
        
        # ノートブックのobjective_functionの中身を貼り付けてください
        
        circuit = ansatz.bind_parameters(param_values)
        energy = energy_expval(circuit, hamiltonian_op, backend, shots)
    
        ##################
        ### EDIT ABOVE ###
        ##################
        
        num_calls += 1
        if report_every > 0 and num_calls % report_every == 0:
            user_messenger.publish({'params': param_values, 'energy': energy, 'num_calls': num_calls})
        
        return energy

    # COBYLAのインスタンスを作成
    optimizer = COBYLA(maxiter=maxiter, tol=tol)

    # 初期値が与えられていなければ、全て0.01とする
    if initial is None:
        initial = np.full(ansatz.num_parameters, 0.01)

    # 最小化
    ret = optimizer.minimize(objective_function, initial)

    # 得られたパラメータ値で最後に一回objective_functionを呼び、最小化されたエネルギーの値を得る
    min_energy = objective_function(ret.x)
    
    return {'params': ret.x, 'energy': min_energy, 'num_calls': num_calls}
