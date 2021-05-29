# importing numpy
import numpy as np
import itertools

# importing qiskit
from qiskit import Aer, QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit import Parameter, ParameterVector

from qiskit.utils import QuantumInstance
from qiskit_machine_learning.neural_networks import CircuitQNN


class InductiveGroversQNN(object):
    def __init__(
        self, input_size, hidden_layer_params=[], oracularizer_type="pairwise-full"
    ):

        self.input_size = input_size
        self.hidden_layer_params = hidden_layer_params
        self.oracularizer_type = oracularizer_type

        def _init_neuron_weights(size, id):
            layer = QuantumRegister(size)
            neuron_qc = QuantumCircuit(layer)

            param_vec = ParameterVector("$I_{%s}$" % id, 3)
            neuron_qc.u(*param_vec, layer)

            return neuron_qc.to_gate(label="Neural Initializer ")

        def _neuron_activation(size, id):
            layer = QuantumRegister(size)
            neuron_qc = QuantumCircuit(layer)

            param_vec = ParameterVector("$A_{%s}$" % id, 3)
            neuron_qc.u(*param_vec, layer)

            return neuron_qc.to_gate(label="Neural Activation ")

        def _neural_entangler(size_1, size_2, id, name, type="pairwise-full"):
            hidden_1 = QuantumRegister(size_1)
            hidden_2 = QuantumRegister(size_2)
            entangler_qc = QuantumCircuit(hidden_2, hidden_1)

            if type == "pairwise-full":
                for neuron_1, neuron_2 in itertools.product(
                    range(size_1), range(size_2)
                ):
                    param_vec = ParameterVector(
                        "$E^{%s}_{%s, %s}$" % (id, neuron_1, neuron_2), 8 * 3
                    )
                    entangler_qc.u(
                        param_vec[0], param_vec[1], param_vec[2], hidden_1[neuron_1]
                    )
                    entangler_qc.u(*param_vec[3:6], hidden_2[neuron_2])
                    entangler_qc.cx(hidden_2[neuron_2], hidden_1[neuron_1])
                    entangler_qc.u(*param_vec[6:9], hidden_1[neuron_1])
                    entangler_qc.u(*param_vec[9:12], hidden_2[neuron_2])
                    entangler_qc.cx(hidden_1[neuron_1], hidden_2[neuron_2])
                    entangler_qc.u(*param_vec[12:15], hidden_1[neuron_1])
                    entangler_qc.u(*param_vec[15:18], hidden_2[neuron_2])
                    entangler_qc.cx(hidden_2[neuron_2], hidden_1[neuron_1])
                    entangler_qc.u(*param_vec[18:21], hidden_1[neuron_1])
                    entangler_qc.u(*param_vec[21:24], hidden_2[neuron_2])
            elif type == "pairwise-linear":
                for neuron_1, neuron_2 in itertools.product(
                    range(size_1), range(size_2)
                ):
                    param_vec = ParameterVector(
                        "$E^{%s}_{%s, %s}$" % (id, neuron_1, neuron_2), 2 * 3
                    )
                    entangler_qc.u(
                        param_vec[0], param_vec[1], param_vec[2], hidden_1[neuron_1]
                    )
                    entangler_qc.u(*param_vec[3:6], hidden_2[neuron_2])
                    entangler_qc.cx(hidden_2[neuron_2], hidden_1[neuron_1])

            return entangler_qc.to_gate(label=name + " ")

        def _diffuser(size):
            output = QuantumRegister(size)
            diffuser_qc = QuantumCircuit(output)

            diffuser_qc.h(output)
            diffuser_qc.x(output)

            diffuser_qc.h(output[0])
            diffuser_qc.mct(output[1:], output[0])
            diffuser_qc.h(output[0])

            diffuser_qc.x(output)
            diffuser_qc.h(output)

            return diffuser_qc.to_gate(label="Grover Diffuser ")

        num_hidden_layers = len(hidden_layer_params)

        hidden_layers = [
            QuantumRegister(size, name="hidden%d" % (num_hidden_layers - l - 1))
            for l, size in enumerate([l["size"] for l in hidden_layer_params])
        ]
        output = QuantumRegister(input_size, name="output")
        input = QuantumRegister(input_size, name="input")
        oracle = QuantumRegister(1, name="oracle")
        measure = ClassicalRegister(input_size, name="measure")

        self.circuit = QuantumCircuit(*hidden_layers, output, input, oracle, measure)

        for layer in hidden_layers:
            self.circuit.h(layer)
        self.circuit.h(output)
        self.circuit.x(oracle)
        self.circuit.h(oracle)

        self.circuit.barrier()

        if num_hidden_layers:

            neuron_params = [
                (_init_neuron_weights(len(layer), layer.name[6:]), layer)
                for layer in hidden_layers
            ]
            for param in neuron_params:
                self.circuit.append(param[0], param[1])
            self.circuit.barrier()

            neuron_activations = [
                (_neuron_activation(len(layer), layer.name[6:]), layer)
                for layer in hidden_layers
            ]
            neural_entanglers = [
                (
                    _neural_entangler(
                        len(hidden_layers[i]),
                        len(hidden_layers[i + 1]),
                        i,
                        "Neural Entangler",
                        hidden_layer_params[i]["type"],
                    ),
                    hidden_layers[i][:] + hidden_layers[i + 1][:],
                )
                for i in range(num_hidden_layers - 1)
            ]
            for activation, entangler in zip(neuron_activations, neural_entanglers):
                self.circuit.append(activation[0], activation[1])
                self.circuit.barrier()
                self.circuit.append(entangler[0], entangler[1])
                self.circuit.barrier()

            self.circuit.append(neuron_activations[-1][0], neuron_activations[-1][1])
            self.circuit.barrier()
            oracle_generator = (
                _neural_entangler(
                    len(hidden_layers[-1]),
                    input_size,
                    len(hidden_layers),
                    "Oracle Generator",
                    hidden_layer_params[-1]["type"],
                ),
                hidden_layers[-1][:] + output[:],
            )
            self.circuit.append(oracle_generator[0], oracle_generator[1])
            self.circuit.barrier()

        for _ in range(int(np.round(np.sqrt(input_size)))):
            oracularizer = (
                _neural_entangler(
                    input_size,
                    input_size,
                    len(hidden_layers) + 1,
                    "Oracularizer",
                    oracularizer_type,
                ),
                output[:] + input[:],
            )
            self.circuit.append(oracularizer[0], oracularizer[1])

            self.circuit.barrier()

            self.circuit.x(input)
            self.circuit.mct(input, oracle)
            self.circuit.x(input)

            self.circuit.barrier()

            oracularizer_dg = oracularizer[0].inverse()
            oracularizer_dg.name = "Oracularizer$^\dagger$ "

            self.circuit.append(oracularizer_dg, oracularizer[1])
            self.circuit.barrier()

            self.circuit.append(_diffuser(input_size), output[:])
            self.circuit.barrier()

        if num_hidden_layers:

            oracle_generator_dg = oracle_generator[0].inverse()
            oracle_generator_dg.name = "Oracle Generator$^\dagger$ "

            final_neuron_activation_dg = neuron_activations[-1][0].inverse()
            final_neuron_activation_dg.name = "Neural Activation$^\dagger$ "

            self.circuit.append(oracle_generator_dg, oracle_generator[1])
            self.circuit.barrier()
            self.circuit.append(final_neuron_activation_dg, neuron_activations[-1][1])
            self.circuit.barrier()

            for activation, entangler in zip(neuron_activations, neural_entanglers):
                entangler_dg = entangler[0].inverse()
                entangler_dg.name = "Neural Entangler$^\dagger$ "

                activation_dg = activation[0].inverse()
                activation_dg.name = "Neural Activation$^\dagger$ "

                self.circuit.append(entangler_dg, entangler[1])
                self.circuit.barrier()
                self.circuit.append(activation_dg, activation[1])
                self.circuit.barrier()

            for param in neuron_params:
                param_dg = param[0].inverse()
                param_dg.name = "Neuron Initializer$^\dagger$ "
                self.circuit.append(param_dg, param[1])
            self.circuit.barrier()

        self.circuit.h(oracle)
        self.circuit.x(oracle)

        self.circuit.barrier()

        self.circuit.measure(output, measure)

    def toCircuitQNN(self):
        qi_qasm = QuantumInstance(Aer.get_backend("qasm_simulator"), shots=10)
        return CircuitQNN(
            self.circuit,
            [],
            self.circuit.parameters,
            sparse=True,
            quantum_instance=qi_qasm,
        )
