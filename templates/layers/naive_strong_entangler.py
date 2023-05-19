"""
Contains the StronglyEntanglingLayers template.
"""
import pennylane as qml
from pennylane.operation import Operation, AnyWires
import numpy as np

class StronglyEntanglingLayers(Operation):
    """Layers consisting of single qubit rotations and entanglers
    
    `L` layers
    `M` wires

    Args:
        weights (tensor_like): weight tensor of shape ``(L, M, 3)``
        wires (Iterable): wires that the template acts on
        ranges (Sequence[int]): sequence determining the range hyperparameter for each subsequent layer; if ``None``
                                using :math:`r=l \mod M` for the :math:`l` th layer and :math:`M` wires.
        imprimitive (pennylane.ops.Operation): two-qubit gate to use, defaults to :class:`~pennylane.ops.CNOT`
        p (tensor_like?): M

    """
    def __init__(self, weights, wires, ranges=None, imprimitive=None, p=None, do_queue=True, id=None):
        # weights: shape
        shape = qml.math.shape(weights)[-3:]

        if shape[1] != len(wires):
            raise ValueError(
                f"Weights tensor must have second dimension of length {len(wires)}; got {shape[1]}"
            )

        if shape[2] != 3:
            raise ValueError(
                f"Weights tensor must have third dimension of length 3; got {shape[2]}"
            )
        
        # range
        if ranges is None:
            if len(wires) > 1:
                # tile ranges with iterations of range(1, n_wires)
                ranges = [(l % (len(wires) - 1)) + 1 for l in range(shape[0])]
            else:
                ranges = [0] * shape[0]
        else:
            if len(ranges) != shape[0]:
                raise ValueError(f"Range sequence must be of length {shape[0]}; got {len(ranges)}")
            for r in ranges:
                if r % len(wires) == 0:
                    raise ValueError(
                        f"Ranges must not be zero nor divisible by the number of wires; got {r}"
                    )
                
        #print(f"ranges = {ranges}")
        # imprimitive
        entangle = imprimitive if imprimitive is not None else qml.CNOT

        # p
        if p is None:
            p = np.zeros(len(wires))
        elif isinstance(p, float):
            p = np.full(len(wires), p)
        elif isinstance(p, list):
            p = np.array(p)
        else:
            p = np.array(p) # or raise ValueError(f"p must be a float or a list"")

        #print(f"p = {p}")
        # circuit
        # control_wire, target_wire: qml.CNOT(wires=[control_wire, target_wire])
        n_layers = shape[0]
        n_wires = shape[1]
        for layer in range(n_layers):
            for wire in range(n_wires):
                qml.Rot(phi=weights[layer][wire][0], theta=weights[layer][wire][1], omega=weights[layer][wire][2], wires=wire)
                qml.DepolarizingChannel(p=p[wire], wires=wire)
            for wire in range(n_wires):
                control_wire, target_wire = wire, (wire + ranges[layer]) % n_wires
                entangle(wires=[control_wire, target_wire])
                qml.DepolarizingChannel(p=p[control_wire], wires=control_wire)
                qml.DepolarizingChannel(p=p[target_wire], wires=target_wire)



    def shape(n_layers, n_wires):
        r"""Returns the expected shape of the weights tensor.

        Args:
            n_layers (int): number of layers
            n_wires (int): number of wires

        Returns:
            tuple[int]: shape
        """

        return n_layers, n_wires, 3