import math
import random


class NeuralNetwork:
    def __init__(self, seed=None):
        rng = random.Random(seed)

        # Each hidden neuron has a weight for every input.
        self.hidden_weights = [
            [rng.uniform(-0.5, 0.5) for _ in range(11)]
            for _ in range(8)
        ]
        self.hidden_biases = [
            rng.uniform(-0.5, 0.5) for _ in range(8)
        ]

        # Each output neuron has a weight for every hidden neuron.
        self.output_weights = [
            [rng.uniform(-0.5, 0.5) for _ in range(8)]
            for _ in range(3)
        ]
        self.output_biases = [
            rng.uniform(-0.5, 0.5) for _ in range(3)
        ]

    def forward(self, observation):
        if len(observation) != 11:
            raise ValueError("Expected exactly 11 observation values")

        hidden = self.calculate_layer(
            observation,
            self.hidden_weights,
            self.hidden_biases,
        )

        return self.calculate_layer(
            hidden,
            self.output_weights,
            self.output_biases,
        )

    @staticmethod
    def calculate_layer(inputs, weights, biases):
        outputs = []

        for neuron_weights, bias in zip(weights, biases):
            total = bias

            for value, weight in zip(inputs, neuron_weights):
                total += value * weight

            outputs.append(math.tanh(total))

        return outputs