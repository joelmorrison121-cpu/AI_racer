"""Population creation, mutation, breeding and saving shared by all trainers."""
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from network import NeuralNetwork, BASE_INPUTS

POPULATION_SIZE = 30
GENERATIONS = 20
ELITE_COUNT = 2
PARENT_COUNT = 6
NEW_NETWORK_COUNT = 4
MUTATION_RATE = 0.15
MUTATION_STRENGTH = 0.2


def mutate(parent, rng):
    child = copy.deepcopy(parent)
    for matrix in (child.hidden_weights, child.output_weights):
        for row in matrix:
            for i in range(len(row)):
                if rng.random() < MUTATION_RATE:
                    row[i] += rng.gauss(0, MUTATION_STRENGTH)
    for biases in (child.hidden_biases, child.output_biases):
        for i in range(len(biases)):
            if rng.random() < MUTATION_RATE:
                biases[i] += rng.gauss(0, MUTATION_STRENGTH)
    return child


def new_population(rng, size=POPULATION_SIZE, inputs=BASE_INPUTS):
    return [NeuralNetwork(seed=rng.randrange(1_000_000_000), inputs=inputs) for _ in range(size)]


def breed(ranked_networks, rng, size=POPULATION_SIZE):
    """Input must already be sorted from highest fitness to lowest.

    The next generation is: ELITE_COUNT unchanged copies, mutated copies of
    the top PARENT_COUNT, then NEW_NETWORK_COUNT fresh random networks.
    """
    children = [copy.deepcopy(n) for n in ranked_networks[:ELITE_COUNT]]
    parents = ranked_networks[:PARENT_COUNT]
    while len(children) < size - NEW_NETWORK_COUNT:
        children.append(mutate(rng.choice(parents), rng))
    inputs = ranked_networks[0].input_count
    children.extend(NeuralNetwork(seed=rng.randrange(1_000_000_000), inputs=inputs)
                    for _ in range(NEW_NETWORK_COUNT))
    return children


def create_run_directory(base=None):
    # Each run has its own files; the root champion.json is never overwritten.
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
    root = Path(base) if base else Path(__file__).resolve().parent / 'training_runs'
    path = root / stamp
    path.mkdir(parents=True)
    return path


def network_to_dict(network):
    return dict(hidden_weights=network.hidden_weights, hidden_biases=network.hidden_biases,
                output_weights=network.output_weights, output_biases=network.output_biases)


def network_from_dict(data):
    network = NeuralNetwork(seed=0)
    network.hidden_weights = [list(row) for row in data['hidden_weights']]
    network.hidden_biases = list(data['hidden_biases'])
    network.output_weights = [list(row) for row in data['output_weights']]
    network.output_biases = list(data['output_biases'])
    return network


def write_json(path, data):
    """Write atomically so an interrupted save never leaves a half-written file."""
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2), encoding='utf-8')
    temporary.replace(path)


def save_champion(network, fitness, generation, path, **extra):
    write_json(path, dict(generation=generation, fitness=fitness,
                          **network_to_dict(network), **extra))


def load_champion(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    return network_from_dict(data), data


def rng_state_to_json(rng):
    version, internal, gauss_next = rng.getstate()
    return [version, list(internal), gauss_next]


def rng_state_from_json(rng, state):
    version, internal, gauss_next = state
    rng.setstate((version, tuple(internal), gauss_next))
