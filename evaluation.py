"""Evaluate one Agent without drawing. Same rules as live training."""
import pygame
from agent import Agent
from network import NeuralNetwork
from track import make_track
from settings import SIMULATION_DT


def run_attempt(network, track=None, **agent_options):
    """Simulate one attempt and return the finished Agent (for detailed stats)."""
    track = track or make_track('classic')
    agent = Agent(network, track, **agent_options)
    while agent.active:
        agent.step(SIMULATION_DT, track)
    return agent


def evaluate(network, track=None, **agent_options):
    agent = run_attempt(network, track, **agent_options)
    return agent.fitness, agent.status


if __name__ == '__main__':
    pygame.init()
    try:
        fitness, reason = evaluate(NeuralNetwork(seed=42))
        print(f'Fitness: {fitness:.1f}\nAttempt ended: {reason}')
    finally:
        pygame.quit()
