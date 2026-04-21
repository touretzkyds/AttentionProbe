"""
Demo 1: Pronoun Resolution Attention Probe
Refactored version using the new modular architecture.
"""

from demo_base import BaseDemo


def main():
    """Run Demo 1: Pronoun Resolution."""
    global demo
    demo = BaseDemo("pronoun_resolution")
    demo.run()


if __name__ == '__main__':
    main() 