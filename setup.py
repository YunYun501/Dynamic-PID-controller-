from setuptools import setup, find_packages

setup(
    name='dynamic_pid_controller',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[line.strip() for line in open('requirements.txt').read().splitlines() if line.strip()],
    description='Pendulum control simulation with PID and RL hooks',
    author='Your Name',
)
