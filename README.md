# QUBO Graph Partitioning with YOLO Integration

## Overview

This project integrates QUBO (Quadratic Unconstrained Binary Optimization) with YOLO (You Only Look Once) object detection to automate graph partitioning. It analyzes images of connected graphs, extracts adjacency matrices using YOLO, and applies QUBO solutions for optimal partitioning.

## Table of Contents

- [QUBO Graph Partitioning with YOLO Integration](#qubo-graph-partitioning-with-yolo-integration)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Features](#features)
    - [Prerequisites](#prerequisites)
  - [Integration Details](#integration-details)
    - [YOLO Integration](#yolo-integration)
    - [QUBO Formulation](#qubo-formulation)
  - [Directory Structure](#directory-structure)

## Introduction

Graph partitioning is essential in parallel computing and network design. This project enhances QUBO with YOLO for robust graph analysis, automating adjacency matrix generation and optimizing partitioning strategies based on graph complexity and connectivity.

## Features

- **Automated Graph Analysis**: Generate and analyze connected graph images automatically.
- **Object Detection with YOLO**: Accurately detect nodes and edges using YOLO.
- **Optimized Partitioning**: Apply QUBO formulations for efficient graph partitioning.
- **Scalability and Flexibility**: Configurable parameters for varying graph complexities.

### Prerequisites

- Python 3.7+
- Required libraries: `numpy`, `Pillow`, `yolov5`, `pyqubo`, `networkx`
- GPU support for efficient YOLO inference

## Integration Details

### YOLO Integration

1. **Model Training**: Train YOLO on a dataset of graph images for node and edge detection.
2. **Detection**: Use YOLO to detect graph components in new images.

### QUBO Formulation

1. **Problem Formulation**: Formulate QUBO problems based on YOLO-generated adjacency matrices.
2. **Optimization**: Solve QUBO to determine optimal graph partitioning.

## Directory Structure

```bash
qubo-graph-partitioning-yolo/
├── yolov5/                     # YOLO model implementation
├── main.py                     # Main script for integration and partitioning
├── requirements.txt            # Dependencies list
├── generate_connected_graphs.py # Script for generating connected graph images
└── README.md                   # Project documentation
```