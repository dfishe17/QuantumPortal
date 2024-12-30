import plotly.graph_objects as go
import networkx as nx
import numpy as np
from typing import List, Dict, Optional

def plot_graph_result(adjacency_matrix, solution, problem_type="maxcut", num_colors=None):
    """
    Visualize the graph with the solution.
    """
    G = nx.from_numpy_array(adjacency_matrix)
    pos = nx.spring_layout(G, k=1, iterations=50)  # Improved layout spacing

    # Edge trace with hover information
    edge_x = []
    edge_y = []
    edge_text = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        weight = adjacency_matrix[edge[0]][edge[1]]
        edge_text.extend([f"Edge {edge[0]}-{edge[1]}<br>Weight: {weight}", "", ""])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='#888'),
        hoverinfo='text',
        text=edge_text,
        mode='lines',
        name='Edges'
    )

    # Node positions
    node_x = []
    node_y = []
    node_text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    # Color scheme and node text based on problem type
    if problem_type == "maxcut":
        colors = ['#FF4B4B' if solution.get(i, 0) == 1 else '#4B4BFF' for i in range(len(G))]
        for i in range(len(G)):
            node_text.append(f"Node {i}<br>Partition: {'A' if solution.get(i, 0) == 1 else 'B'}")
        title = "MAX-CUT Problem Solution<br>Nodes are colored by partition (A: Red, B: Blue)"
    else:  # graph coloring
        color_map = ['#FF4B4B', '#4B4BFF', '#4BFF4B', '#FFD700']  # Red, Blue, Green, Yellow
        color_names = ['Red', 'Blue', 'Green', 'Yellow']
        colors = []
        for i in range(len(G)):
            node_colors = [solution.get(i * num_colors + c, 0) for c in range(num_colors)]
            color_idx = node_colors.index(1) if 1 in node_colors else 0
            colors.append(color_map[color_idx])
            node_text.append(f"Node {i}<br>Color: {color_names[color_idx]}")
        title = "Graph Coloring Solution<br>Each color represents a different group"

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="top center",
        textfont=dict(size=10),
        marker=dict(
            size=30,
            color=colors,
            line=dict(width=2, color='white'),
            symbol='circle'
        ),
        name='Nodes'
    )

    # Create figure with improved layout
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(
                text=title,
                y=0.95,
                x=0.5,
                xanchor='center',
                yanchor='top'
            ),
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[
                dict(
                    text="Hover over nodes and edges for details",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0, y=-0.1
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )
    )

    return fig

def plot_solution_histogram(response):
    """
    Create a histogram of solution energies with enhanced visualization.
    """
    # Extract energies and occurrences
    energies = []
    for datum in response.data():
        energies.extend([datum.energy] * datum.num_occurrences)

    # Calculate statistics
    min_energy = min(energies)
    total_samples = len(energies)
    optimal_count = sum(1 for e in energies if e == min_energy)

    fig = go.Figure(data=[
        go.Histogram(
            x=energies,
            nbinsx=30,
            name='Solutions',
            marker_color='#4B4BFF',
            hovertemplate="Energy: %{x}<br>Count: %{y}<extra></extra>"
        )
    ])

    fig.update_layout(
        title=dict(
            text="Distribution of Solution Energies<br>" +
                 f"Optimal Energy: {min_energy:.2f} (Found {optimal_count} times out of {total_samples} samples)",
            y=0.95,
            x=0.5,
            xanchor='center',
            yanchor='top'
        ),
        xaxis_title="Energy Level",
        yaxis_title="Number of Occurrences",
        bargap=0.1,
        plot_bgcolor='white',
        annotations=[
            dict(
                text="Lower energy values indicate better solutions",
                showarrow=False,
                xref="paper", yref="paper",
                x=0, y=-0.15
            )
        ]
    )

    return fig

def plot_blockchain_network(blockchain):
    """
    Create an interactive visualization of the blockchain network.
    """
    # Create nodes for each block
    node_x = []
    node_y = []
    node_text = []
    edge_x = []
    edge_y = []
    edge_text = []

    # Layout blocks in a timeline
    for i, block in enumerate(blockchain.chain):
        # Node position
        x = i
        y = 0
        node_x.append(x)
        node_y.append(y)

        # Node information
        node_text.append(
            f"Block {block.index}<br>"
            f"Timestamp: {block.timestamp}<br>"
            f"Hash: {block.hash[:10]}...<br>"
            f"Quantum Difficulty: {block.quantum_difficulty}"
        )

        # Edge to previous block
        if i > 0:
            edge_x.extend([x-1, x, None])
            edge_y.extend([0, 0, None])
            edge_text.extend([
                f"Previous Hash: {block.previous_hash[:10]}...",
                "",
                ""
            ])

    # Create edges trace
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='#888'),
        hoverinfo='text',
        text=edge_text,
        mode='lines',
        name='Block Connections'
    )

    # Create nodes trace
    node_colors = ['#1f77b4' if i == 0 else '#2ca02c' for i in range(len(node_x))]
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="top center",
        textfont=dict(size=10),
        marker=dict(
            size=40,
            color=node_colors,
            line=dict(width=2, color='white'),
            symbol='diamond'
        ),
        name='Blocks'
    )

    # Create figure with improved layout
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(
                text="Quantum Blockchain Network Visualization",
                y=0.95,
                x=0.5,
                xanchor='center',
                yanchor='top'
            ),
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[
                dict(
                    text="Hover over blocks and connections for details",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0, y=-0.1
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )
    )

    return fig

def plot_quantum_mining_state(quantum_difficulty, nonce, current_hash):
    """
    Visualize the current state of quantum mining with superposition states.
    """
    # Create binary representation of quantum difficulty
    difficulty_bits = bin(quantum_difficulty)[2:].zfill(8)
    hash_prefix = current_hash[:8]

    # Create qubit state visualization
    qubit_states = []
    for i, bit in enumerate(difficulty_bits):
        # Show superposition states
        if bit == '1':
            qubit_states.append("|1⟩")
        else:
            qubit_states.append("|0⟩")

    # Create heatmap data for quantum states
    z = [[int(bit) for bit in difficulty_bits]]

    # Enhanced heatmap with quantum state annotations
    heatmap = go.Heatmap(
        z=z,
        text=[[state for state in qubit_states]],
        texttemplate="%{text}",
        textfont={"size":20},
        colorscale=[[0, '#4B4BFF'], [1, '#FF4B4B']],
        showscale=False
    )

    # Create figure with quantum state information
    fig = go.Figure(data=[heatmap])

    # Update layout with quantum information
    fig.update_layout(
        title=dict(
            text=(f"Quantum Mining State<br>"
                  f"Current Hash: {hash_prefix}...<br>"
                  f"Nonce: {nonce}<br>"
                  f"Superposition States Active"),
            y=0.95,
            x=0.5,
            xanchor='center',
            yanchor='top'
        ),
        height=200,
        xaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False
        ),
        yaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False
        ),
        margin=dict(l=20, r=20, t=100, b=20),
        annotations=[
            dict(
                text="Quantum States",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0,
                y=-0.2
            )
        ]
    )

    return fig

def plot_quantum_circuit(circuit_data: Dict, transaction_overlay: Optional[Dict] = None):
    """
    Create an interactive visualization of quantum circuit with transaction overlay.

    Args:
        circuit_data: Dictionary containing quantum circuit state information
        transaction_overlay: Optional transaction data to overlay on the circuit
    """
    # Calculate grid dimensions for qubits
    num_qubits = len(circuit_data.get('qubits', []))
    num_steps = len(circuit_data.get('gates', []))

    # Create base figure
    fig = go.Figure()

    # Draw qubit lines
    for i in range(num_qubits):
        fig.add_trace(go.Scatter(
            x=[0, num_steps + 1],
            y=[i, i],
            mode='lines',
            line=dict(color='#888', width=1),
            name=f'Qubit {i}'
        ))

    # Draw quantum gates
    for step, gates in enumerate(circuit_data.get('gates', []), 1):
        for gate in gates:
            qubit = gate['target']
            gate_type = gate['type']

            # Gate symbol
            fig.add_trace(go.Scatter(
                x=[step],
                y=[qubit],
                mode='markers+text',
                marker=dict(
                    symbol='square',
                    size=30,
                    color='#4B4BFF',
                    line=dict(color='white', width=2)
                ),
                text=gate_type,
                textposition='middle center',
                name=f'{gate_type} Gate'
            ))

            # Draw control lines for controlled gates
            if 'control' in gate:
                control = gate['control']
                fig.add_trace(go.Scatter(
                    x=[step, step],
                    y=[control, qubit],
                    mode='lines',
                    line=dict(color='#4B4BFF', width=2),
                    name='Control Line'
                ))

    # Add transaction overlay if provided
    if transaction_overlay:
        overlay_text = (
            f"Transaction: {transaction_overlay['amount']} coins<br>"
            f"From: {transaction_overlay['sender'][:10]}...<br>"
            f"To: {transaction_overlay['recipient'][:10]}..."
        )

        fig.add_annotation(
            x=num_steps / 2,
            y=num_qubits,
            text=overlay_text,
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-40,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='#4B4BFF',
            borderwidth=2,
            font=dict(size=12)
        )

    # Update layout
    fig.update_layout(
        title=dict(
            text='Quantum Circuit Visualization',
            y=0.95,
            x=0.5,
            xanchor='center',
            yanchor='top'
        ),
        showlegend=False,
        hovermode='closest',
        xaxis=dict(
            title='Circuit Steps',
            showgrid=True,
            zeroline=False,
            range=[-0.5, num_steps + 1.5]
        ),
        yaxis=dict(
            title='Qubits',
            showgrid=True,
            zeroline=False,
            range=[-0.5, num_qubits - 0.5]
        ),
        plot_bgcolor='white',
        annotations=[
            dict(
                text="Hover over gates for details",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0,
                y=-0.15
            )
        ]
    )

    return fig

def plot_quantum_state_with_transaction(quantum_state: List[float], transaction_data: Optional[Dict] = None):
    """
    Visualize quantum state amplitudes with transaction information overlay.

    Args:
        quantum_state: List of quantum state amplitudes
        transaction_data: Optional transaction information to overlay
    """
    num_states = len(quantum_state)
    state_labels = [f"|{format(i, f'0{int(np.log2(num_states))}b')}⟩" for i in range(num_states)]

    # Create base state visualization
    fig = go.Figure()

    # Add amplitude bars
    fig.add_trace(go.Bar(
        x=state_labels,
        y=[abs(amp) ** 2 for amp in quantum_state],
        marker_color='#4B4BFF',
        name='State Amplitude'
    ))

    # Add phase angles
    fig.add_trace(go.Scatter(
        x=state_labels,
        y=[np.angle(amp, deg=True) for amp in quantum_state],
        mode='markers',
        marker=dict(
            size=10,
            color='#FF4B4B',
            symbol='diamond'
        ),
        name='Phase Angle',
        yaxis='y2'
    ))

    # Add transaction overlay if provided
    if transaction_data:
        fig.add_annotation(
            x=0.5,
            y=1.1,
            xref='paper',
            yref='paper',
            text=(
                f"Active Transaction:<br>"
                f"Amount: {transaction_data['amount']}<br>"
                f"Quantum Signature: {transaction_data.get('quantum_signature', '')[:20]}..."
            ),
            showarrow=False,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='#4B4BFF',
            borderwidth=2,
            font=dict(size=12)
        )

    # Update layout
    fig.update_layout(
        title='Quantum State Visualization',
        xaxis_title='Basis States',
        yaxis=dict(
            title='Probability Amplitude',
            side='left',
            range=[0, 1]
        ),
        yaxis2=dict(
            title='Phase Angle (degrees)',
            side='right',
            overlaying='y',
            range=[-180, 180]
        ),
        plot_bgcolor='white',
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        )
    )

    return fig