import streamlit as st
import numpy as np
from dwave.system import DWaveSampler, EmbeddingComposite
import plotly.graph_objects as go
from quantum_problems import get_maxcut_qubo, get_graph_coloring_qubo
from visualization import (
    plot_graph_result, plot_solution_histogram, 
    plot_blockchain_network, plot_quantum_mining_state,
    plot_quantum_circuit, plot_quantum_state_with_transaction
)
from problem_templates import get_templates, generate_problem
from quantum_blockchain import QuantumBlockchain, create_quantum_signature
from eth_account import Account
import secrets

st.set_page_config(page_title="Quantum Portal", layout="wide")

# Initialize session state for blockchain
if 'blockchain' not in st.session_state:
    st.session_state.blockchain = QuantumBlockchain()
if 'private_key' not in st.session_state:
    st.session_state.private_key = '0x' + secrets.token_hex(32)
    account = Account.from_key(st.session_state.private_key)
    st.session_state.account = account.address

def main():
    st.title("Quantum Portal")

    # Navigation
    page = st.sidebar.selectbox(
        "Select Interface",
        ["Quantum Optimization", "Quantum Blockchain"]
    )

    if page == "Quantum Optimization":
        show_optimization_interface()
    else:
        show_blockchain_interface()

def show_optimization_interface():
    """Original optimization interface"""
    st.sidebar.header("Problem Selection")
    problem_type = st.sidebar.selectbox(
        "Select Problem Type",
        ["MAX-CUT", "Graph Coloring"]
    )

    try:
        # Initialize the D-Wave sampler
        sampler = EmbeddingComposite(DWaveSampler())

        if problem_type == "MAX-CUT":
            st.header("MAX-CUT Problem")
            st.markdown("""
            The MAX-CUT problem involves dividing a graph into two groups (cutting) such that the number 
            of edges between the groups is maximized.
            """)

            # Template selection
            templates = get_templates("maxcut")
            template_names = ["Custom Problem"] + [t.name for t in templates]
            selected_template_name = st.selectbox(
                "Choose a problem template or create custom problem",
                template_names
            )

            if selected_template_name == "Custom Problem":
                col1, col2 = st.columns(2)
                with col1:
                    num_nodes = st.slider("Number of nodes", 2, 8, 4)
                with col2:
                    edge_probability = st.slider("Edge probability", 0.0, 1.0, 0.5,
                                                  help="Probability of an edge existing between any two nodes")

                # Generate random graph
                adjacency_matrix = np.random.choice(
                    [0, 1],
                    size=(num_nodes, num_nodes),
                    p=[1-edge_probability, edge_probability]
                )
                adjacency_matrix = np.triu(adjacency_matrix) + np.triu(adjacency_matrix, 1).T
            else:
                # Get selected template
                template = next(t for t in templates if t.name == selected_template_name)
                st.markdown(template.description)

                # Template parameters
                params = {}
                cols = st.columns(len(template.default_params))
                for (param_name, default_value), col in zip(template.default_params.items(), cols):
                    if isinstance(default_value, int):
                        params[param_name] = col.number_input(
                            param_name.replace('_', ' ').title(),
                            min_value=2,
                            value=default_value
                        )
                    elif isinstance(default_value, float):
                        params[param_name] = col.slider(
                            param_name.replace('_', ' ').title(),
                            0.0,
                            1.0,
                            default_value
                        )

                # Generate problem from template
                adjacency_matrix = generate_problem(template, **params)

            if st.button("Solve MAX-CUT"):
                with st.spinner("Quantum computation in progress..."):
                    Q = get_maxcut_qubo(adjacency_matrix)

                    # Run on D-Wave
                    response = sampler.sample_qubo(Q, num_reads=1000)

                    # Get the best solution
                    best_solution = response.first.sample

                    # Calculate cut size
                    num_nodes = len(adjacency_matrix)
                    cut_size = sum(adjacency_matrix[i][j]
                                   for i in range(num_nodes)
                                   for j in range(i+1, num_nodes)
                                   if best_solution.get(i, 0) != best_solution.get(j, 0))

                    st.success(f"Solution found! Cut size: {cut_size}")

                    # Visualize results
                    st.subheader("Graph Visualization")
                    st.markdown("The graph shows nodes divided into two groups (red and blue). "
                                 "Edges between different colored nodes contribute to the cut size.")
                    fig = plot_graph_result(adjacency_matrix, best_solution)
                    st.plotly_chart(fig, use_container_width=True)

                    # Show solution histogram
                    st.subheader("Solution Distribution")
                    st.markdown("This histogram shows the energy distribution of all solutions found. "
                                 "Lower energy values indicate better solutions.")
                    hist_fig = plot_solution_histogram(response)
                    st.plotly_chart(hist_fig, use_container_width=True)

        elif problem_type == "Graph Coloring":
            st.header("Graph Coloring Problem")
            st.markdown("""
            The Graph Coloring problem involves assigning colors to nodes such that no adjacent nodes 
            have the same color, using the minimum number of colors possible.
            """)

            # Template selection
            templates = get_templates("coloring")
            template_names = ["Custom Problem"] + [t.name for t in templates]
            selected_template_name = st.selectbox(
                "Choose a problem template or create custom problem",
                template_names
            )

            if selected_template_name == "Custom Problem":
                col1, col2, col3 = st.columns(3)
                with col1:
                    num_nodes = st.slider("Number of nodes", 2, 6, 3)
                with col2:
                    num_colors = st.slider("Number of colors", 2, 4, 3)
                with col3:
                    edge_probability = st.slider("Edge probability", 0.0, 1.0, 0.5,
                                                  help="Probability of an edge existing between any two nodes")

                # Generate random graph
                adjacency_matrix = np.random.choice(
                    [0, 1],
                    size=(num_nodes, num_nodes),
                    p=[1-edge_probability, edge_probability]
                )
                adjacency_matrix = np.triu(adjacency_matrix) + np.triu(adjacency_matrix, 1).T
            else:
                # Get selected template
                template = next(t for t in templates if t.name == selected_template_name)
                st.markdown(template.description)

                # Template parameters
                params = {}
                cols = st.columns(len(template.default_params))
                for (param_name, default_value), col in zip(template.default_params.items(), cols):
                    if isinstance(default_value, int):
                        params[param_name] = col.number_input(
                            param_name.replace('_', ' ').title(),
                            min_value=2,
                            value=default_value
                        )
                    elif isinstance(default_value, float):
                        params[param_name] = col.slider(
                            param_name.replace('_', ' ').title(),
                            0.0,
                            1.0,
                            default_value
                        )

                # Generate problem from template
                adjacency_matrix, num_colors = generate_problem(template, **params)

            if st.button("Solve Graph Coloring"):
                with st.spinner("Quantum computation in progress..."):
                    Q = get_graph_coloring_qubo(adjacency_matrix, num_colors)

                    # Run on D-Wave
                    response = sampler.sample_qubo(Q, num_reads=1000)

                    # Get the best solution
                    best_solution = response.first.sample

                    st.success("Solution found!")

                    # Visualize results
                    st.subheader("Graph Coloring Result")
                    st.markdown("Each color represents a different group. Adjacent nodes should have different colors.")
                    fig = plot_graph_result(adjacency_matrix, best_solution,
                                             problem_type="coloring", num_colors=num_colors)
                    st.plotly_chart(fig, use_container_width=True)

                    # Show solution histogram
                    st.subheader("Solution Distribution")
                    st.markdown("This histogram shows the energy distribution of all solutions found. "
                                 "Lower energy values indicate better solutions.")
                    hist_fig = plot_solution_histogram(response)
                    st.plotly_chart(hist_fig, use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please make sure you have configured your D-Wave credentials properly.")



def show_blockchain_interface():
    """Quantum Blockchain Interface"""
    st.header("Quantum Blockchain Network")
    st.markdown("""
    This interface demonstrates a quantum-enhanced blockchain network that leverages D-Wave's quantum computer
    for mining and security features.
    """)

    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Network View", "Quantum Circuit", "Transactions"])

    with tab1:
        # Blockchain Network Visualization
        st.subheader("Network Visualization")
        fig = plot_blockchain_network(st.session_state.blockchain)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        # Quantum Circuit Visualization
        st.subheader("Quantum Circuit State")

        # Get current circuit state from the quantum circuit
        current_circuit = st.session_state.blockchain.quantum_circuit
        circuit_data = {
            'qubits': range(current_circuit.num_qubits),
            'gates': [
                [{'type': 'H', 'target': i} for i in range(current_circuit.num_qubits)],
                [{'type': 'CNOT', 'control': 0, 'target': 1}]
            ]
        }

        # Get latest transaction for overlay if available
        latest_tx = (st.session_state.blockchain.pending_transactions[-1] 
                    if st.session_state.blockchain.pending_transactions else None)

        # Plot quantum circuit with transaction overlay
        circuit_fig = plot_quantum_circuit(circuit_data, latest_tx)
        st.plotly_chart(circuit_fig, use_container_width=True)

        # Plot quantum state
        if 'quantum_state' in st.session_state:
            state_fig = plot_quantum_state_with_transaction(
                st.session_state.quantum_state,
                latest_tx
            )
            st.plotly_chart(state_fig, use_container_width=True)

    with tab3:
        # Original transaction interface
        st.subheader("Your Quantum Wallet")
        st.code(f"Account Address: {st.session_state.account}")

        # Add new transaction
        st.subheader("New Transaction")
        col1, col2 = st.columns(2)
        with col1:
            recipient = st.text_input("Recipient Address")
            amount = st.number_input("Amount", min_value=0.0, value=1.0)
        with col2:
            message = f"Send {amount} to {recipient}"
            if st.button("Send Transaction"):
                signature = create_quantum_signature(
                    st.session_state.private_key,
                    message.encode()
                )
                block_index = st.session_state.blockchain.add_transaction(
                    st.session_state.account,
                    recipient,
                    amount,
                    signature
                )
                # Store current quantum state for visualization
                quantum_state = [1/np.sqrt(2) * (1 + 1j), 1/np.sqrt(2) * (1 - 1j)]  # Example state
                st.session_state.quantum_state = quantum_state
                st.success(f"Transaction will be added in block {block_index}")

        # Mining section
        st.subheader("Mining")
        col1, col2 = st.columns([2, 1])
        with col1:
            if st.button("Mine New Block"):
                with st.spinner("Quantum Mining in Progress..."):
                    new_block = st.session_state.blockchain.add_block(
                        st.session_state.blockchain.pending_transactions
                    )
                    if new_block:
                        st.session_state.blockchain.pending_transactions = []
                        st.success(f"Block {new_block.index} mined! Hash: {new_block.hash}")
                    else:
                        st.error("Mining failed. The quantum computer couldn't find a valid solution within the attempt limit. Please try again.")

        # Show current mining state
        if st.session_state.blockchain.chain:
            latest_block = st.session_state.blockchain.get_latest_block()
            mining_state = plot_quantum_mining_state(
                latest_block.quantum_difficulty,
                latest_block.nonce,
                latest_block.hash
            )
            st.plotly_chart(mining_state, use_container_width=True)

        # Display blockchain
        st.subheader("Blockchain Explorer")
        for block in st.session_state.blockchain.chain:
            with st.expander(f"Block {block.index}"):
                st.json({
                    "index": block.index,
                    "timestamp": block.timestamp,
                    "data": block.data,
                    "previous_hash": block.previous_hash,
                    "hash": block.hash,
                    "quantum_difficulty": block.quantum_difficulty
                })

        # Verify chain
        if st.button("Verify Blockchain"):
            if st.session_state.blockchain.verify_chain():
                st.success("Blockchain is valid!")
            else:
                st.error("Blockchain verification failed!")


if __name__ == "__main__":
    main()