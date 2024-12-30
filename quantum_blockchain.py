import hashlib
import numpy as np
from quantum_circuits import QuantumCircuit, create_quantum_hash, verify_quantum_signature
from eth_account import Account
from eth_account.messages import encode_defunct
import time
import json
from typing import Optional, List, Dict, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BlockVerificationError(Exception):
    """Custom exception for blockchain verification errors"""
    pass

class QuantumBlock:
    def __init__(self, index, timestamp, data, previous_hash):
        """Initialize block with improved deterministic hashing."""
        try:
            # Initialize basic properties first
            self.index = index
            self.timestamp = str(timestamp)  # Ensure timestamp is always string
            self.data = data
            self.previous_hash = previous_hash
            self.nonce = 0
            self.quantum_state = None
            self.hash = None  # Initialize hash as None first
            self.quantum_difficulty = 1  # Set default difficulty
            logger.info(f"Block {index} properties initialized")

            # Calculate hash only once after all properties are set
            self.hash = self.calculate_hash()
            logger.info(f"Block {index} initialized successfully with hash: {self.hash[:10]}...")
        except Exception as e:
            logger.error(f"Error initializing block {index}: {str(e)}")
            raise BlockVerificationError(f"Block initialization failed: {str(e)}")

    def calculate_hash(self) -> str:
        """Calculate block hash using deterministic quantum-enhanced hashing."""
        try:
            # Ensure all properties are serialized consistently
            block_data = {
                'index': self.index,
                'timestamp': str(self.timestamp),  # Ensure timestamp is string
                'data': self._serialize_value(self.data),
                'previous_hash': self.previous_hash,
                'nonce': self.nonce,
                'quantum_difficulty': self.quantum_difficulty
            }

            # Use canonical JSON serialization for consistency
            block_string = json.dumps(block_data, sort_keys=True, separators=(',', ':'))
            logger.debug(f"Block {self.index} data for hashing: {block_string}")

            # Calculate hash components
            classical_hash = hashlib.sha256(block_string.encode('utf-8')).hexdigest()
            quantum_bits = format(int(classical_hash[:8], 16), '032b')

            # Combine hash components deterministically
            final_hash = classical_hash[:32] + quantum_bits
            logger.debug(f"Block {self.index} hash components: classical={classical_hash[:32]}, quantum={quantum_bits}")

            return final_hash
        except Exception as e:
            logger.error(f"Hash calculation failed for block {self.index}: {str(e)}")
            raise BlockVerificationError(f"Failed to calculate block hash: {str(e)}")

    @staticmethod
    def _serialize_value(v):
        """Helper method to ensure consistent serialization"""
        if isinstance(v, (dict, list)):
            return json.dumps(v, sort_keys=True, separators=(',', ':'))
        return str(v)

class QuantumBlockchain:
    def __init__(self):
        """Initialize blockchain with deterministic genesis block."""
        self.chain: List[QuantumBlock] = []
        self.pending_transactions: List[Dict] = []
        self.quantum_circuit = QuantumCircuit(8)
        try:
            self.create_genesis_block()
            logger.info("Blockchain initialized successfully with genesis block")
        except Exception as e:
            logger.error(f"Blockchain initialization failed: {str(e)}")
            raise BlockVerificationError("Blockchain initialization failed")

    def create_genesis_block(self):
        """Create genesis block with deterministic initialization."""
        try:
            # Create genesis block with minimal data and fixed difficulty
            genesis_block = QuantumBlock(0, "2024-01-01", "Genesis Block", "0")
            genesis_block.quantum_difficulty = 1  # Set fixed difficulty for genesis

            # Store initial hash for verification
            initial_hash = genesis_block.hash

            # Verify the hash calculation is consistent
            verification_hash = genesis_block.calculate_hash()
            if initial_hash != verification_hash:
                logger.error(f"Genesis block hash mismatch: Block={initial_hash[:10]}... != Verification={verification_hash[:10]}...")
                raise BlockVerificationError("Genesis block hash verification failed")

            # Add to chain only after verification
            self.chain.append(genesis_block)
            logger.info(f"Genesis block created successfully with hash: {genesis_block.hash[:10]}...")
        except Exception as e:
            logger.error(f"Failed to create genesis block: {str(e)}")
            raise BlockVerificationError(f"Genesis block creation failed: {str(e)}")

    def calculate_quantum_difficulty(self, block: QuantumBlock) -> None:
        """Calculate and set quantum difficulty for a block."""
        try:
            if block.index == 0:  # Genesis block
                block.quantum_difficulty = 1
            else:
                difficulty = self.generate_quantum_difficulty()
                block.quantum_difficulty = difficulty
            logger.debug(f"Set quantum difficulty {block.quantum_difficulty} for block {block.index}")
        except Exception as e:
            logger.error(f"Failed to calculate quantum difficulty: {str(e)}")
            block.quantum_difficulty = 1  # Fallback difficulty

    def get_latest_block(self) -> Optional[QuantumBlock]:
        return self.chain[-1] if self.chain else None

    def generate_quantum_difficulty(self) -> int:
        """Generate quantum-based mining difficulty."""
        try:
            state = self.quantum_circuit.create_superposition_state()
            measured_state = self.quantum_circuit.measure_state(state)
            return int(''.join(str(bit) for bit in measured_state), 2)
        except Exception as e:
            logger.error(f"Quantum difficulty generation failed: {str(e)}")
            return 1  # Fallback difficulty

    def verify_transaction_signature(self, transaction: Dict) -> bool:
        """Verify the quantum signature of a transaction."""
        try:
            if not all(k in transaction for k in ['sender', 'recipient', 'amount', 'quantum_signature']):
                logger.error("Invalid transaction structure")
                return False

            if not transaction['quantum_signature']:
                logger.error("Missing quantum signature")
                return False

            message = f"Send {transaction['amount']} to {transaction['recipient']}"
            return verify_quantum_signature(message.encode(), transaction['quantum_signature'])
        except Exception as e:
            logger.error(f"Transaction signature verification failed: {str(e)}")
            return False

    def verify_block_data(self, block: QuantumBlock) -> bool:
        """Verify the data structure and content of a block."""
        try:
            if isinstance(block.data, list):
                return all(isinstance(tx, dict) and 
                         all(k in tx for k in ['sender', 'recipient', 'amount']) 
                         for tx in block.data)
            return block.data == "Genesis Block"
        except Exception as e:
            logger.error(f"Block data verification failed: {str(e)}")
            return False

    def verify_chain(self) -> bool:
        """Verify the integrity of the blockchain with improved error handling."""
        try:
            if not self.chain:
                return True

            # Verify genesis block
            genesis = self.chain[0]
            if not self._verify_genesis_block(genesis):
                return False

            # Verify subsequent blocks
            for i in range(1, len(self.chain)):
                current_block = self.chain[i]
                previous_block = self.chain[i-1]

                if not self._verify_block_sequence(current_block, previous_block):
                    return False

                if not self.verify_block_data(current_block):
                    logger.error(f"Invalid block data structure at index {i}")
                    return False

                # Verify transactions
                if isinstance(current_block.data, list) and current_block.data:
                    if not self._verify_block_transactions(current_block):
                        return False

            return True
        except Exception as e:
            logger.error(f"Chain verification failed: {str(e)}")
            return False

    def _verify_genesis_block(self, genesis: QuantumBlock) -> bool:
        """Verify genesis block integrity."""
        if genesis.index != 0 or genesis.previous_hash != "0":
            logger.error("Invalid genesis block structure")
            return False
        try:
            if genesis.hash != genesis.calculate_hash():
                logger.error("Genesis block hash mismatch")
                return False
            return True
        except Exception as e:
            logger.error(f"Genesis block verification failed: {str(e)}")
            return False

    def _verify_block_sequence(self, current_block: QuantumBlock, previous_block: QuantumBlock) -> bool:
        """Verify the sequence and linking of blocks."""
        try:
            if current_block.index != previous_block.index + 1:
                logger.error(f"Invalid block sequence at index {current_block.index}")
                return False
            if current_block.previous_hash != previous_block.hash:
                logger.error(f"Invalid block chain linking at index {current_block.index}")
                return False
            if current_block.hash != current_block.calculate_hash():
                logger.error(f"Block hash mismatch at index {current_block.index}")
                return False
            return True
        except Exception as e:
            logger.error(f"Block sequence verification failed: {str(e)}")
            return False

    def _verify_block_transactions(self, block: QuantumBlock) -> bool:
        """Verify all transactions in a block."""
        try:
            for tx in block.data:
                if not self.verify_transaction_signature(tx):
                    logger.error(f"Invalid transaction signature in block {block.index}")
                    return False
            return True
        except Exception as e:
            logger.error(f"Block transactions verification failed: {str(e)}")
            return False

    def add_block(self, data):
        """Add a new block to the chain with quantum mining."""
        previous_block = self.get_latest_block()
        if not previous_block:
            return None

        new_block = QuantumBlock(
            index=previous_block.index + 1,
            timestamp=str(np.datetime64('now')),
            data=data,
            previous_hash=previous_block.hash
        )

        self.calculate_quantum_difficulty(new_block) # Use the improved method
        new_block.hash = new_block.calculate_hash()

        if self.quantum_mine_block(new_block):
            self.chain.append(new_block)
            return new_block
        return None

    def quantum_mine_block(self, block, target_zeros=1):
        """Mine block using quantum-assisted approach."""
        max_attempts = 100
        attempts = 0

        while not block.hash.startswith('0' * target_zeros) and attempts < max_attempts:
            # Update nonce and recalculate hash
            block.nonce += 1
            block.hash = block.calculate_hash()

            # Apply quantum operations
            gate_index = int(block.hash[:8], 16) % 8
            self.quantum_circuit.apply_quantum_gate("X", [gate_index])

            attempts += 1
            time.sleep(0.1)  # Prevent UI blocking

        return block.hash.startswith('0' * target_zeros)


    def add_transaction(self, sender: str, recipient: str, amount: float, quantum_signature: Optional[str] = None) -> Optional[int]:
        """Add a new transaction with improved validation."""
        try:
            if not sender or not recipient or amount <= 0:
                logger.error("Invalid transaction parameters")
                return None

            transaction = {
                'sender': sender,
                'recipient': recipient,
                'amount': amount,
                'quantum_signature': quantum_signature,
                'timestamp': str(np.datetime64('now'))
            }

            if not self.verify_transaction_signature(transaction):
                logger.error("Transaction signature verification failed")
                return None

            self.pending_transactions.append(transaction)
            latest_block = self.get_latest_block()
            return latest_block.index + 1 if latest_block else 0
        except Exception as e:
            logger.error(f"Failed to add transaction: {str(e)}")
            return None

def create_quantum_signature(private_key: str, message: Union[str, bytes]) -> str:
    """Create a quantum-enhanced digital signature with improved error handling."""
    try:
        account = Account.from_key(private_key)
        message_to_sign = encode_defunct(text=message.decode('utf-8') if isinstance(message, bytes) else message)
        eth_signature = account.sign_message(message_to_sign)
        quantum_hash = create_quantum_hash(eth_signature.signature.hex())
        return f"{eth_signature.signature.hex()}{quantum_hash:064b}"
    except Exception as e:
        logger.error(f"Failed to create quantum signature: {str(e)}")
        raise BlockVerificationError(f"Quantum signature creation failed: {str(e)}")