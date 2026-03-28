import streamlit as st
import joblib
import numpy as np
import chess
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam # Import Adam optimizer

# โหลดโมเดล
ensemble = joblib.load("random_forest_model.pkl")
# Load nn_model with compile=False to avoid deserialization issues, then recompile
nn_model = load_model("nn_model.h5", compile=False)
nn_model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

# feature function (ต้องเหมือนตอน train!)
piece_values = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9
}

def extract_features(fen):
    board = chess.Board(fen)

    material = 0
    for piece_type in piece_values:
        material += len(board.pieces(piece_type, chess.WHITE)) * piece_values[piece_type]
        material -= len(board.pieces(piece_type, chess.BLACK)) * piece_values[piece_type]

    mobility = board.legal_moves.count()

    center_squares = [chess.D4, chess.E4, chess.D5, chess.E5]
    center = sum(board.is_attacked_by(chess.WHITE, sq) for sq in center_squares) - \
             sum(board.is_attacked_by(chess.BLACK, sq) for sq in center_squares)

    white_king = board.king(chess.WHITE)
    black_king = board.king(chess.BLACK)

    king_safety = int(board.is_attacked_by(chess.BLACK, white_king)) - \
                  int(board.is_attacked_by(chess.WHITE, black_king))

    return np.array([[material, mobility, center, king_safety]])

# ---------------- UI ----------------
st.title("♟️ Chess AI Evaluation System")

page = st.sidebar.selectbox(
    "Select Page",
    ["ML Model", "Neural Network", "Test ML", "Test NN"]
)

# ---------------- PAGE 1 ----------------
if page == "ML Model":
    st.header("Machine Learning Model (Ensemble)")

    st.write("""
    This model uses an ensemble of:
    - Random Forest
    - Gradient Boosting
    - Linear Regression

    Features:
    - Material balance
    - Mobility
    - Center control
    - King safety

    The model predicts the evaluation score of a chess position.
    """
)

# ---------------- PAGE 2 ----------------
elif page == "Neural Network":
    st.header("Neural Network Model")

    st.write("""
    Architecture:
    - Dense(64, ReLU)
    - Dense(32, ReLU)
    - Dense(1)

    Loss Function:
    - Mean Squared Error (MSE)

    The neural network learns non-linear relationships in chess positions.
    """
)

# ---------------- PAGE 3 ----------------
elif page == "Test ML":
    st.header("Test Ensemble Model")

    fen = st.text_input("Enter FEN")

    if st.button("Predict (ML)"):
        try:
            features = extract_features(fen)
            pred = ensemble.predict(features)[0]

            st.success(f"Evaluation Score: {pred:.2f}")

            if pred > 0:
                st.write("White is better")
            elif pred < 0:
                st.write("Black is better")
            else:
                st.write("Equal position")

        except Exception as e:
            st.error(f"Invalid FEN or prediction error: {e}")

# ---------------- PAGE 4 ----------------
elif page == "Test NN":
    st.header("Test Neural Network")

    fen = st.text_input("Enter FEN")

    if st.button("Predict (NN)"):
        try:
            features = extract_features(fen)
            pred_nn = nn_model.predict(features)[0][0] # Renamed pred to pred_nn to avoid conflict with ensemble

            st.success(f"Evaluation Score: {pred_nn:.2f}")

            if pred_nn > 0:
                st.write("White is better")
            elif pred_nn < 0:
                st.write("Black is better")
            else:
                st.write("Equal position")

        except Exception as e:
            st.error(f"Invalid FEN or prediction error: {e}")
