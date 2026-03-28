import streamlit as st
import joblib
import numpy as np
import chess

# โหลดโมเดล
ensemble = joblib.load("ensemble_model.pkl")
nn_model = joblib.load("nn_model_sklearn.pkl")

# feature function
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

# ---------------- PAGE 1: ML Model ----------------
if page == "ML Model":
    st.header("🤖 Machine Learning Model (Ensemble)")

    st.subheader("1. การเตรียมข้อมูล (Data Preparation)")
    st.write("""
    Dataset ที่ใช้ในโปรเจคนี้ประกอบด้วย 2 ชุด ได้แก่:
    - **chessData.csv** — ข้อมูลตำแหน่งหมากรุกในรูปแบบ FEN พร้อม Evaluation Score ที่คำนวณโดย Stockfish Engine
    - **chess_games.csv** — ข้อมูลประวัติการแข่งขันหมากรุก ดาวน์โหลดจาก Kaggle

    **ขั้นตอนการเตรียมข้อมูล:**
    - นำ 2 Dataset มารวมกัน และเลือกเฉพาะคอลัมน์ fen และ eval
    - ทำความสะอาดข้อมูล โดยลบแถวที่มีค่า Null และค่า Evaluation ที่เป็น Mate (#)
    - แปลงค่า Evaluation จาก centipawn เป็น pawn (หารด้วย 100)
    - Clamp ค่า Evaluation ให้อยู่ในช่วง -10 ถึง +10 เพื่อป้องกันค่าที่ผิดปกติ
    - สกัด Features 4 ตัวจากแต่ละ FEN ได้แก่ Material Balance, Mobility, Center Control และ King Safety
    - สุ่มตัวอย่าง 2,000 แถว เพื่อลดขนาด Dataset ให้เหมาะสม
    """)

    st.subheader("2. ทฤษฎีของอัลกอริทึม (Algorithm Theory)")
    st.write("""
    โมเดลที่พัฒนาเป็น **Voting Regressor** ซึ่งเป็นเทคนิค Ensemble Learning ที่รวมโมเดลหลายตัวเข้าด้วยกัน
    โดยนำผลการทำนายของแต่ละโมเดลมาเฉลี่ย เพื่อให้ได้ผลลัพธ์ที่แม่นยำและเสถียรกว่าการใช้โมเดลเดี่ยว

    โมเดลทั้ง 3 ที่ใช้ใน Voting Regressor:

    **Random Forest Regressor**
    - สร้าง Decision Tree หลายต้นจากข้อมูลที่สุ่มมา (Bootstrap Sampling)
    - แต่ละต้นทำนายผล แล้วนำมาเฉลี่ยกัน
    - ช่วยลด Overfitting ได้ดี

    **Gradient Boosting Regressor**
    - สร้างโมเดลแบบต่อเนื่อง แต่ละรอบเรียนรู้จากความผิดพลาดของรอบก่อน
    - ใช้ Gradient Descent เพื่อลด Loss Function
    - มีความแม่นยำสูงบนข้อมูลที่มี Pattern ซับซ้อน

    **Linear Regression**
    - สร้างความสัมพันธ์เชิงเส้นระหว่าง Features และ Target
    - ช่วยให้ Voting Regressor มี Base Model ที่เรียบง่าย ลด Variance โดยรวม
    """)

    st.subheader("3. ขั้นตอนการพัฒนาโมเดล (Model Development)")
    st.write("""
    1. สกัด Features จาก FEN String ของแต่ละตำแหน่ง ได้ 4 Features:
       - **Material Balance** — ผลต่างมูลค่าหมากระหว่างฝ่ายขาวและดำ
       - **Mobility** — จำนวนการเดินที่ถูกกฎหมายของผู้เล่นปัจจุบัน
       - **Center Control** — จำนวนช่องกลางกระดาน (d4, e4, d5, e5) ที่แต่ละฝ่ายโจมตี
       - **King Safety** — ตรวจสอบว่า King ของแต่ละฝ่ายถูกโจมตีหรือไม่
    2. แบ่งข้อมูลเป็น Train/Test ด้วย train_test_split (80:20)
    3. สร้าง VotingRegressor จาก Random Forest, Gradient Boosting และ Linear Regression
    4. Train โมเดลด้วยข้อมูล X_train, y_train
    5. วัดผลด้วย Mean Absolute Error (MAE) และ Mean Squared Error (MSE)
    6. บันทึกโมเดลด้วย joblib
    """)

    st.subheader("4. แหล่งอ้างอิง (References)")
    st.write("""
    - Kaggle Chess Dataset: https://www.kaggle.com/datasets/ronakbadhe/chess-evaluations
    - Scikit-learn VotingRegressor: https://scikit-learn.org/stable/modules/ensemble.html#voting-regressor
    - Python-chess Library: https://python-chess.readthedocs.io/
    - Stockfish Chess Engine: https://stockfishchess.org/
    """)

# ---------------- PAGE 2: Neural Network ----------------
elif page == "Neural Network":
    st.header("🧠 Neural Network Model")

    st.subheader("1. การเตรียมข้อมูล (Data Preparation)")
    st.write("""
    ใช้ Dataset และขั้นตอนการเตรียมข้อมูลเดียวกับ ML Model ได้แก่:
    - นำ chessData.csv และ chess_games.csv มารวมกัน
    - ทำความสะอาดข้อมูล ลบค่า Null และ Mate Score
    - แปลง Evaluation จาก centipawn เป็น pawn และ Clamp ในช่วง -10 ถึง +10
    - สกัด Features 4 ตัว: Material Balance, Mobility, Center Control, King Safety
    - แบ่ง Train/Test (80:20)
    """)

    st.subheader("2. ทฤษฎีของอัลกอริทึม (Algorithm Theory)")
    st.write("""
    โมเดลที่พัฒนาเป็น **Multilayer Perceptron (MLP)** ซึ่งเป็น Neural Network แบบ Feedforward
    ที่มีหลาย Layer เรียนรู้ความสัมพันธ์ที่ซับซ้อนแบบ Non-linear ระหว่าง Features และ Evaluation Score

    **โครงสร้างโมเดล:**
    - Input Layer: 4 neurons (Material, Mobility, Center, King Safety)
    - Hidden Layer 1: 64 neurons, ReLU Activation
    - Hidden Layer 2: 32 neurons, ReLU Activation
    - Output Layer: 1 neuron (Evaluation Score)

    **ReLU Activation Function:**
    f(x) = max(0, x) ช่วยให้โมเดลเรียนรู้ความสัมพันธ์แบบ Non-linear ได้
    และแก้ปัญหา Vanishing Gradient ที่พบใน Sigmoid/Tanh

    **Loss Function:**
    ใช้ Mean Squared Error (MSE) เพื่อวัดความแตกต่างระหว่างค่าที่ทำนายและค่าจริง
    """)

    st.subheader("3. ขั้นตอนการพัฒนาโมเดล (Model Development)")
    st.write("""
    1. เตรียม Features และ Target เดียวกับ ML Model
    2. สร้าง MLPRegressor ด้วย Scikit-learn:
       - hidden_layer_sizes = (64, 32)
       - activation = 'relu'
       - max_iter = 500
    3. Train โมเดลด้วย X_train, y_train
    4. วัดผลด้วย MAE และ MSE บน X_test
    5. เปรียบเทียบผลกับ Ensemble Model
    6. บันทึกโมเดลด้วย joblib
    """)

    st.subheader("4. แหล่งอ้างอิง (References)")
    st.write("""
    - Scikit-learn MLPRegressor: https://scikit-learn.org/stable/modules/neural_networks_supervised.html
    - Kaggle Chess Dataset: https://www.kaggle.com/datasets/ronakbadhe/chess-evaluations
    - Python-chess Library: https://python-chess.readthedocs.io/
    - ReLU Activation: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
    """)

# ---------------- PAGE 3: Test ML ----------------
elif page == "Test ML":
    st.header("🧪 Test Ensemble Model")

    st.info("กรอก FEN String ของตำแหน่งหมากรุกที่ต้องการทดสอบ แล้วกด Predict")

    fen = st.text_input("Enter FEN", placeholder="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

    if st.button("Predict (ML)"):
        try:
            features = extract_features(fen)
            pred = ensemble.predict(features)[0]

            st.success(f"Evaluation Score: {pred:.2f}")

            if pred > 0:
                st.write("♔ White is better")
            elif pred < 0:
                st.write("♚ Black is better")
            else:
                st.write("⚖️ Equal position")

        except Exception as e:
            st.error(f"Invalid FEN or prediction error: {e}")

# ---------------- PAGE 4: Test NN ----------------
elif page == "Test NN":
    st.header("🧪 Test Neural Network")

    st.info("กรอก FEN String ของตำแหน่งหมากรุกที่ต้องการทดสอบ แล้วกด Predict")

    fen = st.text_input("Enter FEN", placeholder="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

    if st.button("Predict (NN)"):
        try:
            features = extract_features(fen)
            pred_nn = nn_model.predict(features)[0]

            st.success(f"Evaluation Score: {pred_nn:.2f}")

            if pred_nn > 0:
                st.write("♔ White is better")
            elif pred_nn < 0:
                st.write("♚ Black is better")
            else:
                st.write("⚖️ Equal position")

        except Exception as e:
            st.error(f"Invalid FEN or prediction error: {e}")
