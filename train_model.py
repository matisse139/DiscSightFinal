import time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


def train_disc_sight_model():
    total_start_time = time.time()
    print("=" * 50)
    print("🚀 STARTING DISC SIGHT MODEL TRAINING PIPELINE")
    print("=" * 50)

    # Set seed for reproducible noise generation
    np.random.seed(42)

    # 1. Load the dataset
    print("\n[STEP 1/7] Loading dataset...")
    step_start = time.time()
    df = pd.read_csv("ultimate_biomechanics_data.csv")
    print(
        f"✓ Dataset loaded successfully in {time.time() - step_start:.2f}s!"
    )
    print(f"  └─ Total rows: {df.shape[0]}, Total columns: {df.shape[1]}")

    # --- INJECT SYNTHETIC NOISE ---
    print("\n[NOISE INJECTION] Adding realistic noise to synthetic data...")

    # A. Add Gaussian noise to continuous feature columns (Simulating sensor jitter/error)
    feature_cols = [col for col in df.columns if col != "label"]
    for col in feature_cols:
        # Add 5% noise relative to standard deviation of each feature
        feature_noise = np.random.normal(0, df[col].std() * 0.05, size=len(df))
        df[col] = df[col] + feature_noise

    # B. Inject 5% label flips (Simulating annotation variance or human error)
    label_noise_mask = np.random.rand(len(df)) < 0.05
    df.loc[label_noise_mask, "label"] = 1 - df.loc[label_noise_mask, "label"]

    print("✓ Added 5% Gaussian noise to feature readings")
    print(f"✓ Randomly flipped {label_noise_mask.sum()} target labels (~5%)")
    # ------------------------------

    # 2. Separate features (X) and label (y)
    print("\n[STEP 2/7] Separating features (X) and labels (y)...")
    step_start = time.time()
    X = df.drop("label", axis=1)
    y = df["label"]
    print(f"✓ Features and labels split in {time.time() - step_start:.2f}s!")
    print(
        f"  └─ Feature matrix shape: {X.shape} | Label vector shape: {y.shape}"
    )

    # 3. Split data (80% training, 20% testing)
    print("\n[STEP 3/7] Splitting data into train/test sets (80/20)...")
    step_start = time.time()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"✓ Data split complete in {time.time() - step_start:.2f}s!")
    print(f"  └─ Training samples: {X_train.shape[0]}")
    print(f"  └─ Testing samples:  {X_test.shape[0]}")

    # 4. Initialize the Random Forest model
    print("\n[STEP 4/7] Initializing Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    print("✓ Model initialized (n_estimators=100, random_state=42)")

    # 5. Train the model
    print("\n[STEP 5/7] Training the model (this might take a moment)...")
    step_start = time.time()
    model.fit(X_train, y_train)
    print(
        f"✓ Model training finished successfully in {time.time() - step_start:.2f}s!"
    )

    # 6. Evaluate accuracy
    print("\n[STEP 6/7] Evaluating model performance on test set...")
    step_start = time.time()
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"✓ Evaluation completed in {time.time() - step_start:.2f}s!")

    print("\n" + "-" * 40)
    print(f"📊 MODEL ACCURACY: {accuracy * 100:.2f}%")
    print("-" * 40)
    print("\n📋 Classification Report:")
    print(classification_report(y_test, predictions))

    # 7. Save the trained model
    print("\n[STEP 7/7] Saving trained model to disk...")
    step_start = time.time()
    joblib.dump(model, "ultimate_form_model.pkl")
    print(
        f"✓ Saved 'ultimate_form_model.pkl' in {time.time() - step_start:.2f}s!"
    )

    print("\n" + "=" * 50)
    print(
        f"🎉 PIPELINE COMPLETE IN {time.time() - total_start_time:.2f} SECONDS!"
    )
    print("=" * 50)


if __name__ == "__main__":
    train_disc_sight_model()

joblib.dump(model, 'ultimate_form_model.pkl')
print("✓ Model successfully exported to ultimate_form_model.pkl")
