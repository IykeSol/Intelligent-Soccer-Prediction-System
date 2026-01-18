import joblib
import os

# Load your big model
print("Loading model...")
model = joblib.load("real_data_football_model.pkl")

# Save it again with compression
print("Compressing and saving...")
joblib.dump(model, "real_data_football_model_compressed.joblib", compress=3)

print("Done! Check the new file size.")