from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import boto3
import joblib
import os

app = FastAPI()

CLOUD_BUCKET = os.environ["CLOUD_BUCKET"]
SPACES_REGION = os.environ["SPACES_REGION"]
SPACES_ENDPOINT = os.environ.get(
    "SPACES_ENDPOINT", f"https://{SPACES_REGION}.digitaloceanspaces.com"
)
MODEL_KEY = "models/latest/model.pkl"
MODEL_PATH = os.path.expanduser("~/models/model.pkl")


def download_model():
    """
    Tai file model.pkl tu DigitalOcean Spaces ve may khi server khoi dong.

    Ham nay duoc goi mot lan khi module duoc import.
    Bien AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY duoc dat trong systemd service.
    """
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    client = boto3.client(
        "s3",
        region_name=SPACES_REGION,
        endpoint_url=SPACES_ENDPOINT,
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    )
    client.download_file(CLOUD_BUCKET, MODEL_KEY, MODEL_PATH)
    print(f"Model da duoc tai xuong tu s3://{CLOUD_BUCKET}/{MODEL_KEY}")


download_model()
model = joblib.load(MODEL_PATH)


class PredictRequest(BaseModel):
    features: list[float]


@app.get("/health")
def health():
    """
    Endpoint kiem tra suc khoe server.
    GitHub Actions goi endpoint nay sau khi deploy de xac nhan server dang chay.

    Tra ve: {"status": "ok"}
    """
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    """
    Endpoint suy luan chinh.

    Dau vao : JSON {"features": [f1, f2, ..., f12]}
    Dau ra  : JSON {"prediction": <0|1|2>, "label": <"thap"|"trung_binh"|"cao">}

    Thu tu 12 dac trung (khop voi thu tu trong FEATURE_NAMES cua test):
        fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
        chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
        pH, sulphates, alcohol, wine_type
    """
    if len(req.features) != 12:
        raise HTTPException(
            status_code=400, detail="Expected 12 features (wine quality)"
        )

    pred = int(model.predict([req.features])[0])
    labels = {0: "thap", 1: "trung_binh", 2: "cao"}
    return {"prediction": pred, "label": labels.get(pred, "unknown")}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
