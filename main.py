from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
from sklearn.cluster import KMeans

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://heartfelt-cat-57ff20.netlify.app",  
        "http://localhost:3000",                     
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Weather clustering API"}

@app.get("/cluster")
def cluster_stations(n_clusters: int = 4):
    try:
        print("📄 Loading CSV...")
        df = pd.read_csv("weather.csv")
        print("✅ CSV loaded. Columns:", df.columns.tolist())

        # ✅ Drop rows with missing lat/lon
        df = df.dropna(subset=['latitude', 'longitude'])

        coords = df[['latitude', 'longitude']]
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['cluster'] = kmeans.fit_predict(coords)

        result = df[['station_id', 'city_name', 'latitude', 'longitude', 'cluster']]

        # ✅ Replace any remaining NaNs with None (JSON-safe)
        result = result.where(pd.notnull(result), None)

        print("✅ Clustering done. Sending result...")
        return JSONResponse(content=result.to_dict(orient="records"))

    except Exception as e:
        print("❌ ERROR in /cluster:", str(e))
        return JSONResponse(content={"error": str(e)}, status_code=500)
