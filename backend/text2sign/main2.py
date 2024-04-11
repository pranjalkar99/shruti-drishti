from lang import work
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

last_path=""


@app.get("/last-path")
def get_last_path():
    global last_path
    """
    This endpoint returns the last path accessed.
    """
    return {"last_path": last_path}


@app.post("/classify")
async def classify_text(prompt: str):
    global last_path
    response = work.invoke({"query": prompt})
    print(response)
    last_path = response['videos']['path']
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main2:app", host="0.0.0.0", port=7002)
