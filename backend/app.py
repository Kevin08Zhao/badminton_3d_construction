from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import List, Tuple

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .jobs import JobStore
from .pipeline_runner import run_pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STORAGE_ROOT = PROJECT_ROOT / "result" / "web"
VIDEOS_ROOT = STORAGE_ROOT / "videos"
VIDEOS_ROOT.mkdir(parents=True, exist_ok=True)

jobs = JobStore(root=STORAGE_ROOT)

app = FastAPI(title="ShuttleVision API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class UploadResp(BaseModel):
    video_id: str
    filename: str
    path: str


class CalibrationPoint(BaseModel):
    x: int
    y: int


class StartJobReq(BaseModel):
    video_id: str
    calibration_points: List[CalibrationPoint] = Field(min_length=6, max_length=6)
    perf_mode: str = Field(default="standard", pattern="^(fast|standard|precise)$")


class StartJobResp(BaseModel):
    job_id: str


@app.post("/api/upload", response_model=UploadResp)
async def upload_video(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="missing filename")
    video_id = Path(file.filename).stem + "_" + str(threading.get_ident())
    # ensure unique path
    out_path = VIDEOS_ROOT / f"{video_id}{Path(file.filename).suffix or '.mp4'}"
    content = await file.read()
    out_path.write_bytes(content)
    return UploadResp(video_id=out_path.stem, filename=file.filename, path=str(out_path))


@app.post("/api/jobs", response_model=StartJobResp)
async def start_job(req: StartJobReq):
    # resolve video
    matches = list(VIDEOS_ROOT.glob(f"{req.video_id}.*"))
    if not matches:
        raise HTTPException(status_code=404, detail="video_id not found")
    video_path = matches[0]

    job = jobs.create()
    out_dir = STORAGE_ROOT / job.id
    out_dir.mkdir(parents=True, exist_ok=True)

    def set_progress(p: float, step: str):
        job.progress = max(0.0, min(1.0, float(p)))
        job.step = step

    def log(line: str):
        job.append_log(line)

    def worker():
        job.status = "running"
        jobs.persist(job)
        try:
            result = run_pipeline(
                project_root=PROJECT_ROOT,
                video_path=video_path,
                points_2d=[(p.x, p.y) for p in req.calibration_points],
                out_dir=out_dir,
                perf_mode=req.perf_mode,
                log=log,
                set_progress=set_progress,
            )
            job.meta = result
            job.artifacts = {k: v for k, v in result.get("artifacts", {}).items() if v}
            job.status = "succeeded"
            jobs.persist(job)
        except Exception as e:
            job.error = str(e)
            job.status = "failed"
            jobs.persist(job)

    threading.Thread(target=worker, daemon=True).start()
    return StartJobResp(job_id=job.id)


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    try:
        job = jobs.get(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="job not found")
    return {
        "id": job.id,
        "status": job.status,
        "progress": job.progress,
        "step": job.step,
        "error": job.error,
        "meta": job.meta,
        "artifacts": {k: f"/api/artifacts/{job.id}/{k}" for k in job.artifacts.keys()},
    }


@app.get("/api/jobs/{job_id}/logs")
async def get_logs(job_id: str, offset: int = 0, limit: int = 200):
    try:
        job = jobs.get(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="job not found")
    offset = max(0, int(offset))
    limit = max(1, min(2000, int(limit)))
    lines = job.logs[offset : offset + limit]
    return {"offset": offset, "next_offset": offset + len(lines), "lines": lines}


@app.get("/api/artifacts/{job_id}/{name}")
async def download_artifact(job_id: str, name: str):
    try:
        job = jobs.get(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="job not found")
    if name not in job.artifacts:
        raise HTTPException(status_code=404, detail="artifact not found")
    p = Path(job.artifacts[name])
    if not p.exists():
        raise HTTPException(status_code=404, detail="file missing on disk")
    return FileResponse(str(p), filename=p.name)

