"""Write self-contained HTML rollout visualizations for trained policies."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from group11_balance.sim.env import TwoStageBalanceEnv
from group11_balance.sim.task import TASK_BALANCE, validate_target_wheel_velocity


STATE_NAMES = [
    "theta_l",
    "theta_r",
    "theta_l_dot",
    "theta_r_dot",
    "body_angle",
    "body_rate",
    "pole_angle",
    "pole_rate",
]


def collect_policy_rollout(
    model,
    *,
    algorithm_name: str,
    level: str,
    seed: int,
    task: str = TASK_BALANCE,
    target_wheel_velocity: float = 0.0,
    action_limit: float = 8000.0,
    steps: int = 1000,
    fps: int = 50,
) -> dict:
    env = TwoStageBalanceEnv(
        init_level=level,
        action_limit=action_limit,
        task=task,
        target_wheel_velocity=validate_target_wheel_velocity(target_wheel_velocity),
    )
    obs, _ = env.reset(seed=seed)
    frames: list[dict[str, object]] = []
    total_reward = 0.0
    terminated = False
    truncated = False
    last_info: dict = {}

    for step in range(int(steps)):
        action, _ = model.predict(obs, deterministic=True)
        normalized = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))
        physical = normalized * action_limit
        frames.append(
            {
                "step": step,
                "t": step * env.dt,
                "state": [float(v) for v in obs],
                "normalized_action": normalized,
                "physical_action": physical,
                "reward": float(total_reward),
            }
        )
        obs, reward, terminated, truncated, last_info = env.step(action)
        total_reward += float(reward)
        if terminated or truncated:
            frames.append(
                {
                    "step": step + 1,
                    "t": (step + 1) * env.dt,
                    "state": [float(v) for v in obs],
                    "normalized_action": None,
                    "physical_action": None,
                    "reward": float(total_reward),
                }
            )
            break

    return {
        "meta": {
            "algorithm": algorithm_name,
            "level": level,
            "task": task,
            "target_wheel_velocity": target_wheel_velocity,
            "action_limit": action_limit,
            "dt": env.dt,
            "fps": int(fps),
            "seed": int(seed),
            "wheel_radius_m": float(env.constants.wheel_radius_m),
            "total_reward": total_reward,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "failure_reason": last_info.get("failure_reason") or "",
            "state_names": STATE_NAMES,
        },
        "frames": frames,
    }


def build_rollout_html(payload: dict) -> str:
    data = json.dumps(payload, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Group11 Policy Rollout</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #f5f6f8; color: #202124; }}
    main {{ max-width: 1080px; margin: 22px auto; padding: 0 18px; }}
    canvas {{ width: 100%; aspect-ratio: 16 / 9; background: #fff; border: 1px solid #d7dce2; display: block; }}
    .bar {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin: 14px 0; }}
    button {{ border: 1px solid #aab2bd; background: white; padding: 8px 12px; cursor: pointer; }}
    input[type="range"] {{ flex: 1 1 320px; }}
    pre {{ background: #111827; color: #e5e7eb; padding: 12px; overflow: auto; font-size: 13px; }}
  </style>
</head>
<body>
<main>
  <h2 id="title">Group11 policy rollout</h2>
  <canvas id="scene" width="960" height="540"></canvas>
  <div class="bar">
    <button id="play">Pause</button>
    <input id="scrub" type="range" min="0" value="0" step="1">
    <span id="frameText"></span>
  </div>
  <pre id="stats"></pre>
</main>
<script>
const rollout = {data};
const frames = rollout.frames || [];
const meta = rollout.meta || {{}};
const canvas = document.getElementById("scene");
const ctx = canvas.getContext("2d");
const playBtn = document.getElementById("play");
const scrub = document.getElementById("scrub");
const frameText = document.getElementById("frameText");
const stats = document.getElementById("stats");
document.getElementById("title").textContent = `Group11 ${{meta.algorithm || "Policy"}} ${{meta.task || ""}} rollout`;
scrub.max = Math.max(0, frames.length - 1);
let idx = 0;
let playing = true;
let lastTime = performance.now();

function drawGround(ground, distanceM, followCamera) {{
  const w = canvas.width;
  const pxPerMeter = 220;
  const tickM = 0.1;
  const majorEvery = 5;
  const cameraLeftM = followCamera ? distanceM - w * 0.5 / pxPerMeter : -w * 0.5 / pxPerMeter;
  const firstTick = Math.floor(cameraLeftM / tickM) - 1;
  const lastTick = Math.ceil((cameraLeftM + w / pxPerMeter) / tickM) + 1;
  ctx.fillStyle = "#f8fafc";
  ctx.fillRect(0, ground, w, canvas.height - ground);
  ctx.strokeStyle = "#e5e7eb";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(0, ground);
  ctx.lineTo(w, ground);
  ctx.stroke();
  ctx.font = "12px Arial, sans-serif";
  ctx.textBaseline = "top";
  for (let i = firstTick; i <= lastTick; i++) {{
    const worldM = i * tickM;
    const x = (worldM - cameraLeftM) * pxPerMeter;
    const major = i % majorEvery === 0;
    ctx.strokeStyle = major ? "#94a3b8" : "#cbd5e1";
    ctx.lineWidth = major ? 2 : 1;
    ctx.beginPath();
    ctx.moveTo(x, ground);
    ctx.lineTo(x, ground + (major ? 20 : 10));
    ctx.stroke();
    if (major) {{
      ctx.fillStyle = "#64748b";
      ctx.fillText(worldM.toFixed(1) + "m", x + 4, ground + 24);
    }}
  }}
}}

function drawRobot(state) {{
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, w, h);
  const thetaL = state[0], thetaR = state[1];
  const body = state[4], pole = state[6];
  const wheelPosition = 0.5 * (thetaL + thetaR);
  const distanceM = wheelPosition * Number(meta.wheel_radius_m || 0.033);
  const followCamera = meta.task === "velocity";
  const x = followCamera ? w * 0.5 : w * 0.5 + Math.max(-260, Math.min(260, 15 * wheelPosition));
  const ground = h * 0.74;
  drawGround(ground, distanceM, followCamera);
  const wheelR = 34;
  const leftX = x - 55, rightX = x + 55;
  const wheelY = ground - wheelR;
  ctx.fillStyle = "#374151";
  [[leftX, thetaL], [rightX, thetaR]].forEach(([cx, theta]) => {{
    ctx.beginPath();
    ctx.arc(cx, wheelY, wheelR, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "#111827";
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.moveTo(cx, wheelY);
    ctx.lineTo(cx + Math.sin(theta) * wheelR * 0.72, wheelY - Math.cos(theta) * wheelR * 0.72);
    ctx.stroke();
  }});
  const baseX = x;
  const baseY = wheelY - 18;
  ctx.save();
  ctx.translate(baseX, baseY + 10);
  ctx.rotate(body);
  ctx.fillStyle = "#dbeafe";
  ctx.strokeStyle = "#2563eb";
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.rect(-59, -12, 118, 24);
  ctx.fill();
  ctx.stroke();
  ctx.restore();
  const poleLen = 275;
  const poleEndX = baseX + Math.sin(pole) * poleLen;
  const poleEndY = baseY - Math.cos(pole) * poleLen;
  ctx.lineWidth = 10;
  ctx.strokeStyle = "#dc2626";
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(baseX, baseY);
  ctx.lineTo(poleEndX, poleEndY);
  ctx.stroke();
  ctx.lineCap = "butt";
  ctx.fillStyle = "#111827";
  ctx.beginPath();
  ctx.arc(baseX, baseY, 8, 0, Math.PI * 2);
  ctx.fill();
}}

function draw(i) {{
  if (!frames.length) return;
  const f = frames[i];
  const s = f.state;
  drawRobot(s);
  frameText.textContent = `${{i + 1}} / ${{frames.length}}`;
  scrub.value = i;
  const wheelVelocity = 0.5 * (s[2] + s[3]);
  stats.textContent =
    `algorithm: ${{meta.algorithm}}\\n` +
    `task: ${{meta.task}}, level: ${{meta.level}}, target_wheel_velocity: ${{meta.target_wheel_velocity}}\\n` +
    `time: ${{Number(f.t).toFixed(3)}} s, cumulative_reward: ${{Number(f.reward).toFixed(3)}}\\n` +
    `body_angle_deg: ${{(s[4] * 180 / Math.PI).toFixed(3)}}\\n` +
    `pole_angle_deg: ${{(s[6] * 180 / Math.PI).toFixed(3)}}\\n` +
    `wheel_velocity_rad_s: ${{wheelVelocity.toFixed(3)}}\\n` +
    `normalized_action: ${{f.normalized_action === null ? "" : Number(f.normalized_action).toFixed(5)}}\\n` +
    `physical_action_rad_s2: ${{f.physical_action === null ? "" : Number(f.physical_action).toFixed(2)}}\\n` +
    `terminated: ${{meta.terminated}}, truncated: ${{meta.truncated}}, reason: ${{meta.failure_reason || ""}}`;
}}

function tick(now) {{
  const fps = Number(meta.fps || 50);
  if (playing && frames.length && now - lastTime >= 1000 / fps) {{
    if (idx < frames.length - 1) {{
      idx += 1;
    }} else {{
      playing = false;
      playBtn.textContent = "Play";
    }}
    draw(idx);
    lastTime = now;
  }}
  requestAnimationFrame(tick);
}}

playBtn.onclick = () => {{
  playing = !playing;
  playBtn.textContent = playing ? "Pause" : "Play";
}};
scrub.oninput = () => {{
  idx = Number(scrub.value);
  draw(idx);
}};

draw(0);
requestAnimationFrame(tick);
</script>
</body>
</html>
"""


def write_rollout_html(payload: dict, output: str | Path) -> Path:
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_rollout_html(payload), encoding="utf-8")
    return out


def write_policy_rollout_html(
    model,
    *,
    output: str | Path,
    algorithm_name: str,
    level: str,
    seed: int,
    task: str = TASK_BALANCE,
    target_wheel_velocity: float = 0.0,
    action_limit: float = 8000.0,
    steps: int = 1000,
    fps: int = 50,
) -> Path:
    payload = collect_policy_rollout(
        model,
        algorithm_name=algorithm_name,
        level=level,
        seed=seed,
        task=task,
        target_wheel_velocity=target_wheel_velocity,
        action_limit=action_limit,
        steps=steps,
        fps=fps,
    )
    return write_rollout_html(payload, output)
