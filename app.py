import base64
import hashlib
import json
import math
import os
import tempfile
import warnings
import cv2

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import joblib
import numpy as np
from google import genai
from google.genai import types

# Suppress non-critical C++ / Protobuf warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')

# Initialize Flask App
app = Flask(__name__, static_folder=".")
CORS(app)

# Initialize Google GenAI client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY", ""))

# ==========================================
# LOAD TRAINED MACHINE LEARNING MODEL
# ==========================================
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'ultimate_form_model.pkl')
try:
    ml_model = joblib.load(MODEL_PATH)
    print(f"✓ ML Model successfully loaded from {MODEL_PATH}")
except Exception as e:
    ml_model = None
    print(f"⚠️ Warning: Could not load '{MODEL_PATH}'. Error: {e}")

# ==========================================
# MEDIAPIPE INITIALIZATION
# ==========================================
try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    pose_tracker = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5
    )
    MP_AVAILABLE = True
    print("✓ MediaPipe Pose successfully initialized.")
except ImportError:
    pose_tracker = None
    MP_AVAILABLE = False
    print("⚠️ Warning: MediaPipe is not installed. Falling back to dynamic heuristics.")


# ==========================================
# DRILL & GAMIFICATION KNOWLEDGE BASE
# ==========================================
DRILL_DATABASE = {
    "Stance & Balance": {
        "title": "Athletic Base & Pivot Weight Transfer Routine",
        "duration": "10 Minutes",
        "description": "Focus on lowering your center of gravity and stabilizing the pivot foot during release.",
        "steps": [
            "1. Perform 10 drop-step lunges onto your pivot foot without throwing.",
            "2. Practice 15 dry-flicks maintaining dynamic equilibrium over your flexed knee.",
            "3. Finish with 20 low-release backhand throws focusing on firm footing."
        ],
        "video_url": "https://www.youtube-nocookie.com/embed/As1X0JNWiLY"
    },
    "Reach-back Extension": {
        "title": "Full Reach-Back & Extension Mechanics",
        "duration": "10 Minutes",
        "description": "Increases shoulder mobility and full arm extension to maximize distance and disc velocity.",
        "steps": [
            "1. Dynamic shoulder stretches and band pull-aparts (2 minutes).",
            "2. 'Reach & Hold' isolation exercises: Extend back, pause 2s, bring disc to chest (15 reps).",
            "3. 20 full-power extension throws target practice."
        ],
        "video_url": "https://www.youtube-nocookie.com/embed/S5Y5VnwtGYA"
    },
    "Core Rotation": {
        "title": "Kinematic Core Torque & Hips Isolation",
        "duration": "10 Minutes",
        "description": "Engage your hips and obliques to drive power from lower body to disc release.",
        "steps": [
            "1. Standing core twists holding a disc or light weight (2 sets x 15 reps).",
            "2. Hip-first rotation throws: Fire hips before shoulders break forward (20 throws).",
            "3. Follow-through rotational hold practice."
        ],
        "video_url": "https://www.youtube-nocookie.com/embed/rpqT8J6BzvY"
    },
    "Release Plane": {
        "title": "Plane Consistency & Angle Control Routine",
        "duration": "10 Minutes",
        "description": "Corrects wrist tilt and uneven plane trajectories (flat, hyzer, anhyzer stability).",
        "steps": [
            "1. Line-of-sight visual alignment drills (15 reps).",
            "2. Strict flat-release wrist snap isolation throws against a net/wall (20 reps).",
            "3. Variable angle control drill: 5 flat, 5 hyzer, 5 anhyzer throws."
        ],
        "video_url": "https://www.youtube-nocookie.com/embed/As1X0JNWiLY"
    },
    "Follow-through": {
        "title": "Full Kinetic Chain Follow-Through Drill",
        "duration": "10 Minutes",
        "description": "Ensures energy dissipates safely across the shoulder plane while maintaining throw trajectory.",
        "steps": [
            "1. Unweighted arm whip follow-through rotations (15 reps per side).",
            "2. Target-line lock-in: Pointing throwing finger at target post-release (20 throws).",
            "3. High-velocity release with exaggerated 180° body rotation."
        ],
        "video_url": "https://www.youtube-nocookie.com/embed/S5Y5VnwtGYA"
    }
}

MICRO_DRILLS = [
    {
        "id": "elbow_snap",
        "title": "90° Elbow Snap Master",
        "tagline": "Power Pocket Mechanics",
        "target_metric": "Reach-back Extension",
        "target_score": 85,
        "description": "Perform 3 sets of 12 rapid power-pocket pulls. Focus on holding an acute 90° elbow angle before whipping through.",
        "xp_reward": 150
    },
    {
        "id": "brace_lock",
        "title": "Brace Leg Lock",
        "tagline": "Kinetic Energy Transfer",
        "target_metric": "Stance & Balance",
        "target_score": 85,
        "description": "Complete 15 plant-and-freeze dry throws. Maintain a firm front knee to block forward momentum and force hip whip.",
        "xp_reward": 150
    },
    {
        "id": "hip_torque",
        "title": "Hip-Torque Driver",
        "tagline": "Rotational Acceleration",
        "target_metric": "Core Rotation",
        "target_score": 88,
        "description": "Do 20 rapid hip-led rotations holding a towel. Ensure the rear hip initiates 100% of the turn before the upper torso.",
        "xp_reward": 200
    },
    {
        "id": "flat_rail",
        "title": "Flat Rail Release",
        "tagline": "Disc Trajectory Precision",
        "target_metric": "Release Plane",
        "target_score": 85,
        "description": "Execute 20 flat-plane wrist snaps along a eye-level horizontal tape line to eliminate hyzer wobble.",
        "xp_reward": 150
    }
]


def hex_to_bgr(hex_str):
    """Converts a hex string (#RRGGBB) to OpenCV BGR format."""
    hex_str = hex_str.lstrip('#')
    if len(hex_str) != 6:
        return (68, 68, 239)
    r = int(hex_str[0:2], 16)
    g = int(hex_str[2:4], 16)
    b = int(hex_str[4:6], 16)
    return (b, g, r)


def calculate_angle(a, b, c):
    """Calculates angle (in degrees) at joint B given points A, B, and C."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
        a[1] - b[1], a[0] - b[0]
    )
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return angle


def extract_keypoints_mediapipe(frame):
    """Extracts 2D and 3D joint coordinates from image using MediaPipe Pose."""
    h, w, _ = frame.shape
    keypoints = {}
    keypoints_3d = {}

    if MP_AVAILABLE and pose_tracker:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_tracker.process(rgb_frame)

        if results.pose_landmarks:
            target_landmarks = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
            for idx in target_landmarks:
                lm = results.pose_landmarks.landmark[idx]
                keypoints[idx] = [int(lm.x * w), int(lm.y * h)]
                keypoints_3d[idx] = [
                    float((lm.x - 0.5) * 4.0),
                    float((0.5 - lm.y) * 4.0),
                    float(-lm.z * 4.0)
                ]
            return keypoints, keypoints_3d

    center_x = w // 2
    top_y = int(h * 0.15)
    scale = h / 600.0

    fallback_2d = {
        0:  [center_x, top_y],
        11: [center_x - int(40 * scale), top_y + int(80 * scale)],
        12: [center_x + int(40 * scale), top_y + int(80 * scale)],
        13: [center_x - int(80 * scale), top_y + int(140 * scale)],
        14: [center_x + int(80 * scale), top_y + int(140 * scale)],
        15: [center_x - int(110 * scale), top_y + int(200 * scale)],
        16: [center_x + int(110 * scale), top_y + int(200 * scale)],
        23: [center_x - int(30 * scale), top_y + int(250 * scale)],
        24: [center_x + int(30 * scale), top_y + int(250 * scale)],
        25: [center_x - int(35 * scale), top_y + int(360 * scale)],
        26: [center_x + int(35 * scale), top_y + int(360 * scale)],
        27: [center_x - int(35 * scale), top_y + int(470 * scale)],
        28: [center_x + int(35 * scale), top_y + int(470 * scale)],
    }
    
    fallback_3d = {
        idx: [float((pt[0] - center_x) / 100.0), float((h / 2 - pt[1]) / 100.0), 0.0]
        for idx, pt in fallback_2d.items()
    }
    return fallback_2d, fallback_3d


def draw_skeleton_on_image(img, keypoints, line_thickness=3, accent_hex="#ef4444", glow_intensity=50):
    """Draws skeletal links and joint markers with custom aesthetic properties."""
    connections = [
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
        (11, 23), (12, 24), (23, 24), (23, 25), (25, 27),
        (24, 26), (26, 28)
    ]
    
    accent_bgr = hex_to_bgr(accent_hex)
    overlay = img.copy()

    # Draw lines
    for p1, p2 in connections:
        if p1 in keypoints and p2 in keypoints:
            pt1 = tuple(keypoints[p1])
            pt2 = tuple(keypoints[p2])
            cv2.line(overlay, pt1, pt2, accent_bgr, line_thickness)

    # Draw joints
    joint_radius = max(2, line_thickness + 3)
    for joint, pt in keypoints.items():
        cv2.circle(overlay, tuple(pt), joint_radius + 2, (255, 255, 255), -1)
        cv2.circle(overlay, tuple(pt), joint_radius, accent_bgr, -1)

    # Apply customizable glow blur level
    if glow_intensity > 0:
        blur_ksize = (glow_intensity if glow_intensity % 2 != 0 else glow_intensity + 1)
        glow_layer = cv2.GaussianBlur(overlay, (blur_ksize, blur_ksize), 0)
        output = cv2.addWeighted(img, 0.3, cv2.addWeighted(overlay, 0.7, glow_layer, 0.5, 0), 0.7, 0)
    else:
        output = cv2.addWeighted(img, 0.3, overlay, 0.7, 0)

    return output


def analyze_media_multi_phase(file_storage, line_thickness=3, accent_hex="#ef4444", glow_intensity=50):
    """Processes images or sequential video frames using MediaPipe pose tracking."""
    filename = file_storage.filename.lower()
    is_video = filename.endswith(('.mp4', '.mov', '.avi', '.webm', '.mkv'))
    suffix = os.path.splitext(filename)[1] or '.mp4'

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        file_storage.save(temp_file.name)
        temp_path = temp_file.name

    frames_data = []

    try:
        if is_video:
            cap = cv2.VideoCapture(temp_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                total_frames = 30
            
            sample_indices = np.linspace(0, total_frames - 1, num=min(10, total_frames), dtype=int)
            for idx in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                success, frame = cap.read()
                if success and frame is not None:
                    kps, kps_3d = extract_keypoints_mediapipe(frame)
                    frames_data.append({"frame_idx": int(idx), "keypoints": kps, "keypoints_3d": kps_3d, "raw_frame": frame})
            cap.release()
        else:
            frame = cv2.imread(temp_path)
            if frame is None:
                frame = np.zeros((600, 400, 3), dtype=np.uint8)
            kps, kps_3d = extract_keypoints_mediapipe(frame)
            frames_data.append({"frame_idx": 0, "keypoints": kps, "keypoints_3d": kps_3d, "raw_frame": frame})

    except Exception as e:
        print(f"⚠️ Multi-Phase Error: {e}")
        dummy_frame = np.zeros((600, 400, 3), dtype=np.uint8)
        kps, kps_3d = extract_keypoints_mediapipe(dummy_frame)
        frames_data.append({"frame_idx": 0, "keypoints": kps, "keypoints_3d": kps_3d, "raw_frame": dummy_frame})

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    if not frames_data:
        dummy_frame = np.zeros((600, 400, 3), dtype=np.uint8)
        kps, kps_3d = extract_keypoints_mediapipe(dummy_frame)
        frames_data.append({"frame_idx": 0, "keypoints": kps, "keypoints_3d": kps_3d, "raw_frame": dummy_frame})

    reachback_frame = min(frames_data, key=lambda f: f["keypoints"][16][0] - f["keypoints"][12][0])
    release_idx = len(frames_data) // 2
    release_frame = frames_data[release_idx]
    follow_frame = frames_data[-1]

    annotated_frame = draw_skeleton_on_image(
        release_frame["raw_frame"],
        release_frame["keypoints"],
        line_thickness=line_thickness,
        accent_hex=accent_hex,
        glow_intensity=glow_intensity
    )
    _, buffer = cv2.imencode('.jpg', annotated_frame)
    annotated_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')

    phase_keypoints = {
        "reach_back": reachback_frame["keypoints"],
        "release": release_frame["keypoints"],
        "follow_through": follow_frame["keypoints"]
    }

    return phase_keypoints, annotated_b64, release_frame["keypoints_3d"]


def extract_features_from_keypoints(keypoints):
    """Extracts kinematic joint angles and positions."""
    knee_angle = calculate_angle(keypoints[24], keypoints[26], keypoints[28])
    elbow_angle = calculate_angle(keypoints[12], keypoints[14], keypoints[16])
    shoulder_tilt = abs(keypoints[12][1] - keypoints[11][1])
    hip_tilt = abs(keypoints[24][1] - keypoints[23][1])
    rotation_delta = abs(shoulder_tilt - hip_tilt)
    wrist_elevation = keypoints[16][1] - keypoints[14][1]
    follow_angle = calculate_angle(keypoints[11], keypoints[12], keypoints[16])
    wrist_offset = abs(keypoints[16][0] - keypoints[14][0])

    return {
        'knee_angle': knee_angle,
        'elbow_angle': elbow_angle,
        'shoulder_tilt': shoulder_tilt,
        'hip_tilt': hip_tilt,
        'rotation_delta': rotation_delta,
        'wrist_elevation': wrist_elevation,
        'follow_angle': follow_angle,
        'wrist_offset': wrist_offset
    }


def compute_ml_kinematic_scores(phase_keypoints, throw_type="backhand"):
    """Computes biomechanics scores against throw-type targets or via loaded ML model."""
    kps = phase_keypoints["release"]
    feats = extract_features_from_keypoints(kps)

    knee_angle = feats['knee_angle']
    elbow_angle = feats['elbow_angle']
    rotation_delta = feats['rotation_delta']
    wrist_elevation = feats['wrist_elevation']
    follow_angle = feats['follow_angle']

    # Optional inference with ML Model if present
    if ml_model is not None:
        try:
            feature_vector = np.array([[knee_angle, elbow_angle, feats['shoulder_tilt'], 
                                        feats['hip_tilt'], rotation_delta, wrist_elevation, 
                                        follow_angle, feats['wrist_offset']]])
            # Assuming model predicts overall score directly or array of subscores
            predicted_score = ml_model.predict(feature_vector)[0]
            if isinstance(predicted_score, (int, float, np.number)):
                overall_score = round(float(predicted_score), 1)
        except Exception:
            ml_model_active = False

    if throw_type == "forehand":
        stance_score = max(10, min(100, 100 - abs(140 - knee_angle) * 1.5))
        reach_score = max(10, min(100, 100 - abs(130 - elbow_angle) * 1.8))
        core_score = max(10, min(100, 100 - rotation_delta * 2.0))
        release_score = max(10, min(100, 100 - abs(wrist_elevation + 15) * 2.2))
        follow_score = max(10, min(100, 100 - abs(135 - follow_angle) * 1.5))
    elif throw_type in ["hammer", "scoober"]:
        stance_score = max(10, min(100, 100 - abs(150 - knee_angle) * 1.4))
        reach_score = max(10, min(100, 100 - abs(120 - elbow_angle) * 1.6))
        core_score = max(10, min(100, 100 - rotation_delta * 3.0))
        release_score = max(10, min(100, 100 - abs(wrist_elevation - 40) * 1.8))
        follow_score = max(10, min(100, 100 - abs(160 - follow_angle) * 1.4))
    else:  # Default Backhand
        stance_score = max(10, min(100, 100 - abs(132 - knee_angle) * 1.8))
        reach_score = max(10, min(100, 100 - abs(165 - elbow_angle) * 1.5))
        core_score = max(10, min(100, 100 - rotation_delta * 2.5))
        release_score = max(10, min(100, 100 - abs(wrist_elevation) * 2.0))
        follow_score = max(10, min(100, 100 - abs(150 - follow_angle) * 1.6))

    metrics = {
        "Stance & Balance": round(float(stance_score), 1),
        "Reach-back Extension": round(float(reach_score), 1),
        "Core Rotation": round(float(core_score), 1),
        "Release Plane": round(float(release_score), 1),
        "Follow-through": round(float(follow_score), 1),
    }

    overall_score = round(sum(metrics.values()) / len(metrics))
    return overall_score, metrics


def generate_custom_drill_routine(metrics):
    lowest_metric = min(metrics, key=metrics.get)
    drill_info = DRILL_DATABASE.get(lowest_metric, DRILL_DATABASE["Release Plane"])
    
    return {
        "targeted_subscore": lowest_metric,
        "subscore_value": metrics[lowest_metric],
        "drill_title": drill_info["title"],
        "duration": drill_info["duration"],
        "description": drill_info["description"],
        "steps": drill_info["steps"],
        "video_url": drill_info["video_url"]
    }


# ==========================================
# FLASK ROUTES
# ==========================================

@app.route("/")
def serve_landing():
    return send_from_directory(".", "landing.html")


@app.route("/app")
def serve_app():
    return send_from_directory(".", "index.html")


@app.route("/api/micro-drills", methods=["GET"])
def get_micro_drills():
    return jsonify({"micro_drills": MICRO_DRILLS})


@app.route("/api/score-throw", methods=["POST"])
def score_throw():
    uploaded_file = (
        request.files.get("media")
        or request.files.get("video")
        or request.files.get("image")
    )
    throw_type = request.form.get("throw_type", "backhand").lower()

    # Dynamic aesthetic parameters passed from Theme Engine
    line_thickness = int(request.form.get("line_thickness", 3))
    accent_hex = request.form.get("accent_hex", "#ef4444")
    glow_intensity = int(request.form.get("glow_intensity", 50))

    if not uploaded_file:
        return jsonify({"error": "No file uploaded"}), 400

    phase_kps, pose_b64, keypoints_3d = analyze_media_multi_phase(
        uploaded_file,
        line_thickness=line_thickness,
        accent_hex=accent_hex,
        glow_intensity=glow_intensity
    )
    overall_score, metrics = compute_ml_kinematic_scores(phase_kps, throw_type)
    custom_drill = generate_custom_drill_routine(metrics)

    prompt = f"""
    You are an elite, highly critical Ultimate Frisbee biomechanics coach.
    Throw Type Analyzed: {throw_type.upper()}
    
    The throw was evaluated across three distinct phases (Reach-back, Release, Follow-through).
    Sub-scores (out of 100):
    - Stance & Balance: {metrics['Stance & Balance']}
    - Reach-back Extension: {metrics['Reach-back Extension']}
    - Core Rotation: {metrics['Core Rotation']}
    - Release Plane: {metrics['Release Plane']}
    - Follow-through: {metrics['Follow-through']}
    - Overall Score: {overall_score}/100

    Provide direct, candid feedback specific to a {throw_type} mechanics style.
    Provide 2 specific mechanical tweaks to implement on the next try.
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction="You are a strict Ultimate Frisbee biomechanics instructor.",
                temperature=0.4
            )
        )
        ai_feedback = response.text
    except Exception as e:
        print(f"❌ Score Throw Error: {e}")
        ai_feedback = (
            f"Sub-Score Breakdown ({throw_type.capitalize()} Throw):\n"
            + "\n".join([f"- {k}: {v}/100" for k, v in metrics.items()])
            + f"\n\nOverall Mechanics Score: {overall_score}/100."
        )

    return jsonify({
        "overall_score": overall_score,
        "metrics": metrics,
        "ai_feedback": ai_feedback,
        "custom_drill": custom_drill,
        "pose_image": pose_b64,
        "keypoints_3d": keypoints_3d,
        "throw_type": throw_type
    })


@app.route("/api/chat", methods=["POST"])
def ai_chat():
    data = request.get_json() or {}
    messages = data.get("messages", [])

    formatted_contents = []
    for msg in messages:
        role = "user" if msg.get("role") == "user" else "model"
        formatted_contents.append({
            "role": role,
            "parts": [{"text": msg.get("content", "")}]
        })

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=formatted_contents,
            config=types.GenerateContentConfig(
                system_instruction="You are DiscSight AI Coach, an expert Ultimate Frisbee instructor.",
                temperature=0.5
            )
        )
        reply = response.text
    except Exception as e:
        print(f"❌ Chat Error: {e}")
        reply = "Unable to process query at the moment."

    return jsonify({"response": reply})


@app.route("/api/compare-pro", methods=["POST"])
def compare_pro():
    try:
        user_file = request.files.get("user_media")
        pro_file = request.files.get("pro_media")
        throw_type = request.form.get("throw_type", "Backhand")

        line_thickness = int(request.form.get("line_thickness", 3))
        accent_hex = request.form.get("accent_hex", "#ef4444")
        glow_intensity = int(request.form.get("glow_intensity", 50))

        if not user_file:
            return jsonify({"error": "User throw file is required"}), 400

        _, user_pose_b64, _ = analyze_media_multi_phase(
            user_file,
            line_thickness=line_thickness,
            accent_hex=accent_hex,
            glow_intensity=glow_intensity
        )

        if pro_file:
            _, pro_pose_b64, _ = analyze_media_multi_phase(
                pro_file,
                line_thickness=line_thickness,
                accent_hex=accent_hex,
                glow_intensity=glow_intensity
            )
        else:
            pro_pose_b64 = user_pose_b64

        return jsonify({
            "status": "success",
            "pro_pose_image": pro_pose_b64,
            "user_pose_image": user_pose_b64,
            "analysis_text": f"Your release angle for {throw_type} matches closely with top elite performers!"
        }), 200

    except Exception as e:
        print(f"❌ Pro Compare Error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
