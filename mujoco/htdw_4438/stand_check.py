from pathlib import Path

import mujoco
import numpy as np


MODEL_PATH = Path(__file__).resolve().parent / "scene.xml"

STAND_QPOS = {
    "FL_hip_joint": 0.0,
    "FL_thigh_joint": 0.4838,
    "FL_calf_joint": -0.9461,
    "FR_hip_joint": 0.0,
    "FR_thigh_joint": 0.4838,
    "FR_calf_joint": -0.9461,
    "RL_hip_joint": 0.0,
    "RL_thigh_joint": 0.4838,
    "RL_calf_joint": -0.9461,
    "RR_hip_joint": 0.0,
    "RR_thigh_joint": 0.4838,
    "RR_calf_joint": -0.9461,
}

KP = {
    "hip": 18.0,
    "thigh": 28.0,
    "calf": 28.0,
}
KD = {
    "hip": 0.5,
    "thigh": 0.8,
    "calf": 0.8,
}


def gains_for_joint(joint_name: str) -> tuple[float, float]:
    for key in ("hip", "thigh", "calf"):
        if key in joint_name:
            return KP[key], KD[key]
    return 20.0, 0.7


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    stand_key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    if stand_key >= 0:
        mujoco.mj_resetDataKeyframe(model, data, stand_key)
    else:
        mujoco.mj_resetData(model, data)

    target = {}
    for joint_name, q in STAND_QPOS.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise RuntimeError(f"Missing joint: {joint_name}")
        target[joint_id] = q

    steps = int(5.0 / model.opt.timestep)
    max_ctrl = 0.0
    for _ in range(steps):
        for actuator_id in range(model.nu):
            joint_id = model.actuator_trnid[actuator_id, 0]
            joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            qadr = model.jnt_qposadr[joint_id]
            dadr = model.jnt_dofadr[joint_id]
            kp, kd = gains_for_joint(joint_name)
            torque = kp * (target[joint_id] - data.qpos[qadr]) - kd * data.qvel[dadr]
            lo, hi = model.actuator_ctrlrange[actuator_id]
            data.ctrl[actuator_id] = np.clip(torque, lo, hi)
            max_ctrl = max(max_ctrl, abs(float(data.ctrl[actuator_id])))
        mujoco.mj_step(model, data)

    errors = []
    for joint_id, q in target.items():
        qadr = model.jnt_qposadr[joint_id]
        errors.append(float(data.qpos[qadr] - q))

    print(f"model: {MODEL_PATH}")
    print(f"nq={model.nq} nv={model.nv} nu={model.nu} nbody={model.nbody} ngeom={model.ngeom}")
    print(f"final_base_z={data.qpos[2]:.4f} m")
    print(f"joint_error_l2={np.linalg.norm(errors):.6f} rad")
    print(f"max_abs_ctrl={max_ctrl:.3f} Nm")
    print(f"contacts={data.ncon}")
    print(f"fallen={data.qpos[2] < 0.10}")


if __name__ == "__main__":
    main()
