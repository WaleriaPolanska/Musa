import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_string("""
<mujoco>
  <worldbody>
    <body pos="0 0 0">
      <geom type="sphere" size=".3"/>
    </body>
  </worldbody>
</mujoco>
""")

data = mujoco.MjData(model)


def key_callback(key):
    print("KEY CODE =", key, flush=True)


print("Нажимай клавиши ВНУТРИ окна viewer.")
print("← → ↑ ↓, SPACE, A, D")
print("")

with mujoco.viewer.launch_passive(
    model,
    data,
    key_callback=key_callback,
) as viewer:

    while viewer.is_running():
        viewer.sync()
