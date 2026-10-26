import mujoco
import mujoco.viewer

m = mujoco.MjModel.from_xml_string("""
<mujoco>
  <worldbody>
    <body>
      <geom type="sphere" size=".2"/>
    </body>
  </worldbody>
</mujoco>
""")

d = mujoco.MjData(m)

def key_callback(keycode):
    print("KEY:", keycode)

print("Нажимай клавиши в окне viewer.")

with mujoco.viewer.launch_passive(
    m,
    d,
    key_callback=key_callback,
) as viewer:
    while viewer.is_running():
        viewer.sync()
