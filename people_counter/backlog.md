## **Parked / To-Revisit Items**

1. **Gray-zone hysteresis**

   * Introduce a pixel buffer zone around the curve to prevent miscounts due to small fluctuations (“flapping”).
   * Currently disabled in trajectory tests for simplicity.

2. **Auto-orientation check**

   * Currently implemented and tested for trajectory CSVs.
   * Needs integration into the live video tracker pipeline.

3. **Dynamic camera flip detection**

   * Detect if the camera is physically reoriented (rotated, mirrored, flipped) during operation.
   * Could require dynamic orientation monitoring or short calibration steps.

4. **World-space mapping / real-world coordinates**

   * Optional enhancement to make counting independent of camera orientation and perspective.
   * Useful for multi-camera setups or if the camera is moved frequently.

