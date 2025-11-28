# Learnings

## Gymnasium + Box2D error
- Error: `TypeError: in method 'b2RevoluteJoint___SetMotorSpeed', argument 2 of type 'float32'`.
- Cause: Box2D’s SWIG expects a Python float (float32), but receives a NumPy scalar (likely float64) from `np.sign(...)` on Python 3.13 / NumPy 2.x.
- Quick fix (site-packages): cast to `float` when assigning motorSpeed.
- Alternatives:
  - Upgrade `gymnasium[box2d]`.
  - Pin NumPy to `<2.0`.
  - Use Python 3.11 where pybox2d is better tested.

## Preprocessing and frame stacking
- `preprocess(frame)`: RGB → grayscale with OpenCV, resize to (84, 84).
- `stack_frames(frame, stack_size=4)`: maintains a deque of last 4 preprocessed frames, returns `(4, 84, 84)` uint8 stack.

## DQN CNN and training
- CNN expects input `(B, C, 84, 84)` with values in `[0,1]` float32.
- He (Kaiming) initialization applied:
  - `nn.Conv2d`: `kaiming_normal_`.
  - `nn.Linear`: `kaiming_uniform_`.
- `self.modules()`: iterates all registered submodules to apply init uniformly.
- `isinstance(...)` checks target layer types to use appropriate init per type.

## Normal vs Uniform initialization
- Normal: Gaussian distribution; values cluster near mean; occasional larger values.
- Uniform: even probability across fixed bounds `[a, b]`; hard limits.
- Both Kaiming variants aim to keep variance stable for ReLU; choice often preference/baseline.

## DQN loss and train step
- Loss: Smooth L1 (Huber) between `Q(s,a)` and target `r + γ(1-d) * Q_target(s', a*)`.
- Double DQN: `a*` from online net, value from target net.
- Training step: zero grad, backprop, clip grads, optimizer step.

## Practical tips
- Convert stacked uint8 frames to float32 in `[0,1]` before feeding the network.
- Keep target network synced periodically; optionally use soft updates.
- Ensure environment action space is discrete when using DQN.