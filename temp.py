import math
FPS = 50
SCALE = 30.0  # affects how fast-paced the game is, forces should be adjusted as well

MAIN_ENGINE_POWER = 13.0
SIDE_ENGINE_POWER = 0.6

# INITIAL_RANDOM = 1000.0  # Set 1500 to make game harder

# LANDER_POLY = [(-14, +17), (-17, 0), (-17, -10), (+17, -10), (+17, 0), (+14, +17)]
# LEG_AWAY = 20
LEG_DOWN = 18
# LEG_W, LEG_H = 2, 8
# LEG_SPRING_TORQUE = 40

SIDE_ENGINE_HEIGHT = 14
SIDE_ENGINE_AWAY = 12
MAIN_ENGINE_Y_LOCATION = (
    4  # The Y location of the main engine on the body of the Lander.
)

VIEWPORT_W = 600
VIEWPORT_H = 400


def next_state(env, observation, action):
    x_pos, y_pos, x_veloc, y_veloc, lander_angle, angle_veloc, left_leg, right_leg = observation
    # Apply Engine Impulses

    # Tip is a the (X and Y) components of the rotation of the lander.
    tip = (math.sin(lander_angle), math.cos(lander_angle))

    # Side is the (-Y and X) components of the rotation of the lander.
    side = (-tip[1], tip[0])

    # Generate two random numbers between -1/SCALE and 1/SCALE.
    #set to 0 since it is random
    dispersion = [0,0]

    m_power = 0.0
    if action == 2:
        m_power = 1.0

        # 4 is move a bit downwards, +-2 for randomness
        # The components of the impulse to be applied by the main engine.
        ox = (
            tip[0] * (MAIN_ENGINE_Y_LOCATION / SCALE + 2 * dispersion[0])
            + side[0] * dispersion[1]
        )
        oy = (
            -tip[1] * (MAIN_ENGINE_Y_LOCATION / SCALE + 2 * dispersion[0])
            - side[1] * dispersion[1]
        )

        impulse_pos = (x_pos + ox, y_pos + oy)
        self.lander.ApplyLinearImpulse(
            (-ox * MAIN_ENGINE_POWER * m_power, -oy * MAIN_ENGINE_POWER * m_power),
            impulse_pos,
            True,
        )

    if action in [1, 3]:
        # action = 1 is left, action = 3 is right
        direction = action - 2
        s_power = 1.0

        # The components of the impulse to be applied by the side engines.
        ox = tip[0] * dispersion[0] + side[0] * (
            3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE
        )
        oy = -tip[1] * dispersion[0] - side[1] * (
            3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE
        )

        # The constant 17 is a constant, that is presumably meant to be SIDE_ENGINE_HEIGHT.
        # However, SIDE_ENGINE_HEIGHT is defined as 14
        # This casuses the position of the thurst on the body of the lander to change, depending on the orientation of the lander.
        # This in turn results in an orientation depentant torque being applied to the lander.
        impulse_pos = (
            x_pos + ox - tip[0] * 17 / SCALE,
            y_pos + oy + tip[1] * SIDE_ENGINE_HEIGHT / SCALE,
        )
        self.lander.ApplyLinearImpulse(
            (-ox * SIDE_ENGINE_POWER * s_power, -oy * SIDE_ENGINE_POWER * s_power),
            impulse_pos,
            True,
        )

    pos = self.lander.position
    vel = self.lander.linearVelocity

    state = [
        (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
        (pos.y - (self.helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2),
        vel.x * (VIEWPORT_W / SCALE / 2) / FPS,
        vel.y * (VIEWPORT_H / SCALE / 2) / FPS,
        lander_angle,
        20.0 * self.lander.angularVelocity / FPS,
        1.0 if self.legs[0].ground_contact else 0.0,
        1.0 if self.legs[1].ground_contact else 0.0,
    ]
    assert len(state) == 8


    return np.array(state, dtype=np.float32)