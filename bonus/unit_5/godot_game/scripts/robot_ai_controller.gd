extends AIController3D
class_name RobotAIController

@export var raycast_sensors: Array[RayCastSensor3D]
@onready var player = get_parent() as Player

@export var chest: Chest
@export var key: Key
@export var lever: Lever
@export var raft: Raft

var mouse_movement: Vector2
var previous_mouse_movement: Vector2

var steps_without_lever_pulled: int


func get_obs() -> Dictionary:
	var observations: Array[float] = []
	for raycast_sensor in raycast_sensors:
		observations.append_array(raycast_sensor.get_observation())

	var level_size = 16.0

	var chest_local = to_local(chest.global_position)
	var chest_direction = chest_local.normalized()
	var chest_distance = clampf(chest_local.length(), 0.0, level_size)
	
	var lever_local = to_local(lever.global_position)
	var lever_direction = lever_local.normalized()
	var lever_distance = clampf(lever_local.length(), 0.0, level_size)
		
	var key_local = to_local(key.global_position)
	var key_direction = key_local.normalized()
	var key_distance = clampf(key_local.length(), 0.0, level_size)
	
	var raft_local = to_local(raft.global_position)
	var raft_direction = raft_local.normalized()
	var raft_distance = clampf(raft_local.length(), 0.0, level_size)
	
	var player_speed = player.global_basis.inverse() * player.velocity.limit_length(5.0) / 5.0


	(
		observations
		. append_array(
			[
				chest_direction.x,
				chest_direction.y,
				chest_direction.z,
				chest_distance,
				lever_direction.x,
				lever_direction.y,
				lever_direction.z,
				lever_distance,
				key_direction.x,
				key_direction.y,
				key_direction.z,
				key_distance,
				raft_direction.x,
				raft_direction.y,
				raft_direction.z,
				raft_distance,
				raft.movement_direction_multiplier,
				float(player._is_lever_pulled),
				float(player._is_chest_opened),
				float(player._is_key_collected),
				float(player.is_on_floor()),
				player_speed.x,
				player_speed.y,
				player_speed.z,
			]
		)
	)
	return {"obs": observations}


func get_reward() -> float:
	return reward


func _physics_process(delta: float) -> void:
	# Reset on timeout, this is implemented in parent class to set needs_reset to true,
	# we are re-implementing here to call player.game_over() that handles the game reset.
	n_steps += 1
	if n_steps > reset_after:
		player.game_over()

	# In training or onnx inference modes, this method will be called by sync node with actions provided,
	# For expert demo recording mode, it will be called without any actions (as we set the actions based on human input),
	# For human control mode the method will not be called, so we call it here without any actions provided.
	if control_mode == ControlModes.HUMAN:
		set_action()

	# Reset the game faster if the lever is not pulled.
	steps_without_lever_pulled += 1
	if steps_without_lever_pulled > 200 and (not player._is_lever_pulled):
		player.game_over()


func reset():
	super.reset()
	steps_without_lever_pulled = 0

# Defines the actions for the AI agent ("size": 2 means 2 floats for this action)
func get_action_space() -> Dictionary:
	return {
		"movement": {"size": 2, "action_type": "continuous"},
		"rotation": {"size": 1, "action_type": "continuous"},
		"jump": {"size": 1, "action_type": "continuous"},
		"use_action": {"size": 1, "action_type": "continuous"}
	}


# We return the action values in the same order as defined in get_action_space() (important), but all in one array
# For actions of size 1, we return 1 float in the array, for size 2, 2 floats in the array, etc.
# set_action is called just before get_action by the sync node, so we can read the newly set values
func get_action():
	return [
		# "movement" action values
		player.requested_movement.x,
		player.requested_movement.y,
		# "rotation" action value
		player.requested_rotation.x,
		# "jump" action value (-1 if not requested, 1 if requested)
		-1.0 + 2.0 * float(player.jump_requested),
		# "use_action" action value (-1 if not requested, 1 if requested)
		-1.0 + 2.0 * float(player.use_action_requested)
	]


# Here we set human control and AI control actions to the robot
func set_action(action = null) -> void:
	# If there's no action provided, it means that AI is not controlling the robot (human control),
	if not action:
		# Only rotate if the mouse has moved since the last set_action call
		if previous_mouse_movement == mouse_movement:
			mouse_movement = Vector2.ZERO

		player.requested_movement = Input.get_vector(
			"move_left", "move_right", "move_forward", "move_back"
		)
		player.requested_rotation = mouse_movement

		var use_action = Input.is_action_pressed("requested_action")
		var jump = Input.is_action_pressed("requested_jump")

		player.use_action_requested = use_action
		player.jump_requested = jump

		previous_mouse_movement = mouse_movement	
	else:
		# If there is action provided, we set the actions received from the AI agent 
		player.requested_movement = Vector2(action.movement[0], action.movement[1])
		# The agent only rotates the robot along the Y axis, no need to rotate the camera along X axis
		player.requested_rotation = Vector2(action.rotation[0], 0.0)
		player.jump_requested = bool(action.jump[0] > 0)
		player.use_action_requested = bool(action.use_action[0] > 0)


# Record mouse movement for human and demo_record modes
# We don't directly rotate in input to allow for frame skipping (action_repeat setting) which
# will also be applied to the AI agent in training/inference modes.
func _input(event):
	if not (heuristic == "human" or heuristic == "demo_record"):
		return

	if event is InputEventMouseMotion:
		var movement_scale: float = 0.005
		mouse_movement.y = clampf(event.relative.y * movement_scale, -1.0, 1.0)
		mouse_movement.x = clampf(event.relative.x * movement_scale, -1.0, 1.0)
