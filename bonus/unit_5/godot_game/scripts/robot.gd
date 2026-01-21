extends CharacterBody3D
class_name Player

## Node used to rotate the camera around in X axis
@export var camera_x_rotation: CameraXRotation
## Node with player animations
@export var animation_player: AnimationPlayer
## Visual node, it is updated to face the last movement direction
@export var robot_visual: Node3D
## Movement speed of the robot
@export var movement_speed := 4.0
## Rotation speed of the robot/camera
@export var rotation_speed := 12.0
## Speed at which the visual node direction is updated
@export var visual_rotation_update_speed := 10.0
@export var jump_velocity = 2.5
@export var damping := 100.0

var _is_lever_pulled: bool
var _is_key_collected: bool
var _is_chest_opened: bool

## Used for setting the direction that the robot looks at. Doesn't affect movement direction or collisions.
var _robot_visual_direction: Vector3

@onready var _initial_transform := transform

@onready var _ai_controller := $AIController3D
@onready var _level_manager := get_parent() as LevelManager

#region Set by AIController
var requested_movement: Vector2
var requested_rotation: Vector2
var jump_requested: bool
var use_action_requested: bool
#endregion


func _ready():
	_capture_mouse_if_human_control_mode()


func _physics_process(delta):
	# Set to true by the sync node when reset is requested from Python (starting training, evaluation, etc.)
	if _ai_controller.needs_reset:
		_level_manager.reset_all_resetables()
		_ai_controller.reset()

	_reset_on_fell_down()

	_apply_gravity(delta)
	_process_jump()

	_process_rotation(delta)
	_process_movement(delta)
	_apply_damping(delta)

	_update_visual_rotation(delta)
	move_and_slide()


func _process_rotation(delta):
	if requested_rotation:
		rotation.y -= requested_rotation.x * rotation_speed * delta
		rotation.y = wrapf(rotation.y, -TAU, TAU)
		camera_x_rotation.rotate_camera_x(requested_rotation.y * rotation_speed, delta)
		_robot_visual_direction = Vector3.ZERO


func _apply_gravity(delta):
	if not is_on_floor():
		velocity += get_gravity() * delta


func _process_jump():
	var jump: bool
	jump = jump_requested
	if jump and is_on_floor():
		velocity.y = jump_velocity


func _process_movement(delta):
	var input_dir: Vector2 = requested_movement.limit_length(1.0)
	var move_direction: Vector3 = global_basis * Vector3(input_dir.x, 0, input_dir.y)

	if input_dir:
		velocity += move_direction * movement_speed * delta
		_robot_visual_direction = move_direction
		animation_player.play("walking")
	else:
		animation_player.play("idle")


func _apply_damping(delta):
	velocity.x /= 1.0 + damping * delta
	velocity.z /= 1.0 + damping * delta


func _update_visual_rotation(delta):
	var direction: Vector3 = _robot_visual_direction

	if direction:
		var target_basis := Basis.looking_at(direction)
		robot_visual.global_basis = robot_visual.global_basis.orthonormalized().slerp(
			target_basis, delta * visual_rotation_update_speed
		)


## Designed to be called in the _ready() method
func _capture_mouse_if_human_control_mode():
	# Waits for the tree to load, so the sync node can set the AIController mode before we read it.
	await get_tree().root.ready
	var mode = _ai_controller.control_mode

	if (
		mode == RobotAIController.ControlModes.HUMAN
		or mode == RobotAIController.ControlModes.RECORD_EXPERT_DEMOS
	):
		Input.mouse_mode = Input.MOUSE_MODE_CAPTURED


func game_over(reward = 0.0):
	_ai_controller.reset()
	_ai_controller.reward = reward

	_level_manager.reset_all_resetables()
	_ai_controller.done = true


func reset():
	transform = _initial_transform
	velocity = Vector3.ZERO
	_robot_visual_direction = Vector3.ZERO
	camera_x_rotation.reset()
	_is_chest_opened = false
	_is_lever_pulled = false
	_is_key_collected = false


func _reset_on_fell_down():
	if global_position.y < -5.0:
		game_over(-1.0)


func _on_chest_opened():
	_is_chest_opened = true
	_ai_controller.reward += 1
	print("chest opened")
	await get_tree().create_timer(1.0, true, true).timeout
	game_over()


func _on_lever_activated():
	_is_lever_pulled = true
	_ai_controller.reward += 1
	print("lever pulled")


func _on_key_collected():
	_is_key_collected = true
	_ai_controller.reward += 1
	print("key collected")
